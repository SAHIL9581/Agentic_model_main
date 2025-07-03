import pandas as pd
import io
import os
import json
import shutil
from typing import List, Dict, Optional
from fastapi import FastAPI, File, UploadFile, HTTPException, WebSocket, WebSocketDisconnect, Form, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from pydantic import BaseModel, Field
import asyncio
import uuid
from fastapi import Body
import re

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from langchain_core.pydantic_v1 import BaseModel as LangchainBaseModel
from langchain.tools import tool

import warnings
warnings.filterwarnings("ignore")

from sklearn.cluster import KMeans

# Initialize FastAPI app
app = FastAPI(
    title="Gemini AI Data Analysis API",
    description="A FastAPI application for data analysis using Google Gemini AI",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Upload directory for Docker
UPLOAD_DIR = "/app/well_files"  # Mounted via docker-compose

# Ensure upload directory exists
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Global storage
UPLOADED_FILES_CONTENT = {}
ACTIVE_DF = None
GOOGLE_API_KEY = None
llm = None
uploaded_files = []  # Track uploaded files and their types

# Pydantic models for API
class FileUploadResponse(BaseModel):
    message: str
    filenames: List[str]
    file_count: int

class AnalysisRequest(BaseModel):
    question: str
    filename: Optional[str] = None

class PandasCommandRequest(BaseModel):
    command: str

class AnalysisResponse(BaseModel):
    response: str
    success: bool
    error: Optional[str] = None

class InitializeRequest(BaseModel):
    api_key: str

class ClusteringRequest(BaseModel):
    filename: str
    columns: List[str]
    n_clusters: int = 3
    random_state: Optional[int] = 42

class ClusteringResponse(BaseModel):
    success: bool
    message: str
    preview: Optional[str] = None
    cluster_counts: Optional[Dict[int, int]] = None
    error: Optional[str] = None

# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def send_personal_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except:
                pass

manager = ConnectionManager()

# --- Tool 1: For the original agentic search ---
class FileInspectionArgs(LangchainBaseModel):
    """Input schema for the file_inspector tool."""
    filename: str = Field(description="The name of the file to inspect from the list of uploaded files.")

@tool(args_schema=FileInspectionArgs)
def file_inspector(filename: str) -> str:
    """  
    Reads and returns the first 5 lines of a specified uploaded file (CSV or TXT).
    This helps in understanding the file's structure and content, especially for identifying columns.
    """
    if filename not in UPLOADED_FILES_CONTENT: 
        return f"Error: File '{filename}' not found."
    content_bytes = UPLOADED_FILES_CONTENT[filename]
    try: 
        content_str = content_bytes.decode('utf-8')
    except UnicodeDecodeError: 
        content_str = content_bytes.decode('latin1')
    try:
        df = pd.read_csv(io.StringIO(content_str), on_bad_lines='skip', sep=None, engine='python', nrows=5)
        return f"Successfully read the first 5 rows of '{filename}':\n\n{df.to_string()}"
    except Exception as e:
        first_lines = "\n".join(content_str.splitlines()[:5])
        return f"Could not parse '{filename}' as a CSV, but here are the first 5 lines:\n\n{first_lines}"

# --- Tool 2: The NEW Code Interpreter for deep analysis ---
class CodeInterpreterArgs(LangchainBaseModel):
    """Input schema for the python_data_analyst tool."""
    pandas_command: str = Field(description="A single-line Python command using the 'df' variable to query the loaded pandas DataFrame.")

@tool(args_schema=CodeInterpreterArgs)
def python_data_analyst(pandas_command: str) -> str:
    """
    Executes a pandas command on the loaded DataFrame 'df' and returns the result.
    Use this for any questions that require filtering, calculating, or specific data lookups.
    Example: To find unique operators, the command would be "df['Operator'].unique()"
    """
    global ACTIVE_DF
    if ACTIVE_DF is None:
        return "Error: No DataFrame is currently loaded. Please load a file first."

    df = ACTIVE_DF # Use the globally loaded DataFrame

    try:
        print(f"⚙️ Executing command: {pandas_command}")
        local_scope = {}
        exec(f"result = {pandas_command}", {'df': df}, local_scope)
        result = local_scope.get('result', "Command executed, but no result was returned.")
        return f"Command execution successful. Result:\n{str(result)}"
    except Exception as e:
        return f"Error executing command: {e}. Please check your pandas command syntax."

# --- Tool 3: Clustering analysis tool ---
class ClusteringToolArgs(LangchainBaseModel):
    filename: str = Field(description="The name of the file to cluster.")
    columns: List[str] = Field(description="List of columns to use for clustering.")
    n_clusters: int = Field(default=3, description="Number of clusters.")

@tool(args_schema=ClusteringToolArgs)
def clustering_analysis_tool(filename: str, columns: List[str], n_clusters: int = 3) -> str:
    """
    Perform KMeans clustering on the specified columns of the uploaded file and return cluster labels and a preview.
    """
    file_path = os.path.join(UPLOAD_DIR, filename)
    if not os.path.exists(file_path):
        return f"Error: File '{filename}' not found."
    try:
        with open(file_path, "rb") as f:
            content_bytes = f.read()
        try:
            content_str = content_bytes.decode('utf-8')
        except UnicodeDecodeError:
            content_str = content_bytes.decode('latin1')
        df = pd.read_csv(io.StringIO(content_str), on_bad_lines='skip', sep=None, engine='python')
        if not all(col in df.columns for col in columns):
            return f"Error: Not all specified columns found in file."
        X = df[columns].select_dtypes(include=['number']).dropna()
        if X.empty:
            return "Error: No numeric data found in specified columns."
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        labels = kmeans.fit_predict(X)
        df_result = X.copy()
        df_result['Cluster'] = labels
        preview = df_result.head(10).to_markdown(index=False)
        counts = dict(pd.Series(labels).value_counts().sort_index())
        return f"Clustering successful.\nCluster counts: {counts}\nPreview:\n{preview}"
    except Exception as e:
        return f"Error during clustering: {e}"

def initialize_llm(api_key: str):
    """Initialize the LLM with the provided API key."""
    global llm, GOOGLE_API_KEY
    GOOGLE_API_KEY = api_key
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash-latest",
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=2,
        google_api_key=GOOGLE_API_KEY
    )

# API Endpoints

@app.get("/", response_class=HTMLResponse)
async def get_homepage():
    """Serve a simple HTML interface for the API."""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Gemini AI Data Analysis</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; }
            .container { max-width: 800px; margin: 0 auto; }
            .section { margin: 20px 0; padding: 20px; border: 1px solid #ddd; border-radius: 5px; }
            input, textarea, button { margin: 5px; padding: 10px; }
            button { background: #007bff; color: white; border: none; border-radius: 3px; cursor: pointer; }
            button:hover { background: #0056b3; }
            #chat-messages { height: 300px; overflow-y: scroll; border: 1px solid #ccc; padding: 10px; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🤖 Gemini AI Data Analysis API</h1>
            
            <div class="section">
                <h2>🔑 Initialize API Key</h2>
                <input type="password" id="api-key" placeholder="Enter your Google AI API key">
                <button onclick="initializeAPI()">Initialize</button>
            </div>

            <div class="section">
                <h2>📁 Upload Files</h2>
                <input type="file" id="file-upload" multiple accept=".csv,.txt">
                <button onclick="uploadFiles()">Upload Files</button>
                <div id="upload-status"></div>
            </div>

            <div class="section">
                <h2>💬 Chat with AI</h2>
                <textarea id="question" placeholder="Ask a question about your data..." rows="3" cols="50"></textarea>
                <button onclick="askQuestion()">Ask Question</button>
                <div id="chat-messages"></div>
            </div>

            <div class="section">
                <h2>📊 File Analysis</h2>
                <button onclick="analyzeFiles()">Analyze All Files</button>
                <div id="analysis-result"></div>
            </div>
        </div>

        <script>
            async function initializeAPI() {
                const apiKey = document.getElementById('api-key').value;
                if (!apiKey) {
                    alert('Please enter an API key');
                    return;
                }
                
                try {
                    const response = await fetch('/initialize', {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify({api_key: apiKey})
                    });
                    const result = await response.json();
                    alert(result.message);
                } catch (error) {
                    alert('Error initializing API: ' + error);
                }
            }

            async function uploadFiles() {
                const fileInput = document.getElementById('file-upload');
                const files = fileInput.files;
                
                if (files.length === 0) {
                    alert('Please select files to upload');
                    return;
                }

                const formData = new FormData();
                for (let file of files) {
                    formData.append('files', file);
                }

                try {
                    const response = await fetch('/upload-files', {
                        method: 'POST',
                        body: formData
                    });
                    const result = await response.json();
                    document.getElementById('upload-status').innerHTML = 
                        `<p>✅ ${result.message}</p><p>Files: ${result.filenames.join(', ')}</p>`;
                } catch (error) {
                    alert('Error uploading files: ' + error);
                }
            }

            async function askQuestion() {
                const question = document.getElementById('question').value;
                if (!question) {
                    alert('Please enter a question');
                    return;
                }

                const messagesDiv = document.getElementById('chat-messages');
                messagesDiv.innerHTML += `<p><strong>You:</strong> ${question}</p>`;

                try {
                    const response = await fetch('/chat', {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify({question: question})
                    });
                    const result = await response.json();
                    messagesDiv.innerHTML += `<p><strong>AI:</strong> ${result.response}</p>`;
                    messagesDiv.scrollTop = messagesDiv.scrollHeight;
                } catch (error) {
                    messagesDiv.innerHTML += `<p><strong>Error:</strong> ${error}</p>`;
                }
            }

            async function analyzeFiles() {
                try {
                    const response = await fetch('/analyze-files');
                    const result = await response.json();
                    document.getElementById('analysis-result').innerHTML = 
                        `<p><strong>Analysis:</strong> ${result.response}</p>`;
                } catch (error) {
                    document.getElementById('analysis-result').innerHTML = 
                        `<p><strong>Error:</strong> ${error}</p>`;
                }
            }
        </script>
    </body>
    </html>
    """

@app.post("/initialize")
async def initialize_api(request: InitializeRequest):
    """Initialize the API with Google AI API key."""
    try:
        initialize_llm(request.api_key)
        return {"message": "API initialized successfully!", "success": True}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to initialize API: {str(e)}")

@app.post("/upload-files", response_model=FileUploadResponse)
async def upload_files(files: List[UploadFile] = File(...), file_type: Optional[str] = Form(None)):
    """Upload multiple files for analysis and save to disk. file_type is optional."""
    global uploaded_files
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")
    uploaded_filenames = []
    for file in files:
        try:
            file_path = os.path.join(UPLOAD_DIR, file.filename)
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            uploaded_files.append({"filename": file.filename, "type": file_type})
            uploaded_filenames.append(file.filename)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Error saving file {file.filename}: {str(e)}")
    return FileUploadResponse(
        message=f"Successfully uploaded {len(files)} files",
        filenames=uploaded_filenames,
        file_count=len(files)
    )

@app.post("/chat", response_model=AnalysisResponse)
async def chat_with_ai(request: AnalysisRequest):
    """Chat with the AI about the uploaded data. Supports multiple questions in a single prompt, and answers them in parallel for speed."""
    global llm, ACTIVE_DF
    if not llm:
        raise HTTPException(status_code=400, detail="API not initialized. Please provide API key first.")
    try:
        raw_qs = request.question.strip()
        questions = [q.strip("-•. 0123456789\t") for q in re.split(r"[\n;]|(?:^|\n)[\-•\d]+[\). ]", raw_qs) if q.strip()]
        if len(questions) == 0:
            return AnalysisResponse(response="❌ No valid question(s) provided.", success=False)
        answers = [None] * len(questions)
        # Prepare tasks for Gemini LLM calls
        async def process_question(idx, q):
            question_lower = q.lower()
            extraction_keywords = ["well name", "latitude", "longitude", "extract", "save to csv", "lat", "lon", "long"]
            filename_match = re.search(r"([\w\- ]+\.(csv|txt))", q, re.IGNORECASE)
            filename_param = filename_match.group(1) if filename_match else None
            # Extraction and file logic (sequential, not parallelized for safety)
            if (all(kw in question_lower for kw in ["well", "lat", "long"]) or (
                any(kw in question_lower for kw in extraction_keywords) and "csv" in question_lower)):
                extraction_result = await extract_well_coordinates(Request({'type': 'http', 'headers': []}, receive=None) if not filename_param else Request({'type': 'http', 'headers': [(b'content-type', b'application/json')]}, receive=None)) if not filename_param else await extract_well_coordinates(Request({'type': 'http', 'headers': [(b'content-type', b'application/json')]}, receive=None))
                if filename_param:
                    extraction_result = await extract_well_coordinates(Request({'type': 'http', 'headers': [(b'content-type', b'application/json')]}, receive=None))
                    extraction_result = await extract_well_coordinates(Request({'type': 'http', 'headers': [(b'content-type', b'application/json')]}, receive=lambda: {'filename': filename_param}))
                else:
                    extraction_result = await extract_well_coordinates(Request({'type': 'http', 'headers': []}, receive=None))
                if extraction_result.get("success"):
                    download_url = f"/download/{extraction_result['filename']}"
                    response_msg = f"✅ Extraction complete! <a href='{download_url}' download>Click here to download the CSV file</a>.<br>Rows extracted: {extraction_result['row_count']}"
                    answers[idx] = f"Q{idx+1}: {q}\n{response_msg}"
                else:
                    answers[idx] = f"Q{idx+1}: {q}\n❌ Extraction failed: {extraction_result.get('message')}"
                return
            if any(word in question_lower for word in ["header", "headers", "column", "columns", "fields", "field names"]):
                if request.filename:
                    file_path = os.path.join(UPLOAD_DIR, request.filename)
                    if not os.path.exists(file_path):
                        answers[idx] = f"Q{idx+1}: {q}\n❌ File not found."
                        return
                    try:
                        with open(file_path, "rb") as f:
                            content_bytes = f.read()
                        try:
                            content_str = content_bytes.decode('utf-8')
                        except UnicodeDecodeError:
                            content_str = content_bytes.decode('latin1')
                        df = pd.read_csv(io.StringIO(content_str), on_bad_lines='skip', sep=None, engine='python', nrows=0)
                        headers = list(df.columns)
                        if headers:
                            headers_md = '\n'.join([f'- `{col}`' for col in headers])
                            answers[idx] = f"Q{idx+1}: {q}\n**Headers in `{request.filename}`:**\n{headers_md}"
                        else:
                            answers[idx] = f"Q{idx+1}: {q}\nNo headers found in the file."
                    except Exception as e:
                        answers[idx] = f"Q{idx+1}: {q}\n❌ Error reading file: {str(e)}"
                else:
                    files = [f for f in os.listdir(UPLOAD_DIR) if os.path.isfile(os.path.join(UPLOAD_DIR, f))]
                    if not files:
                        answers[idx] = f"Q{idx+1}: {q}\n❌ No files uploaded. Please upload files first."
                        return
                    all_headers = ""
                    for filename in files:
                        file_path = os.path.join(UPLOAD_DIR, filename)
                        try:
                            with open(file_path, "rb") as f:
                                content_bytes = f.read()
                            try:
                                content_str = content_bytes.decode('utf-8')
                            except UnicodeDecodeError:
                                content_str = content_bytes.decode('latin1')
                            df = pd.read_csv(io.StringIO(content_str), on_bad_lines='skip', sep=None, engine='python', nrows=0)
                            headers = list(df.columns)
                            if headers:
                                headers_md = '\n'.join([f'- `{col}`' for col in headers])
                                all_headers += f"\n**Headers in `{filename}`:**\n{headers_md}\n"
                            else:
                                all_headers += f"\n**Headers in `{filename}`:** No headers found.\n"
                        except Exception as e:
                            all_headers += f"\n**Headers in `{filename}`:** Error reading file: {str(e)}\n"
                    answers[idx] = f"Q{idx+1}: {q}\n{all_headers}"
                return
            elif any(word in question_lower for word in ["summary", "summarize", "overview", "describe", "info", "information", "structure", "about the data"]):
                if request.filename:
                    file_path = os.path.join(UPLOAD_DIR, request.filename)
                    if not os.path.exists(file_path):
                        answers[idx] = f"Q{idx+1}: {q}\n❌ File not found."
                        return
                    try:
                        with open(file_path, "rb") as f:
                            content_bytes = f.read()
                        try:
                            content_str = content_bytes.decode('utf-8')
                        except UnicodeDecodeError:
                            content_str = content_bytes.decode('latin1')
                        df = pd.read_csv(io.StringIO(content_str), on_bad_lines='skip', sep=None, engine='python')
                        sample = df.head(10).to_markdown(index=False)
                        columns = ', '.join(df.columns)
                        prompt = f"""
                        You are a helpful data analyst. Given the following data sample and columns, describe in 20-40 words what this data is about, as if explaining to a non-technical user. Do not just list columns, but infer the likely subject and purpose of the data.

                        **File:** {request.filename}
                        **Columns:** {columns}
                        **Sample Data:**
                        {sample}
                        """
                        # Run Gemini call in executor for speed
                        import concurrent.futures
                        loop = asyncio.get_running_loop()
                        response = await loop.run_in_executor(None, lambda: llm.invoke(prompt))
                        answers[idx] = f"Q{idx+1}: {q}\n{response.content.strip()}"
                    except Exception as e:
                        answers[idx] = f"Q{idx+1}: {q}\n❌ Error reading file: {str(e)}"
                else:
                    files = [f for f in os.listdir(UPLOAD_DIR) if os.path.isfile(os.path.join(UPLOAD_DIR, f))]
                    if not files:
                        answers[idx] = f"Q{idx+1}: {q}\n❌ No files uploaded. Please upload files first."
                        return
                    all_summaries = ""
                    for filename in files:
                        file_path = os.path.join(UPLOAD_DIR, filename)
                        try:
                            with open(file_path, "rb") as f:
                                content_bytes = f.read()
                            try:
                                content_str = content_bytes.decode('utf-8')
                            except UnicodeDecodeError:
                                content_str = content_bytes.decode('latin1')
                            df = pd.read_csv(io.StringIO(content_str), on_bad_lines='skip', sep=None, engine='python')
                            sample = df.head(10).to_markdown(index=False)
                            columns = ', '.join(df.columns)
                            prompt = f"""
                            You are a helpful data analyst. Given the following data sample and columns, describe in 20-40 words what this data is about, as if explaining to a non-technical user. Do not just list columns, but infer the likely subject and purpose of the data.

                            **File:** {filename}
                            **Columns:** {columns}
                            **Sample Data:**
                            {sample}
                            """
                            import concurrent.futures
                            loop = asyncio.get_running_loop()
                            response = await loop.run_in_executor(None, lambda: llm.invoke(prompt))
                            all_summaries += f"**{filename}:** {response.content.strip()}\n\n"
                        except Exception as e:
                            all_summaries += f"**{filename}:** Error reading file: {str(e)}\n\n"
                    answers[idx] = f"Q{idx+1}: {q}\n{all_summaries}"
                return
            if any(word in question_lower for word in ["cluster", "clustering", "kmeans", "group data", "find groups"]):
                filename_match = re.search(r"([\w\- ]+\.(csv|txt))", q, re.IGNORECASE)
                filename_param = filename_match.group(1) if filename_match else None
                n_clusters_match = re.search(r"(\d+)\s*clusters?", q)
                n_clusters = int(n_clusters_match.group(1)) if n_clusters_match else 3
                columns_match = re.findall(r"[\"'\[]([\w ,]+)[\"'\]]", q)
                columns = [col.strip() for col in columns_match[0].split(',')] if columns_match else []
                if not filename_param or not columns:
                    answers[idx] = f"Q{idx+1}: {q}\n❌ Please specify the file and columns for clustering (e.g., 'cluster data in data.csv using columns [col1, col2]')."
                    return
                result = clustering_analysis_tool(filename=filename_param, columns=columns, n_clusters=n_clusters)
                answers[idx] = f"Q{idx+1}: {q}\n{result}"
                return
            # Default: use Gemini LLM as before (run in executor for speed)
            if request.filename:
                file_path = os.path.join(UPLOAD_DIR, request.filename)
                if not os.path.exists(file_path):
                    answers[idx] = f"Q{idx+1}: {q}\n❌ File not found."
                    return
                try:
                    with open(file_path, "rb") as f:
                        content_bytes = f.read()
                    try:
                        content_str = content_bytes.decode('utf-8')
                    except UnicodeDecodeError:
                        content_str = content_bytes.decode('latin1')
                    ACTIVE_DF = pd.read_csv(io.StringIO(content_str), on_bad_lines='skip', sep=None, engine='python')
                    max_rows = min(100, len(ACTIVE_DF))
                    data_sample = ACTIVE_DF.head(max_rows).to_markdown(index=False)
                    prompt = f"""
                    You are a helpful data analyst. Only answer questions based on the uploaded data below. If the answer is not present in the data, reply: 'The answer is not available in the uploaded data.'

                    **File:** `{request.filename}`
                    **Sample Data (first {max_rows} rows):**

                    {data_sample}

                    **Question:** {q}

                    Provide your answer in a clear, well-formatted way. Use Markdown for tables and lists. If the question can't be answered from the data, reply: 'The answer is not available in the uploaded data.'
                    """
                    import concurrent.futures
                    loop = asyncio.get_running_loop()
                    response = await loop.run_in_executor(None, lambda: llm.invoke(prompt))
                    formatted_response = f"Q{idx+1}: {q}\n<div style='font-family:monospace; white-space:pre-wrap;'>\n{response.content}\n</div>"
                    answers[idx] = formatted_response
                except Exception as e:
                    answers[idx] = f"Q{idx+1}: {q}\n❌ Error reading file: {str(e)}"
                return
            else:
                files = [f for f in os.listdir(UPLOAD_DIR) if os.path.isfile(os.path.join(UPLOAD_DIR, f))]
                if not files:
                    answers[idx] = f"Q{idx+1}: {q}\n❌ No files uploaded. Please upload files first."
                    return
                previews = ""
                for filename in files:
                    file_path = os.path.join(UPLOAD_DIR, filename)
                    try:
                        with open(file_path, "rb") as f:
                            content_bytes = f.read()
                        try:
                            df = pd.read_csv(io.StringIO(content_bytes.decode('utf-8')), on_bad_lines='skip', sep=None, engine='python')
                        except UnicodeDecodeError:
                            df = pd.read_csv(io.StringIO(content_bytes.decode('latin1')), on_bad_lines='skip', sep=None, engine='python')
                        previews += f"\n--- **{filename}** ---\n" + df.head(10).to_markdown(index=False) + "\n"
                    except Exception as e:
                        previews += f"\n--- **{filename}** ---\nCould not display as table. Error: {e}\n"
                prompt = f"""
                You are a helpful data analyst. Only answer questions based on the uploaded data below. If the answer is not present in the data, reply: 'The answer is not available in the uploaded data.'

                **Samples from uploaded files:**
                {previews}

                **Question:** {q}

                Provide your answer in a clear, well-formatted way. Use Markdown for tables and lists. If the question can't be answered from the data, reply: 'The answer is not available in the uploaded data.'
                """
                import concurrent.futures
                loop = asyncio.get_running_loop()
                response = await loop.run_in_executor(None, lambda: llm.invoke(prompt))
                formatted_response = f"Q{idx+1}: {q}\n<div style='font-family:monospace; white-space:pre-wrap;'>\n{response.content}\n</div>"
                answers[idx] = formatted_response
        # Run all Gemini LLM questions in parallel
        await asyncio.gather(*(process_question(idx, q) for idx, q in enumerate(questions)))
        final_response = "\n\n".join(answers)
        return AnalysisResponse(response=final_response, success=True)
    except Exception as e:
        return AnalysisResponse(response=f"❌ Error: {str(e)}", success=False, error=str(e))

@app.get("/analyze-files", response_model=AnalysisResponse)
async def analyze_all_files():
    """Analyze all uploaded files to identify map data and provide a user-friendly summary."""
    files = [f for f in os.listdir(UPLOAD_DIR) if os.path.isfile(os.path.join(UPLOAD_DIR, f))]
    if not files:
        return AnalysisResponse(
            response="❌ No files uploaded. Please upload files first.",
            success=False,
            error="No files available"
        )
    try:
        map_keywords = ["latitude", "longitude", "lat", "long", "depth"]
        recommendations = []
        detailed_analyses = ""
        for filename in files:
            file_path = os.path.join(UPLOAD_DIR, filename)
            try:
                with open(file_path, "rb") as f:
                    content_bytes = f.read()
                try:
                    df = pd.read_csv(io.StringIO(content_bytes.decode('utf-8')), on_bad_lines='skip', sep=None, engine='python')
                except UnicodeDecodeError:
                    df = pd.read_csv(io.StringIO(content_bytes.decode('latin1')), on_bad_lines='skip', sep=None, engine='python')
                columns = [col for col in df.columns]
                columns_lower = [col.lower() for col in columns]
                found = [kw for kw in map_keywords if any(kw in col for col in columns_lower)]
                if found:
                    recommendations.append(f"✅ **{filename}** likely contains map data (columns: {', '.join(found)})")
                else:
                    recommendations.append(f"❌ **{filename}** does not appear to contain map data columns.")
                # Generate a short, plain-language description using Gemini
                if 'llm' in globals() and llm:
                    sample = df.head(10).to_markdown(index=False)
                    columns_str = ', '.join(columns)
                    prompt = f"""
                    You are a helpful data analyst. Given the following data sample and columns, describe in 20-40 words what this data is about, as if explaining to a non-technical user. Do not just list columns, but infer the likely subject and purpose of the data.

                    **File:** {filename}
                    **Columns:** {columns_str}
                    **Sample Data:**
                    {sample}
                    """
                    try:
                        description = llm.invoke(prompt).content.strip()
                    except Exception as e:
                        description = f"(Could not generate description: {str(e)})"
                else:
                    description = "(AI description not available)"
                detailed_analyses += f"""
                <div style='margin-bottom:1.5em;'>
                <b>File:</b> <code>{filename}</code><br/>
                <b>Description:</b> {description}<br/>
                <b>Rows:</b> {df.shape[0]}, <b>Columns:</b> {df.shape[1]}<br/>
                <b>Column Names:</b> {', '.join(columns)}<br/>
                <b>Sample Data:</b><br/>{df.head(5).to_markdown(index=False)}
                </div>
                """
            except Exception as e:
                recommendations.append(f"⚠️ **{filename}** could not be analyzed: {str(e)}")
                detailed_analyses += f"<div><b>File:</b> <code>{filename}</code><br/>Error: {str(e)}</div>"
        summary = "<br/>".join(recommendations)
        full_response = f"""
        <div style='font-family:monospace; white-space:pre-wrap;'>
        🤖 <b>File Analysis Recommendation:</b><br/><br/>{summary}<br/><br/>{detailed_analyses}
        </div>
        """
        return AnalysisResponse(response=full_response, success=True)
    except Exception as e:
        return AnalysisResponse(response=f"❌ Error: {str(e)}", success=False, error=str(e))

@app.get("/files")
async def list_uploaded_files():
    """List all uploaded files and their types."""
    return {
        "files": uploaded_files,
        "count": len(uploaded_files)
    }

@app.get("/active-dataframe")
async def get_active_dataframe_info():
    """Get information about the currently active DataFrame."""
    global ACTIVE_DF
    
    if ACTIVE_DF is None:
        return {"message": "No active DataFrame", "shape": None, "columns": None}
    
    return {
        "shape": ACTIVE_DF.shape,
        "columns": list(ACTIVE_DF.columns),
        "dtypes": ACTIVE_DF.dtypes.to_dict(),
        "head": ACTIVE_DF.head().to_dict()
    }

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time chat."""
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            message_data = json.loads(data)
            
            if message_data.get("type") == "chat":
                question = message_data.get("question", "")
                if question and llm:
                    response = llm.invoke(question)
                    await manager.send_personal_message(
                        json.dumps({
                            "type": "response",
                            "content": response.content
                        }),
                        websocket
                    )
                else:
                    await manager.send_personal_message(
                        json.dumps({
                            "type": "error",
                            "content": "API not initialized or no question provided"
                        }),
                        websocket
                    )
    except WebSocketDisconnect:
        manager.disconnect(websocket)

@app.post("/extract-well-coordinates")
async def extract_well_coordinates(request: Request):
    """Extract well name, latitude, and longitude from uploaded files (CSV or TXT) and save to a new CSV. If well name is missing, fill with 'Unknown' or 'Well_{row_number}'. If filename is provided, only process that file."""
    data = await request.json() if request.headers.get('content-type', '').startswith('application/json') else {}
    filename_param = data.get('filename') if isinstance(data, dict) else None
    files = [f for f in os.listdir(UPLOAD_DIR) if os.path.isfile(os.path.join(UPLOAD_DIR, f))]
    if not files:
        return {"success": False, "message": "No files uploaded."}
    if filename_param:
        files = [f for f in files if f.lower() == filename_param.lower()]
        if not files:
            return {"success": False, "message": f"File '{filename_param}' not found."}
    # Possible column name patterns
    well_patterns = ["well", "well name", "name"]
    lat_patterns = ["lat", "latitude"]
    long_patterns = ["lon", "long", "longitude"]
    extracted_rows = []
    for filename in files:
        file_path = os.path.join(UPLOAD_DIR, filename)
        try:
            with open(file_path, "rb") as f:
                content_bytes = f.read()
            try:
                content_str = content_bytes.decode('utf-8')
            except UnicodeDecodeError:
                content_str = content_bytes.decode('latin1')
            df = None
            try:
                df = pd.read_csv(io.StringIO(content_str), on_bad_lines='skip', sep=None, engine='python')
            except Exception:
                if filename.lower().endswith('.txt'):
                    for sep in ['\t', ' ', ',']:
                        try:
                            df = pd.read_csv(io.StringIO(content_str), on_bad_lines='skip', sep=sep, engine='python')
                            if len(df.columns) > 1:
                                break
                        except Exception:
                            continue
            if df is None or len(df.columns) < 2:
                continue
            lower_cols = [c.lower() for c in df.columns]
            well_col = next((c for c in df.columns if any(p in c.lower() for p in well_patterns)), None)
            lat_col = next((c for c in df.columns if any(p in c.lower() for p in lat_patterns)), None)
            long_col = next((c for c in df.columns if any(p in c.lower() for p in long_patterns)), None)
            if lat_col and long_col:
                cols = [well_col, lat_col, long_col] if well_col else [lat_col, long_col]
                subset = df[cols].copy()
                subset = subset.dropna(subset=[lat_col, long_col])
                if not well_col:
                    subset.insert(0, "Well Name", [f"Well_{i+1}" for i in range(len(subset))])
                else:
                    subset[well_col] = subset[well_col].fillna('').astype(str)
                    subset[well_col] = [name if name.strip() and name.lower() != 'nan' else f"Well_{i+1}" for i, name in enumerate(subset[well_col])]
                    subset = subset[[well_col, lat_col, long_col]]
                    subset.columns = ["Well Name", "Latitude", "Longitude"]
                extracted_rows.append(subset)
        except Exception as e:
            continue
    if not extracted_rows:
        return {"success": False, "message": "No files with latitude and longitude columns found or could not parse text files as tables."}
    result_df = pd.concat(extracted_rows, ignore_index=True)
    output_filename = "well_coordinates_extracted.csv"
    output_path = os.path.join(UPLOAD_DIR, output_filename)
    result_df.to_csv(output_path, index=False)
    return {
        "success": True,
        "message": f"Extracted {len(result_df)} rows from {len(extracted_rows)} file(s). Saved to {output_filename}.",
        "filename": output_filename,
        "row_count": len(result_df),
        "columns": ["Well Name", "Latitude", "Longitude"]
    }

@app.get("/download/{filename}")
async def download_file(filename: str):
    file_path = os.path.join(UPLOAD_DIR, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(file_path, filename=filename, media_type='application/octet-stream')

@app.post("/cluster-analysis", response_model=ClusteringResponse)
async def cluster_analysis(request: ClusteringRequest):
    """Perform KMeans clustering on specified columns of an uploaded file."""
    file_path = os.path.join(UPLOAD_DIR, request.filename)
    if not os.path.exists(file_path):
        return ClusteringResponse(success=False, message="File not found.", error="File not found.")
    try:
        with open(file_path, "rb") as f:
            content_bytes = f.read()
        try:
            content_str = content_bytes.decode('utf-8')
        except UnicodeDecodeError:
            content_str = content_bytes.decode('latin1')
        df = pd.read_csv(io.StringIO(content_str), on_bad_lines='skip', sep=None, engine='python')
        if not all(col in df.columns for col in request.columns):
            return ClusteringResponse(success=False, message="Not all specified columns found in file.", error="Missing columns.")
        X = df[request.columns].select_dtypes(include=['number']).dropna()
        if X.empty:
            return ClusteringResponse(success=False, message="No numeric data found in specified columns.", error="No numeric data.")
        kmeans = KMeans(n_clusters=request.n_clusters, random_state=request.random_state)
        labels = kmeans.fit_predict(X)
        df_result = X.copy()
        df_result['Cluster'] = labels
        preview = df_result.head(10).to_markdown(index=False)
        counts = dict(pd.Series(labels).value_counts().sort_index())
        return ClusteringResponse(success=True, message="Clustering successful.", preview=preview, cluster_counts=counts)
    except Exception as e:
        return ClusteringResponse(success=False, message=f"Error during clustering: {e}", error=str(e))

@app.post("/extract-well-locations")
async def extract_well_locations():
    """
    Inspect all uploaded files to find columns representing well name/uwi, latitude, and longitude.
    Extract and standardize these columns and save to a new CSV file with columns: uwi, latitude, longitude.
    Handles both CSV and Excel files. Optimized for speed: only reads full file if headers match.
    """
    files = [f for f in os.listdir(UPLOAD_DIR) if os.path.isfile(os.path.join(UPLOAD_DIR, f))]
    if not files:
        return {"success": False, "message": "No files uploaded."}
    # Patterns for column detection
    well_patterns = ["uwi", "well", "well_name", "well name", "name"]
    lat_patterns = ["lat", "latitude"]
    long_patterns = ["lon", "lng", "long", "longitude"]
    found_file = None
    well_col = lat_col = long_col = None
    file_type = None
    # 1. FAST HEADER SCAN
    for filename in files:
        file_path = os.path.join(UPLOAD_DIR, filename)
        try:
            if filename.lower().endswith('.csv'):
                df_head = pd.read_csv(file_path, on_bad_lines='skip', sep=None, engine='python', nrows=0)
                file_type = 'csv'
            elif filename.lower().endswith(('.xls', '.xlsx')):
                df_head = pd.read_excel(file_path, nrows=0)
                file_type = 'excel'
            else:
                continue
            lower_cols = [c.lower() for c in df_head.columns]
            # Prefer 'uwi' as well identifier
            for c in df_head.columns:
                if 'uwi' in c.lower():
                    well_col = c
                    break
            if not well_col:
                for c in df_head.columns:
                    if any(p in c.lower() for p in well_patterns):
                        well_col = c
                        break
            for c in df_head.columns:
                if any(p in c.lower() for p in lat_patterns):
                    lat_col = c
                    break
            for c in df_head.columns:
                if any(p in c.lower() for p in long_patterns):
                    long_col = c
                    break
            if well_col and lat_col and long_col:
                found_file = filename
                break  # Only extract from the first matching file
        except Exception:
            continue
    if not (found_file and well_col and lat_col and long_col):
        return {"success": False, "message": "No file with well, latitude, and longitude columns found."}
    # 2. LOAD ONLY THE MATCHING FILE FULLY
    file_path = os.path.join(UPLOAD_DIR, found_file)
    try:
        if file_type == 'csv':
            df = pd.read_csv(file_path, on_bad_lines='skip', sep=None, engine='python')
        else:
            df = pd.read_excel(file_path)
        subset = df[[well_col, lat_col, long_col]].copy()
        subset = subset.dropna(subset=[lat_col, long_col])
        subset.columns = ["uwi", "latitude", "longitude"]
        output_filename = "well_locations_extracted.csv"
        output_path = os.path.join(UPLOAD_DIR, output_filename)
        subset.to_csv(output_path, index=False)
        return {
            "success": True,
            "message": f"Extracted {len(subset)} rows from file '{found_file}'. Saved to {output_filename}.",
            "filename": output_filename,
            "row_count": len(subset),
            "columns": ["uwi", "latitude", "longitude"]
        }
    except Exception as e:
        return {"success": False, "message": f"Error extracting data: {str(e)}"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=3002)
    