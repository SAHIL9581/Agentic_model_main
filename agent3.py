import pandas as pd
import io
import os
import json
import shutil
import zipfile
from typing import List, Dict, Optional, Any
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

@tool
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

@tool
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

@tool
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

# Utility function to recursively scan folders and summarize CSV files
def scan_folder_structure(base_dir: str) -> list[dict[str, Any]]:
    folder_summary = []
    for root, dirs, files in os.walk(base_dir):
        rel_root = os.path.relpath(root, base_dir)
        for file in files:
            if file.lower().endswith('.csv'):
                file_path = os.path.join(root, file)
                try:
                    df = pd.read_csv(file_path, nrows=0)
                    headers = list(df.columns)
                except Exception:
                    headers = []
                folder_summary.append({
                    "folder": rel_root,
                    "file": file,
                    "path": os.path.relpath(file_path, base_dir),
                    "headers": headers
                })
    return folder_summary

# Function to scan file system for folders and CSV files
def scan_file_system(start_path: str = ".") -> dict[str, Any]:
    """Scan the file system for folders and CSV files."""
    result = {
        "folders": [],
        "csv_files": [],
        "total_folders": 0,
        "total_csv_files": 0
    }
    
    try:
        for root, dirs, files in os.walk(start_path):
            # Add folders
            for dir_name in dirs:
                full_path = os.path.join(root, dir_name)
                rel_path = os.path.relpath(full_path, start_path)
                result["folders"].append({
                    "name": dir_name,
                    "full_path": full_path,
                    "relative_path": rel_path
                })
            
            # Add CSV files
            for file_name in files:
                if file_name.lower().endswith('.csv'):
                    full_path = os.path.join(root, file_name)
                    rel_path = os.path.relpath(full_path, start_path)
                    result["csv_files"].append({
                        "name": file_name,
                        "full_path": full_path,
                        "relative_path": rel_path,
                        "size": os.path.getsize(full_path) if os.path.exists(full_path) else 0
                    })
        
        result["total_folders"] = len(result["folders"])
        result["total_csv_files"] = len(result["csv_files"])
        
    except Exception as e:
        result["error"] = str(e)
    
    return result

# Cache for file data to improve performance
FILE_CACHE = {}

def get_cached_file_data(filename: str) -> Optional[pd.DataFrame]:
    """Get cached file data or load and cache it."""
    if filename in FILE_CACHE:
        return FILE_CACHE[filename]
    
    # First try the main upload directory
    file_path = os.path.join(UPLOAD_DIR, filename)
    if not os.path.exists(file_path):
        # If not found, search in extracted zip directories
        for root, dirs, files in os.walk(UPLOAD_DIR):
            if filename in files:
                file_path = os.path.join(root, filename)
                break
        else:
            return None
    
    try:
        with open(file_path, "rb") as f:
            content_bytes = f.read()
        try:
            content_str = content_bytes.decode('utf-8')
        except UnicodeDecodeError:
            content_str = content_bytes.decode('latin1')
        
        df = pd.read_csv(io.StringIO(content_str), on_bad_lines='skip', sep=None, engine='python')
        FILE_CACHE[filename] = df
        return df
    except Exception:
        return None

def extract_zip_file(zip_path: str, extract_to: str) -> List[str]:
    """Extract a zip file and return list of extracted files."""
    extracted_files = []
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
            extracted_files = zip_ref.namelist()
        return extracted_files
    except Exception as e:
        print(f"Error extracting zip file: {e}")
        return []

def process_uploaded_zip(zip_filename: str) -> Dict[str, Any]:
    """Process an uploaded zip file and extract its contents."""
    zip_path = os.path.join(UPLOAD_DIR, zip_filename)
    if not os.path.exists(zip_path):
        return {"success": False, "message": "Zip file not found"}
    
    try:
        # Create a subdirectory for extracted files
        extract_dir = os.path.join(UPLOAD_DIR, f"extracted_{zip_filename.replace('.zip', '')}")
        os.makedirs(extract_dir, exist_ok=True)
        
        # Extract the zip file
        extracted_files = extract_zip_file(zip_path, extract_dir)
        
        if not extracted_files:
            return {"success": False, "message": "Failed to extract zip file or zip file is empty"}
        
        # Find CSV files in extracted contents
        csv_files = []
        for root, dirs, files in os.walk(extract_dir):
            for file in files:
                if file.lower().endswith('.csv'):
                    csv_files.append(os.path.join(root, file))
        
        return {
            "success": True,
            "message": f"Successfully extracted {len(extracted_files)} files from zip",
            "extracted_files": extracted_files,
            "csv_files": csv_files,
            "extract_dir": extract_dir
        }
    except Exception as e:
        return {"success": False, "message": f"Error processing zip file: {str(e)}"}

# Utility to yield all dataframes from uploaded files, including CSV/Excel inside ZIPs

def iter_uploaded_dataframes():
    files = [f for f in os.listdir(UPLOAD_DIR) if os.path.isfile(os.path.join(UPLOAD_DIR, f))]
    for filename in files:
        file_path = os.path.join(UPLOAD_DIR, filename)
        if filename.lower().endswith('.zip'):
            try:
                with zipfile.ZipFile(file_path, 'r') as zip_ref:
                    for zipinfo in zip_ref.infolist():
                        if zipinfo.filename.lower().endswith('.csv'):
                            with zip_ref.open(zipinfo) as f:
                                try:
                                    df = pd.read_csv(f, on_bad_lines='skip', sep=None, engine='python')
                                    yield (f"{zipinfo.filename} (from {filename})", df)
                                except Exception:
                                    continue
                        elif zipinfo.filename.lower().endswith('.xlsx'):
                            with zip_ref.open(zipinfo) as f:
                                try:
                                    df = pd.read_excel(f)
                                    yield (f"{zipinfo.filename} (from {filename})", df)
                                except Exception:
                                    continue
            except Exception:
                continue
        else:
            df = get_cached_file_data(filename)
            if df is not None:
                yield (filename, df)

# Mapping of known file names to human-readable summaries
FILE_SUMMARIES = {
    'liner.csv': 'This data likely describes the dimensions and characteristics of geological formations (e.g., oil or gas reservoirs) identified by unique well identifiers (UWI), including their depth, size, and any relevant remarks.',
    'cores.csv': 'This dataset likely tracks core samples from oil and gas wells. It records the location, depth intervals, recovery data, and sample descriptions for each core, aiding in reservoir analysis.',
    'perfs.csv': 'This data likely tracks well completion activities in oil and gas wells. It records details like well location (UWI), completion intervals (TOP, BASE), methods, and dates, to monitor well performance and operations.',
    'production tests.csv': 'This dataset records results from oil well production tests. It includes measurements of oil, gas, and water production rates, pressures, and other relevant parameters for various wells and tests.',
    'well_coordinates_extracted.csv': 'This data shows the geographic locations of oil or gas wells. Each row represents a well, identified by a unique name and pinpointed by its latitude and longitude coordinates.',
    'casing.csv': 'This data likely tracks cement usage in oil well casings. It records the well\'s ID, depth, casing size, and the amount of cement used at each depth.',
    'formation tops.txt': 'This data likely represents measured depths (MD) of geological formations. The missing well information suggests it\'s a preliminary or incomplete dataset of formation top depths.',
    'well data.csv': 'This dataset tracks oil and gas well information in Texas, including well location, operator, production details, and dates for various activities like permitting, completion, and abandonment. It likely supports well management and production analysis.',
    'dsts.csv': 'This dataset records details of well tests conducted in oil and gas wells. It includes test type, well location data, pressure and flow measurements, and operational information for each test.'
}

# Mapping of known file names to brief analyses
FILE_BRIEF_ANALYSIS = {
    'liner.csv': 'Contains information on well liners, including UWI (Unique Well Identifier), top and base depths, liner size, and remarks.',
    'cores.csv': 'Provides data on core samples, including UWI, formation name, top and base depths, date of sampling, recovered core length, description, core quality, core type, and remarks.',
    'perfs.csv': 'Details on well perforations, including UWI, source, top and base depths, perforation dates, status, diameter, number of shots, perforation method, completion type, remarks, and formation name.',
    'production tests.csv': 'Records of production tests, including UWI, test type, formation name, top and base depths, test date, oil, gas, and water volumes and units, flowing tubing pressure, casing pressure, static tubing pressure, static casing pressure, bottom hole temperature, bottom hole pressure, choke size, test duration, and remarks.',
    'well_coordinates_extracted.csv': 'Contains well coordinates (latitude and longitude) for a subset of wells.',
    'casing.csv': 'Information on well casing, including UWI, depth, size, amount (often cement in sacks), and remarks.',
    'formation tops.txt': 'A list of formation tops with measured depth (MD) values, but UWIs are mostly missing, making it difficult to link to other datasets.',
    'well data.csv': 'Comprehensive well data including WSN, UWI, well number, well label, operator, historical operator, lease name, lease number, field name, formation at total depth, producing formation, surface and bottom coordinates, datum, county, Texas block, Texas section, section, state, Texas survey, Texas abstract, Texas Railroad District, well remarks, and various well parameters.',
    'dsts.csv': 'Data on Drill Stem Tests (DSTs), including UWI, test type, formation name, top and base depths, date, initial and final hydrostatic pressure, initial and final flowing pressure, initial and final shut-in pressure, bottom hole temperature, bottom hole pressure, choke size, cushion amount and type, open and shut times, and remarks.'
}

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
                <input type="file" id="file-upload" multiple accept=".csv,.txt,.zip">
                <button onclick="uploadFiles()">Upload Files</button>
                <div id="upload-status"></div>
                <p><small>💡 You can upload CSV files, text files, or ZIP files containing CSV data.</small></p>
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
    global uploaded_files, FILE_CACHE
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")
    
    uploaded_filenames = []
    extracted_files = []
    
    for file in files:
        try:
            if file.filename is None:
                continue
            file_path = os.path.join(UPLOAD_DIR, file.filename)
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            uploaded_files.append({"filename": file.filename, "type": file_type})
            uploaded_filenames.append(file.filename)
            
            # Clear cache when new files are uploaded
            if file.filename in FILE_CACHE:
                del FILE_CACHE[file.filename]
            
            # Handle zip files
            if file.filename.lower().endswith('.zip'):
                zip_result = process_uploaded_zip(file.filename)
                if zip_result["success"]:
                    extracted_files.extend(zip_result.get("csv_files", []))
                    # Add extracted CSV files to uploaded files list
                    for csv_file in zip_result.get("csv_files", []):
                        csv_filename = os.path.basename(csv_file)
                        uploaded_files.append({"filename": csv_filename, "type": "extracted_from_zip"})
                        uploaded_filenames.append(csv_filename)
                        
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Error saving file {file.filename}: {str(e)}")
    
    message = f"Successfully uploaded {len(files)} files"
    if extracted_files:
        message += f" and extracted {len(extracted_files)} CSV files from zip"
    
    return FileUploadResponse(
        message=message,
        filenames=uploaded_filenames,
        file_count=len(uploaded_filenames)
    )

@app.post("/chat", response_model=AnalysisResponse)
async def chat_with_ai(request: AnalysisRequest):
    """Chat with the AI about the uploaded data. Optimized for performance."""
    global llm, ACTIVE_DF
    if not llm:
        raise HTTPException(status_code=400, detail="API not initialized. Please provide API key first.")
    try:
        raw_qs = request.question.strip()
        questions = [q.strip("-•. 0123456789\t") for q in re.split(r"[\n;]|(?:^|\n)[\-•\d]+[\). ]", raw_qs) if q.strip()]
        if len(questions) == 0:
            return AnalysisResponse(response="❌ No valid question(s) provided.", success=False)
        answers = [""] * len(questions)
        for idx, q in enumerate(questions):
            question_lower = q.lower()
            # --- Well Location Extraction & Save Handler ---
            extraction_keywords = ["well name", "latitude", "longitude", "extract", "save to csv", "lat", "lon", "long"]
            if (all(kw in question_lower for kw in ["well", "lat", "long"]) or (
                any(kw in question_lower for kw in extraction_keywords) and "csv" in question_lower)):
                extracted_rows = []
                seen = set()
                for fname, df in iter_uploaded_dataframes():
                    if fname in seen:
                        continue
                    seen.add(fname)
                    try:
                        well_patterns = ["uwi", "well", "well name", "name"]
                        lat_patterns = ["lat", "latitude"]
                        long_patterns = ["lon", "long", "longitude"]
                        well_col = next((c for c in df.columns if any(p in c.lower() for p in well_patterns)), None)
                        lat_col = next((c for c in df.columns if any(p in c.lower() for p in lat_patterns)), None)
                        long_col = next((c for c in df.columns if any(p in c.lower() for p in long_patterns)), None)
                        if well_col and lat_col and long_col:
                            subset = df[[well_col, lat_col, long_col]].copy()
                            subset = subset.dropna(subset=[lat_col, long_col])
                            subset.columns = ["uwi", "latitude", "longitude"]
                            extracted_rows.append(subset)
                    except Exception:
                        continue
                if not extracted_rows:
                    answers[idx] = f"Q{idx+1}: {q}\n❌ No files with well, latitude, and longitude columns found."
                    continue
                result_df = pd.concat(extracted_rows, ignore_index=True)
                output_filename = "well_locations_extracted.csv"
                output_path = os.path.join(UPLOAD_DIR, output_filename)
                result_df.to_csv(output_path, index=False)
                table_md = result_df.head(20).to_markdown(index=False)
                download_url = f"/download/{output_filename}"
                answers[idx] = f"Q{idx+1}: {q}\n✅ Extraction complete! <a href='{download_url}' download>Click here to download the CSV file</a>.<br>Rows extracted: {len(result_df)}\n\n{table_md}"
                continue
            # --- Well Location Extraction Handler (table only) ---
            if any(word in question_lower for word in ["well locations", "extract well locations", "well coordinates", "extract all well locations", "extract all well coordinates", "well name, latitude, longitude"]):
                extracted_rows = []
                seen = set()
                for fname, df in iter_uploaded_dataframes():
                    if fname in seen:
                        continue
                    seen.add(fname)
                    try:
                        well_patterns = ["well", "well name", "name", "uwi"]
                        lat_patterns = ["lat", "latitude"]
                        long_patterns = ["lon", "long", "longitude"]
                        well_col = next((c for c in df.columns if any(p in c.lower() for p in well_patterns)), None)
                        lat_col = next((c for c in df.columns if any(p in c.lower() for p in lat_patterns)), None)
                        long_col = next((c for c in df.columns if any(p in c.lower() for p in long_patterns)), None)
                        if lat_col and long_col:
                            cols = [well_col, lat_col, long_col] if well_col else [lat_col, long_col]
                            subset = df[cols].copy()
                            subset = subset.dropna(subset=[lat_col, long_col])
                            if not well_col:
                                subset.insert(0, "Well Name", [f"Well_{i+1}" for i in range(len(subset))])
                                subset.columns = ["Well Name", "Latitude", "Longitude"]
                            else:
                                subset[well_col] = subset[well_col].fillna('').astype(str)
                                subset[well_col] = [name if name.strip() and name.lower() != 'nan' else f"Well_{i+1}" for i, name in enumerate(subset[well_col])]
                                subset = subset[[well_col, lat_col, long_col]]
                                subset.columns = ["Well Name", "Latitude", "Longitude"]
                            extracted_rows.append(subset)
                    except Exception:
                        continue
                if not extracted_rows:
                    answers[idx] = f"Q{idx+1}: {q}\n❌ No files with latitude and longitude columns found."
                    continue
                result_df = pd.concat(extracted_rows, ignore_index=True)
                table_md = result_df.to_markdown(index=False)
                answers[idx] = f"Q{idx+1}: {q}\n\n{table_md}"
                continue
            # --- Brief Analysis Handler ---
            if any(word in question_lower for word in ["brief analysis", "analysis", "analyze", "analyses"]):
                all_analyses = ""
                seen = set()
                for fname, df in iter_uploaded_dataframes():
                    if fname in seen:
                        continue
                    seen.add(fname)
                    base = fname.lower().split(' (from ')[0]
                    desc = FILE_BRIEF_ANALYSIS.get(base, FILE_SUMMARIES.get(base, 'No domain-specific analysis available for this file.'))
                    all_analyses += f"**{fname}:** {desc}\n\n"
                answers[idx] = f"Q{idx+1}: {q}\n{all_analyses}"
                continue
            # --- Header/Column Handler ---
            if any(word in question_lower for word in ["header", "headers", "column", "columns", "fields", "field names"]):
                all_headers = ""
                seen = set()
                for fname, df in iter_uploaded_dataframes():
                    if fname in seen:
                        continue
                    seen.add(fname)
                    try:
                        headers = list(df.columns)
                        if headers:
                            headers_md = '\n'.join([f'- `{col}`' for col in headers])
                            all_headers += f"\n**Headers in `{fname}`:**\n{headers_md}\n"
                        else:
                            all_headers += f"\n**Headers in `{fname}`:** No headers found.\n"
                    except Exception:
                        all_headers += f"\n**Headers in `{fname}`:** Error reading file.\n"
                answers[idx] = f"Q{idx+1}: {q}\n{all_headers}"
                continue
            # --- Summary Handler ---
            elif any(word in question_lower for word in ["summary", "summarize", "overview", "describe", "info", "information", "structure", "about the data"]):
                all_summaries = ""
                seen = set()
                for fname, df in iter_uploaded_dataframes():
                    if fname in seen:
                        continue
                    seen.add(fname)
                    base = fname.lower().split(' (from ')[0]
                    desc = FILE_SUMMARIES.get(base, 'No domain-specific summary available for this file.')
                    all_summaries += f"**{fname}:** {desc}\n\n"
                answers[idx] = f"Q{idx+1}: {q}\n{all_summaries}"
                continue
            # --- Clustering Handler ---
            if any(word in question_lower for word in ["cluster", "clustering", "kmeans", "group data", "find groups"]):
                # Try to extract filename and columns
                filename_match = re.search(r"([\w\- ]+\.(csv|txt))", q, re.IGNORECASE)
                filename_param = filename_match.group(1) if filename_match else None
                n_clusters_match = re.search(r"(\d+)\s*clusters?", q)
                n_clusters = int(n_clusters_match.group(1)) if n_clusters_match else 3
                columns_match = re.search(r"\[(.*?)\]", q)
                if columns_match:
                    columns = [col.strip() for col in columns_match.group(1).split(',')]
                else:
                    columns_match = re.search(r'"([^"]+)"', q) or re.search(r"'([^']+)'", q)
                    columns = [col.strip() for col in columns_match.group(1).split(',')] if columns_match else []

                if not filename_param or not columns:
                    answers[idx] = f"Q{idx+1}: {q}\n❌ Please specify the file and columns for clustering (e.g., 'cluster data in data.csv using columns [col1, col2]')."
                    continue
                # Find the file (including inside ZIPs)
                found = False
                for fname, df in iter_uploaded_dataframes():
                    if fname.lower().startswith(filename_param.lower()):
                        found = True
                        try:
                            if not all(col in df.columns for col in columns):
                                answers[idx] = f"Q{idx+1}: {q}\n❌ Not all specified columns found in file. Available columns: {list(df.columns)}"
                                break
                            from sklearn.cluster import KMeans
                            X = df[columns].select_dtypes(include=['number']).dropna()
                            if X.empty:
                                answers[idx] = f"Q{idx+1}: {q}\n❌ No numeric data found in the specified columns."
                                break
                            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                            labels = kmeans.fit_predict(X)
                            df_result = X.copy()
                            df_result['Cluster'] = labels
                            preview = df_result.head(10).to_markdown(index=False)
                            value_counts = pd.Series(labels).value_counts().sort_index()
                            counts = {int(k): int(v) for k, v in value_counts.items()}
                            answers[idx] = f"Q{idx+1}: {q}\n✅ Clustering complete. Cluster counts: {counts}\n\n{preview}"
                        except Exception as e:
                            answers[idx] = f"Q{idx+1}: {q}\n❌ Error during clustering: {str(e)}"
                        break
                if not found:
                    answers[idx] = f"Q{idx+1}: {q}\n❌ File '{filename_param}' not found."
                continue
            # --- File/Folder Listing Handler ---
            if any(word in question_lower for word in ["list", "folders", "subfolders", "csv files", "directory", "scan", "find files"]):
                try:
                    # List all folders, subfolders, and CSV files with full paths
                    folder_list = []
                    csv_list = []
                    for root, dirs, files in os.walk(UPLOAD_DIR):
                        for d in dirs:
                            rel_path = os.path.relpath(os.path.join(root, d), UPLOAD_DIR)
                            folder_list.append(rel_path)
                        for f in files:
                            if f.lower().endswith('.csv'):
                                full_path = os.path.join(root, f)
                                rel_path = os.path.relpath(full_path, UPLOAD_DIR)
                                csv_list.append((f, os.path.abspath(full_path), rel_path))
                    response_text = f"Q{idx+1}: {q}\n\n**Folders and Subfolders:**\n"
                    if folder_list:
                        for folder in folder_list:
                            response_text += f"- `{folder}`\n"
                    else:
                        response_text += "No folders found.\n"
                    response_text += f"\n**CSV Files:**\n"
                    if csv_list:
                        for name, full_path, rel_path in csv_list:
                            response_text += f"- **{name}**\n  - Full Path: `{full_path}`\n  - Relative Path: `{rel_path}`\n"
                    else:
                        response_text += "No CSV files found.\n"
                    answers[idx] = response_text
                except Exception as e:
                    answers[idx] = f"Q{idx+1}: {q}\n❌ Error listing files/folders: {str(e)}"
                continue
            # --- General Prompt Handler ---
            else:
                previews = ""
                seen = set()
                for fname, df in iter_uploaded_dataframes():
                    if fname in seen:
                        continue
                    seen.add(fname)
                    try:
                        preview = df.head(10).to_markdown(index=False)
                        if preview:
                            previews += f"\n--- **{fname}** ---\n{preview}\n"
                        else:
                            previews += f"\n--- **{fname}** ---\nCould not display as table.\n"
                    except Exception:
                        previews += f"\n--- **{fname}** ---\nCould not read file.\n"
                answers[idx] = f"Q{idx+1}: {q}\n{previews}"
                continue
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
        map_keywords = ["latitude", "longitude", "lat", "long", "depth", "uwi", "well"]
        recommendations = []
        detailed_analyses = ""
        
        for filename in files:
            df = get_cached_file_data(filename)
            if df is None:
                recommendations.append(f"⚠️ **{filename}** could not be analyzed: File not found or unreadable")
                detailed_analyses += f"<div><b>File:</b> <code>{filename}</code><br/>Error: Could not read file</div>"
                continue
            
            columns = list(df.columns)
            columns_lower = [col.lower() for col in columns]
            found = [kw for kw in map_keywords if any(kw in col for col in columns_lower)]
            
            if found:
                recommendations.append(f"✅ **{filename}** likely contains map data (columns: {', '.join(found)})")
            else:
                recommendations.append(f"❌ **{filename}** does not appear to contain map data columns.")
            
            # Generate a short, plain-language description using Gemini
            if llm:
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
                    response = llm.invoke(prompt)
                    description = str(response.content).strip()
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
    """Extract well name, latitude, and longitude from uploaded files (CSV or TXT) and save to a new CSV."""
    files = [f for f in os.listdir(UPLOAD_DIR) if os.path.isfile(os.path.join(UPLOAD_DIR, f))]
    if not files:
        return {"success": False, "message": "No files uploaded."}
    
    # Possible column name patterns
    well_patterns = ["well", "well name", "name", "uwi"]
    lat_patterns = ["lat", "latitude"]
    long_patterns = ["lon", "long", "longitude"]
    extracted_rows = []
    
    for filename in files:
        df = get_cached_file_data(filename)
        if df is None:
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
                well_names = pd.Series([f"Well_{i+1}" for i in range(len(subset))])
                subset.insert(0, "Well Name", well_names)
            else:
                subset[well_col] = subset[well_col].fillna('').astype(str)
                subset[well_col] = [name if name.strip() and name.lower() != 'nan' else f"Well_{i+1}" for i, name in enumerate(subset[well_col])]
                subset = subset[[well_col, lat_col, long_col]]
                subset.columns = ["Well Name", "Latitude", "Longitude"]
            
            extracted_rows.append(subset)
    
    if not extracted_rows:
        return {"success": False, "message": "No files with latitude and longitude columns found."}
    
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
        df = get_cached_file_data(request.filename)
        if df is None:
            return ClusteringResponse(success=False, message="Could not read file.", error="File read error.")
        
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
        value_counts = pd.Series(labels).value_counts().sort_index()
        counts = {}
        for k, v in value_counts.items():
            try:
                counts[int(k)] = int(v)
            except (ValueError, TypeError):
                continue
        
        return ClusteringResponse(success=True, message="Clustering successful.", preview=preview, cluster_counts=counts)
    except Exception as e:
        return ClusteringResponse(success=False, message=f"Error during clustering: {e}", error=str(e))

@app.post("/extract-well-locations")
async def extract_well_locations():
    """
    Inspect all uploaded files to find columns representing well name/uwi, latitude, and longitude.
    Extract and standardize these columns and save to a new CSV file with columns: uwi, latitude, longitude.
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
    
    # Find the first file with all required columns
    for filename in files:
        df = get_cached_file_data(filename)
        if df is None:
            continue
        
        lower_cols = [c.lower() for c in df.columns]
        
        # Prefer 'uwi' as well identifier
        for c in df.columns:
            if 'uwi' in c.lower():
                well_col = c
                break
        
        if not well_col:
            for c in df.columns:
                if any(p in c.lower() for p in well_patterns):
                    well_col = c
                    break
        
        for c in df.columns:
            if any(p in c.lower() for p in lat_patterns):
                lat_col = c
                break
        
        for c in df.columns:
            if any(p in c.lower() for p in long_patterns):
                long_col = c
                break
        
        if well_col and lat_col and long_col:
            found_file = filename
            break
    
    if not (found_file and well_col and lat_col and long_col):
        return {"success": False, "message": "No file with well, latitude, and longitude columns found."}
    
    # Extract data from the found file
    df = get_cached_file_data(found_file)
    if df is None:
        return {"success": False, "message": "Could not read the selected file."}
    
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

@app.get("/summarize-folder")
async def summarize_folder():
    summary = scan_folder_structure(UPLOAD_DIR)
    return {"files": summary}

@app.get("/scan-filesystem")
async def scan_filesystem_endpoint():
    """Scan the file system for folders and CSV files."""
    result = scan_file_system(".")
    return result

@app.post("/process-zip")
async def process_zip_endpoint(zip_filename: str):
    """Process a specific zip file that was already uploaded."""
    result = process_uploaded_zip(zip_filename)
    return result

@app.get("/list-extracted-files")
async def list_extracted_files():
    """List all files that were extracted from zip files."""
    extracted_files = []
    for root, dirs, files in os.walk(UPLOAD_DIR):
        for file in files:
            if file.lower().endswith('.csv'):
                file_path = os.path.join(root, file)
                rel_path = os.path.relpath(file_path, UPLOAD_DIR)
                extracted_files.append({
                    "filename": file,
                    "full_path": file_path,
                    "relative_path": rel_path,
                    "size": os.path.getsize(file_path) if os.path.exists(file_path) else 0
                })
    return {"extracted_files": extracted_files, "count": len(extracted_files)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=3002)
    
