from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import pandas as pd
import io
import os
from app.core.agent import AnalysisResult, PipelineOrchestrator

app = FastAPI(title="GenAI Data Quality Agent")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static directory
static_dir = os.path.join(os.path.dirname(__file__), "app/static")
if not os.path.exists(static_dir):
    os.makedirs(static_dir)
app.mount("/static", StaticFiles(directory=static_dir), name="static")

pipeline = PipelineOrchestrator()

@app.get("/")
def read_root():
    return FileResponse(os.path.join(static_dir, "index.html"))

@app.post("/analyze", response_model=AnalysisResult)
async def analyze_dataset(file: UploadFile = File(...)):
    if not file.filename.endswith(('.csv', '.json', '.xlsx')):
        raise HTTPException(status_code=400, detail="Invalid file format. Please upload CSV, JSON, or Excel.")
    
    contents = await file.read()
    
    try:
        if file.filename.endswith('.csv'):
            df = pd.read_csv(io.BytesIO(contents))
        elif file.filename.endswith('.json'):
            df = pd.read_json(io.BytesIO(contents))
        elif file.filename.endswith('.xlsx'):
            df = pd.read_excel(io.BytesIO(contents))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error parsing file: {str(e)}")
        
    result = pipeline.analyze(df)
    return result

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
