from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import pandas as pd
import io
import os
from app.core.agent import (
    AnalysisResult, PipelineOrchestrator, FailureCategory, FailureInfo, 
    DecisionReasoning, LayerStatus
)
from app.core.layers.layer_11_logging import Layer11Logging

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

def create_api_failure(category: FailureCategory, message: str, layer: str = "API") -> AnalysisResult:
    """Create a standardized failure result for API-level errors."""
    user_actions = {
        FailureCategory.INPUT_MALFORMED: "Check file format and encoding. Re-export from source system.",
        FailureCategory.INPUT_MISSING: "Ensure file is uploaded and is CSV, JSON, or Excel format.",
        FailureCategory.SYSTEM_UNHANDLED_EXCEPTION: "Unexpected error. Contact support.",
    }
    
    logger = Layer11Logging()
    logger.log_event("API", "FAILURE", {"category": category.value, "message": message})
    
    return AnalysisResult(
        composite_score=0.0,
        overall_status="NO_ACTION",
        summary=f"Analysis failed: {message}",
        decision_reasoning=DecisionReasoning(
            decision_state="NO_ACTION",
            primary_reason=f"API failure: {category.value}",
            rule_triggers=[message],
            ai_inputs=["Analysis not started"],
            overridden_by_rules=False
        ),
        dimensions=[],
        anomalies=[],
        trace_id=logger.trace_id,
        confidence_band="LOW",
        confidence_score=0,
        layer_trace=[LayerStatus(layer=layer, status="FAIL", details=message)],
        logs=[str(log) for log in logger.get_logs()],
        responsible_party="DATA_ENGINEERING",
        next_steps=[user_actions.get(category, "Contact support."), "Fix issue", "Retry upload"],
        liability_summary="⛔ Analysis not started. No processing permitted.",
        failure_info=FailureInfo(
            category=category.value,
            message=message,
            layer=layer,
            recoverable=False,
            user_action_required=user_actions.get(category, "Contact support.")
        )
    )

@app.get("/")
def read_root():
    return FileResponse(os.path.join(static_dir, "index.html"))

@app.post("/analyze", response_model=AnalysisResult)
async def analyze_dataset(file: UploadFile = File(...), simulate_failure: bool = False):
    # Check file format
    if not file.filename.endswith(('.csv', '.json', '.xlsx')):
        return create_api_failure(
            FailureCategory.INPUT_MALFORMED,
            f"Invalid file format: {file.filename}. Please upload CSV, JSON, or Excel.",
            "File Upload"
        )
    
    contents = await file.read()
    
    # Parse file
    try:
        if file.filename.endswith('.csv'):
            df = pd.read_csv(io.BytesIO(contents))
        elif file.filename.endswith('.json'):
            df = pd.read_json(io.BytesIO(contents))
        elif file.filename.endswith('.xlsx'):
            df = pd.read_excel(io.BytesIO(contents))
    except Exception as e:
        return create_api_failure(
            FailureCategory.INPUT_MALFORMED,
            f"Error parsing file: {str(e)}",
            "File Parsing"
        )
    
    # Run pipeline with exception handling
    try:
        result = pipeline.analyze(df, simulate_ml_failure=simulate_failure)
        return result
    except Exception as e:
        return create_api_failure(
            FailureCategory.SYSTEM_UNHANDLED_EXCEPTION,
            f"Pipeline error: {str(e)}",
            "Pipeline"
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

