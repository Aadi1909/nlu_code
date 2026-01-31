from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, Dict, List
import logging
import sys
from pathlib import Path
import uvicorn

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from nlu.nlu_pipeline import NLUPipeline, DialogueManager

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI
app = FastAPI(
    title="Battery Smart NLU API",
    description="NLU Pipeline for Voice Assistant",
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

# Global NLU pipeline and dialogue manager
nlu_pipeline = None
dialogue_manager = None


# Request/Response Models
class NLURequest(BaseModel):
    text: str = Field(..., description="Input text from user")
    session_id: Optional[str] = Field(None, description="Session ID for multi-turn")
    context: Optional[Dict] = Field(None, description="Context (driver_id, vehicle_id, etc.)")
    
    class Config:
        schema_extra = {
            "example": {
                "text": "where is my battery swap station",
                "session_id": "session_123",
                "context": {"driver_id": "DL123456"}
            }
        }


class NLUResponse(BaseModel):
    text: str
    intent: str
    confidence: float
    entities: List[Dict]
    slots: Dict
    is_complete: bool
    missing_slots: List[str]
    next_action: str
    backend_service: Optional[str]
    requires_agent: bool
    response_text: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    models_loaded: bool
    version: str


# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize models on startup"""
    global nlu_pipeline, dialogue_manager
    
    logger.info("Starting NLU API...")
    
    try:
        # Initialize NLU pipeline
        logger.info("Loading NLU models...")
        nlu_pipeline = NLUPipeline(
            intent_model_path='models/intent_classifier',
            entity_model_path='models/entity_extractor',
            config_path='config/intents.yaml'
        )
        
        # Initialize dialogue manager
        dialogue_manager = DialogueManager(nlu_pipeline)
        
        logger.info("NLU API started successfully")
        
    except Exception as e:
        logger.error(f"Failed to start NLU API: {str(e)}")
        raise


# Health check endpoint
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Check API health"""
    return {
        "status": "healthy" if nlu_pipeline is not None else "unhealthy",
        "models_loaded": nlu_pipeline is not None,
        "version": "1.0.0"
    }


# Single-turn NLU processing
@app.post("/nlu/process", response_model=NLUResponse)
async def process_nlu(request: NLURequest):
    """
    Process text through NLU pipeline (single-turn)
    """
    if nlu_pipeline is None:
        raise HTTPException(status_code=503, detail="NLU models not loaded")
    
    try:
        logger.info(f"Processing: '{request.text}'")
        
        result = nlu_pipeline.process(
            text=request.text,
            context=request.context
        )
        
        return NLUResponse(**result)
    
    except Exception as e:
        logger.error(f"Error processing NLU request: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# Multi-turn conversation
@app.post("/dialogue/turn")
async def dialogue_turn(request: NLURequest):
    """
    Process conversation turn (multi-turn with state management)
    """
    if dialogue_manager is None:
        raise HTTPException(status_code=503, detail="Dialogue manager not loaded")
    
    if not request.session_id:
        raise HTTPException(status_code=400, detail="session_id is required for dialogue")
    
    try:
        logger.info(f"Processing turn for session {request.session_id}: '{request.text}'")
        
        response = dialogue_manager.process_turn(
            text=request.text,
            session_id=request.session_id,
            context=request.context
        )
        
        return response
    
    except Exception as e:
        logger.error(f"Error processing dialogue turn: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# Validate slot endpoint
@app.post("/nlu/validate_slot")
async def validate_slot(slot_name: str, value: str):
    """Validate individual slot value"""
    try:
        is_valid = nlu_pipeline.slot_filler._validate_slot_value(slot_name, value)
        return {"valid": is_valid, "slot_name": slot_name, "value": value}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Get intent info
@app.get("/nlu/intents")
async def get_intents():
    """Get list of supported intents"""
    return {
        "intents": list(nlu_pipeline.config.get('tier_1_intents', {}).keys()),
        "tier_2": list(nlu_pipeline.config.get('valid_intents', []))
    }


# Batch processing
@app.post("/nlu/batch")
async def batch_process(texts: List[str], context: Optional[Dict] = None):
    """Process multiple texts in batch"""
    if nlu_pipeline is None:
        raise HTTPException(status_code=503, detail="NLU models not loaded")
    
    results = []
    for text in texts:
        try:
            result = nlu_pipeline.process(text, context)
            results.append(result)
        except Exception as e:
            results.append({"error": str(e), "text": text})
    
    return {"results": results, "total": len(texts)}


# Run server
if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )