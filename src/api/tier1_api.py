#!/usr/bin/env python3
"""
Battery Smart NLU API - Tier 1 Intent Classification

Handles:
- 7 Tier 1 intents (bot resolves directly)
- 3 Agent handoff intents (transfer to human)
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, Dict, List
import logging
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from tier1_classifier import Tier1Classifier, IntentCategory, TIER1_RESPONSES, AGENT_HANDOFF_MESSAGES

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI
app = FastAPI(
    title="Battery Smart NLU API",
    description="Tier 1 Intent Classifier for Battery Smart Voice Assistant",
    version="2.0.0"
)

# Add CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global classifier
classifier: Tier1Classifier = None


# ==================== MODELS ====================

class NLURequest(BaseModel):
    text: str = Field(..., description="Input text from user")
    session_id: Optional[str] = None
    driver_id: Optional[str] = None
    language: str = Field(default="hinglish", description="Response language")
    
    class Config:
        schema_extra = {
            "example": {
                "text": "mera wallet balance batao",
                "driver_id": "AB123456",
                "language": "hinglish"
            }
        }


class NLUResponse(BaseModel):
    intent: str
    confidence: float
    category: str  # 'tier1' or 'agent'
    method: str  # 'rule', 'model', or 'fallback'
    action: str
    entities: Dict
    should_handoff: bool
    response: Optional[str] = None
    top_predictions: List[Dict] = []


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    version: str


# ==================== STARTUP ====================

@app.on_event("startup")
async def startup():
    """Initialize classifier on startup"""
    global classifier
    
    logger.info("Starting Tier 1 NLU API...")
    
    try:
        # Try to load ML model if available
        model_paths = [
            "/mnt/sagemaker-nvme/intent_models_full",
            "/mnt/sagemaker-nvme/intent_models",
            "models/intent_classifier"
        ]
        
        model_path = None
        for path in model_paths:
            if Path(path).exists():
                model_path = path
                break
        
        classifier = Tier1Classifier(
            model_path=model_path,
            confidence_threshold=0.5
        )
        
        logger.info("✅ Tier 1 NLU API started successfully")
        
    except Exception as e:
        logger.error(f"Failed to start: {e}")
        # Still start with rule-based only
        classifier = Tier1Classifier()
        logger.info("⚠️ Started with rule-based classifier only")


# ==================== ENDPOINTS ====================

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Check API health"""
    return {
        "status": "healthy" if classifier else "unhealthy",
        "model_loaded": classifier.model is not None if classifier else False,
        "version": "2.0.0"
    }


@app.post("/nlu/classify", response_model=NLUResponse)
async def classify_intent(request: NLURequest):
    """
    Classify user intent
    
    Returns:
    - intent: Detected intent
    - category: 'tier1' (bot handles) or 'agent' (handoff)
    - action: What should happen next
    - entities: Extracted entities (driver_id, booking_id, etc.)
    """
    if not classifier:
        raise HTTPException(status_code=503, detail="Classifier not loaded")
    
    try:
        result = classifier.classify(request.text)
        response_dict = result.to_dict()
        
        # Add response message based on category
        if result.category == IntentCategory.TIER_1:
            # For Tier 1, provide a template response
            if result.intent in TIER1_RESPONSES:
                templates = TIER1_RESPONSES[result.intent]
                lang = request.language if request.language in templates else 'hinglish'
                response_dict['response'] = templates[lang]
        else:
            # For agent handoff, provide handoff message
            response_dict['response'] = AGENT_HANDOFF_MESSAGES.get(
                result.intent,
                AGENT_HANDOFF_MESSAGES['unknown']
            )
        
        logger.info(f"Query: '{request.text}' → {result.intent} ({result.category.value})")
        
        return NLUResponse(**response_dict)
    
    except Exception as e:
        logger.error(f"Error classifying: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/nlu/batch")
async def batch_classify(requests: List[NLURequest]):
    """Classify multiple queries at once"""
    if not classifier:
        raise HTTPException(status_code=503, detail="Classifier not loaded")
    
    results = []
    for req in requests:
        result = classifier.classify(req.text)
        results.append(result.to_dict())
    
    return {"results": results}


@app.get("/intents")
async def list_intents():
    """List all supported intents"""
    return {
        "tier1_intents": list(Tier1Classifier.INTENT_CONFIG.keys()),
        "tier1_count": len([i for i, c in Tier1Classifier.INTENT_CONFIG.items() 
                          if c['category'] == IntentCategory.TIER_1]),
        "agent_count": len([i for i, c in Tier1Classifier.INTENT_CONFIG.items() 
                          if c['category'] == IntentCategory.AGENT]),
        "descriptions": {
            intent: config['description']
            for intent, config in Tier1Classifier.INTENT_CONFIG.items()
        }
    }


# ==================== RUN ====================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
