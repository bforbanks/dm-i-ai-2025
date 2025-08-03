import uvicorn
from fastapi import FastAPI
import datetime
import time
from utils import validate_prediction
from model import predict
from loguru import logger
from pydantic import BaseModel

HOST = "0.0.0.0"
PORT = 8000

class MedicalStatementRequestDto(BaseModel):
    statement: str

class MedicalStatementResponseDto(BaseModel):
    statement_is_true: int
    statement_topic: int

app = FastAPI()
start_time = time.time()

def warm_up_models():
    """Preload models and embeddings to avoid cold start delays"""
    logger.info("🔥 Warming up models...")
    
    try:
        # Import the active model to trigger loading
        import importlib
        from model import ACTIVE_MODEL
        model_module = importlib.import_module(f"{ACTIVE_MODEL}.model")
        
        # Test prediction to warm up all components
        test_statement = "Euglycemic diabetic ketoacidosis is characterized by blood glucose less than 250 mg/dL with metabolic acidosis."
        logger.info("🧪 Running test prediction to warm up models...")
        
        start_warmup = time.time()
        truth, topic = model_module.predict(test_statement)
        warmup_time = time.time() - start_warmup
        
        logger.info(f"✅ Warm-up complete! Test prediction: truth={truth}, topic={topic}")
        logger.info(f"⏱️  Warm-up time: {warmup_time:.2f}s")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Warm-up failed: {e}")
        return False

# Warm up models immediately when this module is loaded
logger.info("🔥 Pre-warming models...")
warm_up_models()

@app.on_event("startup")
async def startup_event():
    """API startup event"""
    logger.info("🚀 API server ready to receive requests!")

@app.get('/api')
def hello():
    return {
        "service": "emergency-healthcare-rag",
        "uptime": '{}'.format(datetime.timedelta(seconds=time.time() - start_time))
    }

@app.get('/')
def index():
    return "Your endpoint is running!"

@app.post('/predict', response_model=MedicalStatementResponseDto)
def predict_endpoint(request: MedicalStatementRequestDto):

    logger.info(f'Received statement: {request.statement[:100]}...')

    # Get prediction from model
    statement_is_true, statement_topic = predict(request.statement)

    # Validate prediction format
    validate_prediction(statement_is_true, statement_topic)

    # Return the prediction
    response = MedicalStatementResponseDto(
        statement_is_true=statement_is_true,
        statement_topic=statement_topic
    )
    logger.info(f'Returning prediction: true={statement_is_true}, topic={statement_topic}')
    return response

if __name__ == '__main__':

    uvicorn.run(
        'api:app',
        host=HOST,
        port=PORT
    )
