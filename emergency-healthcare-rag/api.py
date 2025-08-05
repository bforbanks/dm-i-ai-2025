import uvicorn
import argparse
import sys
import importlib
from fastapi import FastAPI
import datetime
import time
from utils import validate_prediction
from loguru import logger
from pydantic import BaseModel

HOST = "0.0.0.0"
PORT = 8000

# Import model-agnostic functions
from model import predict, set_active_model, get_active_model

class MedicalStatementRequestDto(BaseModel):
    statement: str

class MedicalStatementResponseDto(BaseModel):
    statement_is_true: int
    statement_topic: int

app = FastAPI()
start_time = time.time()

def warm_up_models():
    """Preload models and embeddings to avoid cold start delays"""
    model_name = get_active_model()
    logger.info(f"🔥 Warming up {model_name} model...")
    
    try:
        # Test prediction to warm up all components (includes both BM25 search and LLM)
        test_statement = "Euglycemic diabetic ketoacidosis is characterized by blood glucose less than 250 mg/dL with metabolic acidosis."
        logger.info(f"🧪 Running test prediction to warm up {model_name}...")
        
        start_warmup = time.time()
        truth, topic = predict(test_statement)
        warmup_time = time.time() - start_warmup
        
        logger.info(f"✅ Warm-up complete! Test prediction: truth={truth}, topic={topic}")
        logger.info(f"⏱️  Warm-up time: {warmup_time:.2f}s")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Warm-up failed: {e}")
        return False

# Don't warm up immediately - wait for model to be set via CLI or default

@app.on_event("startup")
async def startup_event():
    """API startup event"""
    logger.info("🚀 API server ready to receive requests!")

@app.get('/api')
def hello():
    # Get current dataset info
    try:
        current_model = get_active_model()
        if current_model == "match-and-choose-model-1":
            topic_model_module = importlib.import_module(f"{current_model}.topic_model")
            dataset = "condensed_topics" if topic_model_module.USE_CONDENSED_TOPICS else "topics"
        else:
            dataset = "unknown"
    except:
        dataset = "unknown"
    
    return {
        "service": "emergency-healthcare-rag",
        "model": get_active_model(),
        "dataset": dataset,
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

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Emergency Healthcare RAG API')
    parser.add_argument(
        '--model', 
        type=str, 
        default=None,
        help='Model to use (e.g., match-and-choose-model-1, separated-models-2, combined-model-2)'
    )
    parser.add_argument(
        '--threshold', 
        type=str, 
        default=None,
        help='Threshold to use (float or "NA", default: config default)'
    )
    parser.add_argument('--use-condensed-topics', action='store_true', default=True,
                       help='Use condensed_topics (default: True, set --no-use-condensed-topics for regular topics)')
    parser.add_argument('--no-use-condensed-topics', dest='use_condensed_topics', action='store_false',
                       help='Use regular topics instead of condensed_topics')
    parser.add_argument('--host', type=str, default=HOST, help=f'Host to bind to (default: {HOST})')
    parser.add_argument('--port', type=int, default=PORT, help=f'Port to bind to (default: {PORT})')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    
    # Set model if provided
    if args.model:
        set_active_model(args.model)
        logger.info(f"🎯 Active model set to: {args.model}")
    
    # Set threshold if provided
    if args.threshold:
        try:
            # Import threshold setting function from the active model's config
            current_model = get_active_model()
            config_module = importlib.import_module(f"{current_model}.config")
            
            if args.threshold.upper() == 'NA':
                threshold = 'NA'
            else:
                threshold = float(args.threshold)
            
            config_module.set_threshold(threshold)
            logger.info(f"🎯 Threshold set to: {threshold}")
        except Exception as e:
            logger.warning(f"⚠️  Failed to set threshold '{args.threshold}': {e}")
    
    # Set topic configuration if provided
    if hasattr(args, 'use_condensed_topics'):
        try:
            # Import threshold setting function from the active model's config
            current_model = get_active_model()
            
            # For match-and-choose-model-1, we need to set the topic_model configuration
            if current_model == "match-and-choose-model-1":
                topic_model_module = importlib.import_module(f"{current_model}.topic_model")
                topic_model_module.USE_CONDENSED_TOPICS = args.use_condensed_topics
                topic_type = "condensed_topics" if args.use_condensed_topics else "topics"
                logger.info(f"📚 Using {topic_type} for knowledge base")
            else:
                # For other models, we might need different handling
                logger.info(f"📚 Topic configuration not yet implemented for {current_model}")
                
        except Exception as e:
            logger.warning(f"⚠️  Failed to set topic configuration: {e}")
    
    # Log current model and dataset
    current_model = get_active_model()
    topic_type = "condensed_topics" if args.use_condensed_topics else "topics"
    logger.info(f"🎯 Starting API with model: {current_model}")
    logger.info(f"📚 Using dataset: {topic_type}")
    
    # Warm up models
    logger.info("🔥 Pre-warming models...")
    warm_up_success = warm_up_models()
    
    if not warm_up_success:
        logger.warning("⚠️  Model warm-up failed, but continuing anyway...")
    
    # Start server
    uvicorn.run(
        'api:app',
        host=args.host,
        port=args.port
    )
