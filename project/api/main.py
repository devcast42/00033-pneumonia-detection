from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import io
import torch
import sys
import os

# Add project root to sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(PROJECT_ROOT)

from inference.predict import load_model, predict_image

default_origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]
cors_origins_env = os.getenv("CORS_ORIGINS", "")
env_origins = [origin.strip() for origin in cors_origins_env.split(",") if origin.strip()]
allowed_origins = env_origins if env_origins else default_origins

app = FastAPI(
    title="Pneumonia Detection API",
    description="API to detect pneumonia from chest X-ray images using Deep Learning models.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables to hold models
models = {}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@app.on_event("startup")
async def startup_event():
    """
    Load models on startup.
    """
    # Paths to models
    cnn_path = os.path.join(PROJECT_ROOT, "models", "cnn_model.pth")
    transfer_path = os.path.join(PROJECT_ROOT, "models", "transfer_model.pth")
    
    # Load CNN Model
    if os.path.exists(cnn_path):
        print(f"Loading CNN model from {cnn_path}...")
        models['cnn'] = load_model(cnn_path, model_type='cnn', device=device)
    else:
        print(f"Warning: CNN model not found at {cnn_path}")

    # Load Transfer Model
    if os.path.exists(transfer_path):
        print(f"Loading Transfer Learning model from {transfer_path}...")
        models['transfer'] = load_model(transfer_path, model_type='transfer', device=device)
    else:
        print(f"Warning: Transfer model not found at {transfer_path}")

@app.get("/")
async def root():
    return {"message": "Welcome to Pneumonia Detection API. Use POST /predict to analyze images."}

@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    model_type: str = "transfer"
):
    """
    Predict pneumonia from an uploaded image file.
    model_type: 'cnn' or 'transfer' (default: 'transfer')
    """
    if model_type not in ['cnn', 'transfer']:
        raise HTTPException(status_code=400, detail="Invalid model_type. Use 'cnn' or 'transfer'.")
    
    if model_type not in models or models[model_type] is None:
        raise HTTPException(status_code=503, detail=f"Model '{model_type}' is not loaded or available.")
    
    # Read image content
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image.")
    
    try:
        contents = await file.read()
        image_stream = io.BytesIO(contents)
        
        # Predict
        result = predict_image(models[model_type], image_stream, device=device)
        
        if "error" in result:
            raise HTTPException(status_code=500, detail=result["error"])
            
        return JSONResponse(content=result)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")

if __name__ == "__main__":
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
