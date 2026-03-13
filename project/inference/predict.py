import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import sys
import os

# Add project root to sys.path if running as script
if __name__ == "__main__":
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.definitions import SimpleCNN, get_transfer_model
from utils.preprocessing import get_val_transforms

def load_model(model_path, model_type='cnn', device='cpu'):
    """
    Load a trained model.
    model_type: 'cnn' or 'transfer'
    """
    if model_type == 'cnn':
        model = SimpleCNN()
    elif model_type == 'transfer':
        model = get_transfer_model(model_name='resnet18', pretrained=False) # Pretrained=False because we load weights
    else:
        raise ValueError("Invalid model_type. Use 'cnn' or 'transfer'.")
    
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
    except FileNotFoundError:
        print(f"Error: Model file not found at {model_path}")
        return None
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

    model.to(device)
    model.eval()
    return model

def predict_image(model, image_file, device='cpu'):
    """
    Predict class for an image.
    image_file: file path or file-like object (e.g. from API upload)
    """
    transform = get_val_transforms()
    
    try:
        image = Image.open(image_file).convert('RGB')
        image = transform(image).unsqueeze(0).to(device)
    except Exception as e:
        return {"error": f"Failed to process image: {str(e)}"}
    
    with torch.no_grad():
        output = model(image)
        probability = torch.sigmoid(output).item()
        
        # Mapping: 0 -> NORMAL, 1 -> PNEUMONIA (assuming alphabetical order in ImageFolder)
        # Verify class_to_idx during training if possible, but standard is alphabetical.
        # NORMAL comes before PNEUMONIA.
        
        if probability > 0.5:
            prediction = "PNEUMONIA"
            confidence = probability
        else:
            prediction = "NORMAL"
            confidence = 1 - probability
            
    return {
        "prediction": prediction,
        "confidence": round(confidence, 4)
    }

if __name__ == "__main__":
    # Example usage
    if len(sys.argv) < 3:
        print("Usage: python predict.py <image_path> <model_path> [model_type]")
        sys.exit(1)
        
    img_path = sys.argv[1]
    mod_path = sys.argv[2]
    mod_type = sys.argv[3] if len(sys.argv) > 3 else 'cnn'
    
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loaded_model = load_model(mod_path, mod_type, dev)
    
    if loaded_model:
        result = predict_image(loaded_model, img_path, dev)
        print(result)
