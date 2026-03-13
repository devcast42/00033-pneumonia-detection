import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.dataset_loader import get_dataloaders
from models.definitions import get_transfer_model
from utils.trainer import train_model

def main():
    # Configuration
    DATA_DIR = os.path.join('dataset', 'chest_xray')
    MODEL_SAVE_PATH = os.path.join('models', 'transfer_model.pth')
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 10
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Data Loaders
    print("Loading datasets...")
    train_loader, val_loader, test_loader, class_to_idx = get_dataloaders(DATA_DIR, BATCH_SIZE)
    print(f"Classes: {class_to_idx}")
    
    # Model
    print("Initializing ResNet18 (Transfer Learning)...")
    model = get_transfer_model(model_name='resnet18', pretrained=True).to(device)
    
    # Loss and Optimizer
    criterion = nn.BCEWithLogitsLoss()
    # Only optimize parameters that require gradients (fc layer)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE)
    
    # Train
    print("Starting training...")
    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        num_epochs=NUM_EPOCHS,
        save_path=MODEL_SAVE_PATH
    )
    
    print("Training complete.")

if __name__ == '__main__':
    main()
