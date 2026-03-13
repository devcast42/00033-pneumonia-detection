import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.dataset_loader import get_dataloaders
from models.definitions import SimpleCNN
from utils.trainer import train_model

def main():
    # Configuration
    DATA_DIR = os.path.join('dataset', 'chest_xray')
    MODEL_SAVE_PATH = os.path.join('models', 'cnn_model.pth')
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
    print("Initializing SimpleCNN...")
    model = SimpleCNN().to(device)
    
    # Loss and Optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
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
