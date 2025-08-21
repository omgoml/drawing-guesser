import torch 
import torch.nn as nn 
import torch.optim as optim
import os 
from env import *
from Net.model.cnn import Model

class ModelUtilites:
    @staticmethod
    def save_model(model: nn.Module, optimizer: optim.Optimizer, epoch:int, test_accuracy):
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "test_accuracy": test_accuracy,
        },MODEL_PATH)
        
    @staticmethod
    def load_model():
        if not os.path.exists(MODEL_PATH):
            print(f"File path: {MODEL_PATH} not exist")
            return None, None 

        model = Model().to(DEVICE)
        checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)

        model.load_state_dict(checkpoint["model_state_dict"])
        
        print("Model loaded successfully") 
        print(f"Epoch: {checkpoint["epoch"]}")
        print(f"Best accuracy: {checkpoint["best_accuracy"]:.2f}")
        
        return model, checkpoint
