import torch
import torch.nn as nn
import os 
from torchvision import transforms 
from PIL import Image
from utils.ModelUtilities import ModelUtilites
from env import *

class InferenceModel:
    def __init__(self) -> None:
        self.model: nn.Module | None = None 

        self.transform = transforms.Compose([
            transforms.Resize((28,28)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,),(0.3081,))
        ])

    def load_trained_model(self):
        
        if self.model is None:
            print("Model not loaded")
            return "fail"

        try:
            loaded_model, checkpoint = ModelUtilites.load_model()

            if loaded_model is None:
                print("Model hadn't trained please retrain")
                return "fail"
            
            if checkpoint is None:
                print("invalid checkpoint format")
                return "fail"

            if "epoch" in checkpoint and checkpoint["epoch"] < EPOCHS:
                print("Model still in the training progess")
                return "fail"

            self.model = loaded_model
            self.model.eval()
            print("Model loaded successfully")
            return "success"
        
        except Exception as e:
            print(f"Error: {e}")
            return "fail"

    def predict_single_image(self, image_tensor:torch.Tensor):

        if self.model is None:
            print("Model not loaded")
            return None, None, None

        if len(image_tensor.size()) == 3:
            image_tensor = image_tensor.unsqueeze_(0)
        elif len(image_tensor.size()) == 2:
            image_tensor = image_tensor.unsqueeze_(0).unsqueeze_(0)

        image_tensor = self.transform(image_tensor).to(DEVICE)

        with torch.no_grad():
            output = self.model(image_tensor)
            probabilities = torch.softmax(output, dim=1)
            prediction = output.argmax(dim=1).item() 
            confidence = probabilities.argmax(dim=1)[0].item()

        return CATEGORIES[prediction], confidence, probabilities

    def prediction_image_path(self, image_path: str):
    
        if not os.path.exists(image_path):
            print(f"image path: {image_path} not exist")
            return None, None, None 

        img = Image.open(image_path).convert("L")
        img = img.resize((28,28))

        image_tensor = torch.Tensor(self.transform(img))

        return self.predict_single_image(image_tensor)
