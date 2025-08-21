import torch 
import torch.nn as nn 
import torch.optim as optim
from env import *
from utils.ModelUtilities import ModelUtilites
from tqdm import tqdm
from model.cnn import Model
from data.DataHandler import get_dataset

class TrainingProgress:
    
    def __init__(self) -> None:
        self.train_loader, self.validate_loader, self.test_loader = get_dataset()
        self.model = Model().to(DEVICE) 
        self.optimizer = optim.AdamW(self.model.parameters(),lr=0.001, weight_decay=WEIGHT_DECAY) 
        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1) 
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=self.optimizer,
            mode="max",
            factor=0.5,
            patience=5,
        )

        self.history = {
            "train_losses": [],
            "train_accuracies": [],
            "validate_losses": [],
            "validate_accuracies": [],
        }    

    def train_epoch(self):
        self.model.train()
        running_loss = 0 
        correct = 0 
        total = 0

        progress_bar = tqdm(self.train_loader, desc="Training")

        for images, labels in progress_bar:
            images, labels = images.to(DEVICE, non_blocking=True), labels.to(DEVICE, non_blocking=True)
            
            self.optimizer.zero_grad()
            output = self.model(images)
            loss = self.criterion(output, labels)
            loss.backward()

            self.optimizer.step()

            running_loss += loss.item()
            _,prediction = torch.max(output,1)
            correct += prediction.eq(labels.view_as(prediction)).sum().item()
            total += labels.size(0)

            current_accuracy = 100 * correct / total 

            progress_bar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "accuracy": f"{current_accuracy:.2f}",
                "LR": f"{self.scheduler.get_last_lr()[0]:.6f}"
            })

        return running_loss / len(self.train_loader), 100 * correct / total

    def validate(self):
        self.model.eval()
        test_loss = 0 
        correct = 0 
        total = 0 

        with torch.no_grad():

            progress_bar = tqdm(self.test_loader, desc="Testing")

            for images, labels in progress_bar:
                images, labels = images.to(DEVICE, non_blocking=True), labels.to(DEVICE, non_blocking=True)
            
                output = self.model(images)
                loss = self.criterion(output, labels)

                test_loss += loss.item()
                _,prediction = torch.max(output,1)
                correct += prediction.eq(labels.view_as(prediction)).sum().item()
                total += labels.size(0)

                current_accuracy = 100 * correct / total 

                progress_bar.set_postfix({
                    "loss": f"{loss.item():.4f}",
                    "accuracy": f"{current_accuracy:.2f}",
                })

        return test_loss / len(self.test_loader), 100 * correct / total


    def evaluate(self, model_state):

        if model_state is not None:
            self.model.load_state_dict(model_state)
        
        self.model.eval()
        test_loss = 0 
        correct = 0 
        total = 0 

        with torch.no_grad():

            progress_bar = tqdm(self.test_loader, desc="Testing")

            for images, labels in progress_bar:
                images, labels = images.to(DEVICE, non_blocking=True), labels.to(DEVICE, non_blocking=True)
            
                output = self.model(images)
                loss = self.criterion(output, labels)

                test_loss += loss.item()
                _,prediction = torch.max(output,1)
                correct += prediction.eq(labels.view_as(prediction)).sum().item()
                total += labels.size(0)

                current_accuracy = 100 * correct / total 

                progress_bar.set_postfix({
                    "loss": f"{loss.item():.4f}",
                    "accuracy": f"{current_accuracy:.2f}",
                })

        return test_loss / len(self.test_loader), 100 * correct / total
    
    def train(self):
        best_validate_accuracy = 0
        best_model_state = None  
        count = 1
        try:
            for epoch in range(EPOCHS):
                print(f"Epoch: {epoch + 1} / {EPOCHS}")

                train_loss, train_accuracy = self.train_epoch()
                self.history["train_losses"].append(train_loss)
                self.history["train_accuracies"].append(train_accuracy)

                validate_loss, validate_accuracy = self.validate()
                self.history["validate_losses"].append(validate_loss)
                self.history["validate_accuracies"].append(validate_accuracy)

                self.scheduler.step(validate_accuracy)

                count += 1

                if validate_accuracy > best_validate_accuracy:
                    best_validate_accuracy = validate_accuracy
                    best_model_state = self.model.state_dict().copy()
                    print(f"Best validate accuracy: {best_validate_accuracy:.2f}")

            _,test_accuracy = self.evaluate(best_model_state)

            print(f"Test Accuracy: {test_accuracy:.2f}")

            ModelUtilites.save_model(model=self.model, optimizer=self.optimizer, epoch=count, test_accuracy=test_accuracy)

            print("Model saved successfully")

        except Exception as e:
            print(f"Error: {e}")
