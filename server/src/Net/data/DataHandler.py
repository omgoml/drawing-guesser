from typing import Literal
import os
import numpy as np
import urllib.request
import json
import random
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from env import *
from PIL import Image, ImageDraw

class DatasetHandler(Dataset):
    def __init__(self,
        root_dir: str = ROOT_PATH,
        categories: list[str] | None = None,
        split: Literal["train", "validate", "test"] = "train",
        sample_ratio: float = 0.1,
        download: bool = True,
        format_type: Literal["numpy", "raw"] = "numpy",
        transform: transforms.Compose | None = None,
        cache_processed: bool = True,
    ) -> None:
        super().__init__()
        
        self.data_dir: str = root_dir
        self.split = split
        self.sample_ratio = sample_ratio
        self.transform = transform 
        self.cache_processed = cache_processed
       
        if categories is None: 
            self.categories = CATEGORIES.copy()
        else:
            self.categories = categories

        os.makedirs(self.data_dir, exist_ok=True)
        
        np.random.seed(42)
        torch.manual_seed(42)
        random.seed(42)
        
        if torch.cuda.is_available():
            torch.cuda.manual_seed(42)
            torch.cuda.manual_seed_all(42)

        if download:
            self._download_data(format_type)

        self.class_to_index = {label: index for index, label in enumerate(self.categories)}
    
    def _download_data(self, format_type: Literal["numpy", "raw"] = "numpy"):
        
        base_url: str = "https://storage.googleapis.com/quickdraw_dataset/full/"

        if format_type == "numpy":
            base_url += "numpy_bitmap/"
            file_ext = ".npy"
        else:
            base_url += "raw/"
            file_ext = ".ndjson"

        print(f"Starting download of {len(self.categories)} categories...")
        
        for i, category in enumerate(self.categories):
            file_name = f"{category.replace(" ", "%20")}{file_ext}"
            url = base_url + file_name  
            file_path = os.path.join(self.data_dir, f"{category}{file_ext}")

            if os.path.exists(file_path):
                print(f"Already exists {i+1}/{len(self.categories)}: {category}")

            try:
                print(f"Downloading {i+1}/{len(self.categories)}: {category}...")
                
                urllib.request.urlretrieve(url, file_path)

            except Exception as e:
                print(f"Error downloading {category}: {e}")
                
                #removing the falied category 
                self.categories.remove(category)
        
        print(f"Download complete! Successfully downloaded {len(self.categories)} categories")

    def _load_data(self, format_type: Literal["numpy", "raw"] = "numpy"):
        #storing data for training
        all_data = [] 
        all_labels = [] 
        for idx, category in enumerate(self.categories):
            #checking the numpy format is available
            file_path = os.path.join(self.data_dir, f"{category}.npy")
            if os.path.exists(file_path):
                data = np.load(file_path)

                total_sample = len(data)
                sample_size = int(total_sample * self.sample_ratio)

                indices = np.random.choice(total_sample, sample_size, replace=False)
                data = data[indices]

                train_end = int(0.8 * total_sample)
                validate_end = int(0.9 * total_sample)

                if self.split == "train":
                    data = data[:train_end]
                elif self.split == "validate":
                    data = data[train_end: validate_end]
                else:
                    data = data[validate_end:]

                all_data.append(data)
                all_labels.extend([idx] * len(data))

            else:
                file_path = os.path.join(self.data_dir, f"{category}.ndjson")
                if os.path.exists(file_path):
                    drawings = self._load_ndjson_data(file_path)
                    images = [self._stroke_to_image(d) for d in drawings]
                    all_data.append(images)
                    all_labels.extend([idx] * len(images))

        if format_type == "numpy" and all_data:
            return np.concatenate(all_data, axis=0), np.array(all_labels)
        else:
            return np.array(all_data), np.array(all_labels)

    def _load_ndjson_data(self, file_path:str):
        #storing all the loaded drawing from the dataset
        drawings = []
        
        #checking if the images and labels is recognized by Google prevent the unrelevant data
        total_recoginized = 0 
        with open(file_path, "r") as file:
            for line in file:
                drawing = json.loads(line)

                if drawing["recoginized"]:
                    total_recoginized += 1 

        sample_size = int(total_recoginized * self.sample_ratio)

        selected_indices = set(random.sample(range(total_recoginized),sample_size))

        recognized_counter = 0 

        with open(file_path, "r") as file:
            for line in file:
                drawing = json.loads(line)

                if drawing["recognized"]:
                    if recognized_counter in selected_indices:
                        drawings.append(drawings)

                    recognized_counter += 1 
        
        total_sample = len(drawings)
        train_end = int(0.8 * total_sample)
        validate_end = int(0.9 * total_sample)

        if self.split == "train":
            drawings = drawings[:train_end]
        elif self.split == "validate":
            drawings = drawings[train_end: validate_end]
        else:
            drawings = drawings[validate_end:]

        return drawings

    def _stroke_to_image(self, drawing):

        img = Image.new(mode="L", size=(256,256), color=255)
        draw = ImageDraw.Draw(img)

        for stroke in drawing["drawing"]:
            if len(stroke[0]) > 1:
                point = list(zip(stroke[0], stroke[1]))
                draw.line(xy=point, fill=0, width=2)

        img = img.resize((28,28))

        return np.array(img)

def get_dataset(
    root_dir: str = ROOT_PATH,
    sample_ratio: float = 0.1,
    batch_size:int = BATCH_SIZE,
    num_worker:int = NUM_WORKER,
):

    train_transform = transforms.Compose([
        transforms.RandomRotation(10),
        transforms.RandomAffine(0, translate=(0.1,0.1)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,),(0.3081,)),
        transforms.Resize((28,28))
    ])

    evaluate_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,),(0.3081,)),
        transforms.Resize((28,28))
    ])

    train_data = DatasetHandler(
            root_dir=root_dir,
            categories=CATEGORIES,
            split="train",
            sample_ratio=sample_ratio,
            transform=train_transform,
            format_type="raw",
    )

    validate_data = DatasetHandler(
        root_dir=root_dir,
        categories=CATEGORIES,
        split="validate",
        sample_ratio=sample_ratio,
        transform=evaluate_transform,
        format_type="raw",
    )

    test_data = DatasetHandler(
        root_dir=root_dir,
        categories=CATEGORIES,
        split="test",
        sample_ratio=sample_ratio,
        transform=evaluate_transform,
        format_type="raw",
    )

    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_worker,
        pin_memory=torch.cuda.is_available(),
        drop_last=True
    )

    validate_loader = DataLoader(
        validate_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_worker,
        pin_memory=torch.cuda.is_available()
    ) 

    test_loader = DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_worker,
        pin_memory=torch.cuda.is_available(),
    )

    return train_loader, validate_loader, test_loader
