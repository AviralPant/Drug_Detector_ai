import os
import cv2
import numpy as np
import pandas as pd
import random
import shutil
import kagglehub
from PIL import Image
import base64
import json

class KaggleDatasetIntegration:
    def __init__(self):
        """Initialize the Kaggle dataset integration."""
        self.dataset_path = None
        self.drug_images_dir = 'datasets/drug_images'
        self.mapping_file = 'datasets/drug_mapping.json'
        self.mapping = {}
        self.load_mapping()
        
    def download_dataset(self):
        """Download the Kaggle dataset."""
        try:
            # Download latest version
            path = kagglehub.dataset_download("pkdarabi/the-drug-name-detection-dataset")
            print("Path to dataset files:", path)
            self.dataset_path = path
            
            # Create the drug images directory if it doesn't exist
            os.makedirs(self.drug_images_dir, exist_ok=True)
            
            # Copy the dataset to our project structure
            self._organize_dataset()
            
            return True
        except Exception as e:
            print(f"Error downloading dataset: {e}")
            return False
    
    def _organize_dataset(self):
        """Organize the dataset into our project structure."""
        if not self.dataset_path:
            print("Dataset path not set. Please download the dataset first.")
            return
        
        # The Kaggle dataset structure might be:
        # path/drug_name/image_files
        
        # We'll create a mapping of drug names to image paths
        drug_mapping = {}
        
        # Walk through the dataset directory
        for root, dirs, files in os.walk(self.dataset_path):
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    # Get the drug name from the directory name
                    drug_name = os.path.basename(root)
                    
                    # Create a directory for this drug in our project structure
                    drug_dir = os.path.join(self.drug_images_dir, drug_name)
                    os.makedirs(drug_dir, exist_ok=True)
                    
                    # Copy the image to our project structure
                    src_path = os.path.join(root, file)
                    dst_path = os.path.join(drug_dir, file)
                    shutil.copy2(src_path, dst_path)
                    
                    # Add to mapping
                    if drug_name not in drug_mapping:
                        drug_mapping[drug_name] = []
                    drug_mapping[drug_name].append(dst_path)
        
        # Save the mapping
        with open(self.mapping_file, 'w') as f:
            json.dump(drug_mapping, f)
        
        print(f"Dataset organized. Mapping saved to {self.mapping_file}")
        self.mapping = drug_mapping
    
    def load_mapping(self):
        """Load the drug image mapping."""
        if os.path.exists(self.mapping_file):
            with open(self.mapping_file, 'r') as f:
                self.mapping = json.load(f)
    
    def get_drug_image(self, drug_name):
        """Get a random image for a drug."""
        if drug_name in self.mapping and self.mapping[drug_name]:
            # Return a random image for this drug
            return random.choice(self.mapping[drug_name])
        return None
    
    def get_drug_image_base64(self, drug_name):
        """Get a drug image as base64 string."""
        image_path = self.get_drug_image(drug_name)
        if image_path:
            with open(image_path, "rb") as image_file:
                encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
                return encoded_string
        return None
    
    def get_available_drugs(self):
        """Get list of available drugs in the dataset."""
        return list(self.mapping.keys())
    
    def preprocess_image(self, image_path, target_size=(224, 224)):
        """Preprocess an image for model input."""
        # Load the image
        img = cv2.imread(image_path)
        if img is None:
            return None
        
        # Resize the image
        img = cv2.resize(img, target_size)
        
        # Convert to RGB (OpenCV uses BGR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Normalize pixel values
        img = img / 255.0
        
        return img
    
    def get_random_drug_image(self):
        """Get a random drug image from the dataset."""
        if not self.mapping:
            return None, None
        
        # Select a random drug
        drug_name = random.choice(list(self.mapping.keys()))
        
        # Get a random image for this drug
        image_path = random.choice(self.mapping[drug_name])
        
        return drug_name, image_path