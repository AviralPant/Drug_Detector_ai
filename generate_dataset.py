import os
import numpy as np
import pandas as pd
import cv2
from PIL import Image, ImageDraw, ImageFont
import random
import shutil

def create_directories():
    """Create necessary directories for the dataset."""
    dirs = [
        'datasets/eye_images/train/drug_user',
        'datasets/eye_images/train/non_user',
        'datasets/eye_images/validation/drug_user',
        'datasets/eye_images/validation/non_user',
        'datasets/eye_images/test/drug_user',
        'datasets/eye_images/test/non_user'
    ]
    
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)
    
    return dirs

def generate_eye_image(is_drug_user=True, image_size=(224, 224)):
    """
    Generate a synthetic eye image.
    
    Args:
        is_drug_user: Whether the eye shows signs of drug use
        image_size: Size of the image (width, height)
        
    Returns:
        A PIL Image object
    """
    # Create a new image with a light background
    image = Image.new('RGB', image_size, color=(240, 240, 240))
    draw = ImageDraw.Draw(image)
    
    # Draw the eye shape (ellipse)
    eye_width = image_size[0] - 40
    eye_height = image_size[1] - 60
    eye_x = 20
    eye_y = 30
    
    draw.ellipse([eye_x, eye_y, eye_x + eye_width, eye_y + eye_height], fill=(255, 255, 255), outline=(200, 200, 200))
    
    # Draw the iris
    iris_radius = min(eye_width, eye_height) // 3
    iris_x = image_size[0] // 2
    iris_y = image_size[1] // 2
    
    # Iris color varies
    iris_colors = [
        (101, 67, 33),    # Brown
        (28, 107, 160),   # Blue
        (46, 125, 50),    # Green
        (143, 57, 27),    # Dark brown
        (61, 90, 128)     # Gray-blue
    ]
    iris_color = random.choice(iris_colors)
    
    draw.ellipse([iris_x - iris_radius, iris_y - iris_radius, 
                 iris_x + iris_radius, iris_y + iris_radius], 
                 fill=iris_color, outline=(0, 0, 0))
    
    # Draw the pupil
    if is_drug_user:
        # Drug users often have dilated pupils
        pupil_radius = int(iris_radius * random.uniform(0.6, 0.9))
    else:
        # Normal pupil size
        pupil_radius = int(iris_radius * random.uniform(0.3, 0.5))
    
    draw.ellipse([iris_x - pupil_radius, iris_y - pupil_radius, 
                 iris_x + pupil_radius, iris_y + pupil_radius], 
                 fill=(0, 0, 0))
    
    # Draw blood vessels (redness) - more prominent in drug users
    if is_drug_user:
        num_vessels = random.randint(10, 30)
        vessel_color = (200, 50, 50)  # More red
    else:
        num_vessels = random.randint(0, 10)
        vessel_color = (180, 80, 80)  # Less red
    
    for _ in range(num_vessels):
        # Random start and end points for vessels
        start_x = random.randint(eye_x, eye_x + eye_width)
        start_y = random.randint(eye_y, eye_y + eye_height)
        
        # Vessels radiate from the iris
        angle = random.uniform(0, 2 * np.pi)
        length = random.randint(10, 40)
        end_x = start_x + int(length * np.cos(angle))
        end_y = start_y + int(length * np.sin(angle))
        
        # Draw the vessel as a line
        draw.line([start_x, start_y, end_x, end_y], fill=vessel_color, width=1)
    
    # Add some noise to make it look more realistic
    for _ in range(100):
        x = random.randint(0, image_size[0])
        y = random.randint(0, image_size[1])
        r = random.randint(0, 50)
        g = random.randint(0, 50)
        b = random.randint(0, 50)
        draw.point([x, y], fill=(r, g, b))
    
    return image

def generate_dataset(num_samples=1000, train_ratio=0.7, val_ratio=0.15):
    """
    Generate a dataset of eye images.
    
    Args:
        num_samples: Total number of samples to generate
        train_ratio: Ratio of samples for training
        val_ratio: Ratio of samples for validation
        
    Returns:
        None (saves images to disk)
    """
    dirs = create_directories()
    
    # Calculate number of samples for each set
    num_train = int(num_samples * train_ratio)
    num_val = int(num_samples * val_ratio)
    num_test = num_samples - num_train - num_val
    
    # Generate drug user and non-user images in equal proportions
    for i in range(num_samples // 2):
        # Drug user images
        if i < num_train // 2:
            save_dir = 'datasets/eye_images/train/drug_user'
        elif i < (num_train + num_val) // 2:
            save_dir = 'datasets/eye_images/validation/drug_user'
        else:
            save_dir = 'datasets/eye_images/test/drug_user'
        
        image = generate_eye_image(is_drug_user=True)
        image.save(f"{save_dir}/drug_user_{i}.jpg")
        
        # Non-user images
        if i < num_train // 2:
            save_dir = 'datasets/eye_images/train/non_user'
        elif i < (num_train + num_val) // 2:
            save_dir = 'datasets/eye_images/validation/non_user'
        else:
            save_dir = 'datasets/eye_images/test/non_user'
        
        image = generate_eye_image(is_drug_user=False)
        image.save(f"{save_dir}/non_user_{i}.jpg")
    
    print(f"Generated {num_samples} eye images:")
    print(f"- Training: {num_train} images")
    print(f"- Validation: {num_val} images")
    print(f"- Test: {num_test} images")

def create_drug_symptoms_dataset():
    """Create a CSV file with drug symptoms data."""
    # Define the data structure
    data = {
        'drug_type': [],
        'drug_name': [],
        'symptom_code': [],
        'symptom_name': [],
        'cf_value': []
    }
    
    # Drug types and names
    drug_types = {
        'P01': 'Cocaine',
        'P02': 'Marijuana',
        'P03': 'Ecstasy',
        'P04': 'Heroin',
        'P05': 'Methamphetamine',
        'P06': 'Hallucinogen',
        'P07': 'Amphetamine',
        'P08': 'Pethidine',
        'P09': 'Codeine',
        'P10': 'Morphine'
    }
    
    # Symptoms
    symptoms = {
        'G01': 'Out of breath',
        'G02': 'Anxious and restless',
        'G03': 'Nausea and vomiting',
        'G04': 'Diarrhea',
        'G05': 'Convulsions',
        'G06': 'Easy to get angry',
        'G07': 'Depression',
        'G08': 'Changes in sleep patterns',
        'G09': 'Sweating',
        'G10': 'Chills (Hot cold)',
        'G11': 'Shaking',
        'G12': 'Insomnia',
        'G13': 'Fast heart rate',
        'G14': 'Increased blood pressure',
        'G15': 'Difficult to focus',
        'G16': 'Difficult to rest',
        'G17': 'Weight loss',
        'G18': 'Dry mouth',
        'G19': 'Blurred vision',
        'G20': 'Changes in skin color',
        'G21': 'Constipation',
        'G22': 'Stomachache',
        'G23': 'Drowsiness',
        'G24': 'Itching',
        'G25': 'Difficulty urinating',
        'G26': 'Mood swings',
        'G27': 'Dizziness'
    }
    
    # Rule base and CF values (simplified)
    rule_base = {
        'P01': {'G01': 0.8, 'G02': 0.8, 'G03': 0.8, 'G04': 1.0, 'G05': 0.8},
        'P02': {'G06': 1.0, 'G02': 0.8, 'G07': 0.8, 'G08': 0.8, 'G09': 0.8, 'G10': 0.8},
        'P03': {'G05': 0.8, 'G11': 0.8, 'G12': 0.8, 'G13': 0.8, 'G14': 1.0},
        'P04': {'G15': 1.0, 'G02': 1.0, 'G07': 0.8, 'G16': 0.8},
        'P05': {'G11': 0.8, 'G01': 0.8, 'G16': 0.8, 'G17': 0.8, 'G18': 1.0},
        'P06': {'G09': 0.8, 'G11': 0.8, 'G18': 1.0, 'G19': 0.8, 'G10': 0.8, 'G14': 0.8},
        'P07': {'G18': 1.0, 'G03': 0.8, 'G04': 0.8, 'G05': 0.8, 'G01': 0.8, 'G20': 0.8},
        'P08': {'G07': 0.8, 'G13': 0.8, 'G05': 0.8, 'G03': 0.8},
        'P09': {'G27': 1.0, 'G03': 0.8, 'G18': 0.8, 'G21': 0.8, 'G22': 0.8},
        'P10': {'G23': 1.0, 'G24': 0.8, 'G09': 0.8, 'G25': 0.8, 'G26': 1.0}
    }
    
    # Populate the data
    for drug_code, drug_name in drug_types.items():
        for symptom_code, cf_value in rule_base[drug_code].items():
            data['drug_type'].append(drug_code)
            data['drug_name'].append(drug_name)
            data['symptom_code'].append(symptom_code)
            data['symptom_name'].append(symptoms[symptom_code])
            data['cf_value'].append(cf_value)
    
    # Create DataFrame and save to CSV
    df = pd.DataFrame(data)
    df.to_csv('datasets/drug_symptoms.csv', index=False)
    
    print("Drug symptoms dataset created and saved to 'datasets/drug_symptoms.csv'")
    print(f"Dataset shape: {df.shape}")
    print("Sample data:")
    print(df.head())

if __name__ == "__main__":
    # Create datasets directory if it doesn't exist
    os.makedirs('datasets', exist_ok=True)
    
    # Generate drug symptoms dataset
    create_drug_symptoms_dataset()
    
    # Generate eye images dataset
    generate_dataset(num_samples=1000)