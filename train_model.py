import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import cv2
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

class EyeAnalysisModel:
    def __init__(self, input_shape=(224, 224, 3)):
        """
        Initialize the eye analysis model.
        
        Args:
            input_shape: Shape of input images
        """
        self.input_shape = input_shape
        self.model = self._build_model()
        self.history = None
    
    def _build_model(self):
        """Build the CNN model for eye analysis."""
        # Load pre-trained MobileNetV2 model
        base_model = MobileNetV2(
            weights='imagenet', 
            include_top=False, 
            input_shape=self.input_shape
        )
        
        # Freeze the base model
        base_model.trainable = False
        
        # Add custom layers
        inputs = Input(shape=self.input_shape)
        x = base_model(inputs, training=False)
        x = GlobalAveragePooling2D()(x)
        x = Dense(128, activation='relu')(x)
        x = Dropout(0.5)(x)
        outputs = Dense(1, activation='sigmoid')(x)
        
        model = Model(inputs, outputs)
        
        # Compile the model
        model.compile(
            optimizer=Adam(learning_rate=0.0001),
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.AUC()]
        )
        
        return model
    
    def train(self, train_dir, validation_dir, epochs=20, batch_size=32):
        """
        Train the model.
        
        Args:
            train_dir: Directory containing training images
            validation_dir: Directory containing validation images
            epochs: Number of training epochs
            batch_size: Batch size for training
            
        Returns:
            Training history
        """
        # Data augmentation for training
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest'
        )
        
        # Only rescaling for validation
        validation_datagen = ImageDataGenerator(rescale=1./255)
        
        # Flow training images in batches
        train_generator = train_datagen.flow_from_directory(
            train_dir,
            target_size=(self.input_shape[0], self.input_shape[1]),
            batch_size=batch_size,
            class_mode='binary'
        )
        
        # Flow validation images in batches
        validation_generator = validation_datagen.flow_from_directory(
            validation_dir,
            target_size=(self.input_shape[0], self.input_shape[1]),
            batch_size=batch_size,
            class_mode='binary'
        )
        
        # Define callbacks
        checkpoint = ModelCheckpoint(
            'models/eye_model.h5',
            monitor='val_accuracy',
            verbose=1,
            save_best_only=True,
            mode='max'
        )
        
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=5,
            verbose=1,
            restore_best_weights=True
        )
        
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=3,
            min_lr=0.00001,
            verbose=1
        )
        
        # Train the model
        self.history = self.model.fit(
            train_generator,
            steps_per_epoch=train_generator.samples // batch_size,
            epochs=epochs,
            validation_data=validation_generator,
            validation_steps=validation_generator.samples // batch_size,
            callbacks=[checkpoint, early_stopping, reduce_lr]
        )
        
        return self.history
    
    def evaluate(self, test_dir):
        """
        Evaluate the model on test data.
        
        Args:
            test_dir: Directory containing test images
            
        Returns:
            Evaluation metrics
        """
        # Only rescaling for test
        test_datagen = ImageDataGenerator(rescale=1./255)
        
        # Flow test images in batches
        test_generator = test_datagen.flow_from_directory(
            test_dir,
            target_size=(self.input_shape[0], self.input_shape[1]),
            batch_size=32,
            class_mode='binary',
            shuffle=False
        )
        
        # Evaluate the model
        loss, accuracy, auc = self.model.evaluate(test_generator)
        
        print(f"Test Loss: {loss:.4f}")
        print(f"Test Accuracy: {accuracy:.4f}")
        print(f"Test AUC: {auc:.4f}")
        
        # Get predictions
        predictions = self.model.predict(test_generator)
        predicted_classes = (predictions > 0.5).astype(int).flatten()
        true_classes = test_generator.classes
        
        # Classification report
        class_labels = list(test_generator.class_indices.keys())
        report = classification_report(true_classes, predicted_classes, target_names=class_labels)
        print("Classification Report:")
        print(report)
        
        # Confusion matrix
        cm = confusion_matrix(true_classes, predicted_classes)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.savefig('models/confusion_matrix.png')
        
        return {
            'loss': loss,
            'accuracy': accuracy,
            'auc': auc,
            'classification_report': report,
            'confusion_matrix': cm
        }
    
    def plot_training_history(self):
        """Plot the training history."""
        if self.history is None:
            print("No training history available. Train the model first.")
            return
        
        # Plot training & validation accuracy values
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(self.history.history['accuracy'])
        plt.plot(self.history.history['val_accuracy'])
        plt.title('Model Accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper left')
        
        # Plot training & validation loss values
        plt.subplot(1, 2, 2)
        plt.plot(self.history.history['loss'])
        plt.plot(self.history.history['val_loss'])
        plt.title('Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper left')
        
        plt.tight_layout()
        plt.savefig('models/training_history.png')
        plt.show()
    
    def save_model(self, filepath):
        """Save the trained model."""
        self.model.save(filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load a trained model."""
        self.model = tf.keras.models.load_model(filepath)
        print(f"Model loaded from {filepath}")
    
    def predict(self, image_path):
        """
        Make a prediction on a single image.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Prediction probability and class
        """
        # Load and preprocess the image
        img = tf.keras.preprocessing.image.load_img(
            image_path, target_size=(self.input_shape[0], self.input_shape[1])
        )
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0
        
        # Make prediction
        prediction = self.model.predict(img_array)[0][0]
        
        # Get class
        if prediction > 0.5:
            predicted_class = "drug_user"
            class_label = "Drug User"
        else:
            predicted_class = "non_user"
            class_label = "Non-User"
        
        return {
            'probability': float(prediction),
            'predicted_class': predicted_class,
            'class_label': class_label
        }
    
    def extract_features(self, image):
        """
        Extract features from an eye image for additional analysis.
        
        Args:
            image: Input image (numpy array)
            
        Returns:
            Dictionary of extracted features
        """
        # Convert to grayscale for pupil detection
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (7, 7), 0)
        
        # Detect eyes using Haar cascades
        eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        eyes = eye_cascade.detectMultiScale(blurred, 1.1, 5)
        
        features = {
            'pupil_dilation': 0.5,  # Default value
            'redness': 0.0,
            'eye_detected': len(eyes) > 0
        }
        
        if len(eyes) > 0:
            # Get the largest detected eye
            x, y, w, h = max(eyes, key=lambda eye: eye[2] * eye[3])
            
            # Extract eye region
            eye_region = gray[y:y+h, x:x+w]
            
            # Detect pupil using thresholding
            _, threshold = cv2.threshold(eye_region, 50, 255, cv2.THRESH_BINARY_INV)
            
            # Find contours
            contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                # Get the largest contour (assumed to be the pupil)
                pupil = max(contours, key=cv2.contourArea)
                
                # Calculate pupil area relative to eye area
                pupil_area = cv2.contourArea(pupil)
                eye_area = w * h
                features['pupil_dilation'] = pupil_area / eye_area if eye_area > 0 else 0.5
            
            # Calculate redness (in the original image)
            eye_color = image[y:y+h, x:x+w]
            b, g, r = cv2.split(eye_color)
            
            # Redness is high when red channel is higher than blue and green
            red_mask = (r > g) & (r > b)
            features['redness'] = np.sum(red_mask) / (w * h) if (w * h) > 0 else 0.0
        
        return features

if __name__ == "__main__":
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    # Initialize the model
    model = EyeAnalysisModel()
    
    # Train the model
    print("Training the model...")
    model.train(
        train_dir='datasets/eye_images/train',
        validation_dir='datasets/eye_images/validation',
        epochs=20,
        batch_size=32
    )
    
    # Plot training history
    model.plot_training_history()
    
    # Evaluate the model
    print("\nEvaluating the model...")
    evaluation = model.evaluate(test_dir='datasets/eye_images/test')
    
    # Save the model
    model.save_model('models/eye_model.h5')
    
    # Test prediction on a sample image
    print("\nTesting prediction on a sample image...")
    sample_image = 'datasets/eye_images/test/drug_user/drug_user_0.jpg'
    if os.path.exists(sample_image):
        prediction = model.predict(sample_image)
        print(f"Prediction: {prediction['class_label']} with probability {prediction['probability']:.4f}")
    else:
        print(f"Sample image {sample_image} not found.")