from kaggle_dataset_integration import KaggleDatasetIntegration
import os
import sys
import numpy as np
import pandas as pd
import tensorflow as tf
from flask import Flask, request, jsonify, render_template, send_from_directory
from werkzeug.utils import secure_filename
from PIL import Image
import io
import base64
import cv2
import json
from datetime import datetime

# Import our drug diagnosis system
from drug_diagnosis_system import DrugDiagnosisAPI
from train_model import EyeAnalysisModel

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize the drug diagnosis API
drug_api = DrugDiagnosisAPI()

# Initialize the eye analysis model
eye_model = EyeAnalysisModel()
try:
    eye_model.load_model('models/eye_model.h5')
    print("Eye analysis model loaded successfully.")
except:
    print("Eye analysis model not found. Please train the model first.")
    eye_model = None

@app.route('/')
def index():
    """Render the main page."""
    return render_template('index.html')
@app.route('/api/analyze-eye', methods=['POST'])
def analyze_eye():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file:
        # Read the image file
        file_bytes = file.read()
        
        # Convert to numpy array
        npimg = np.frombuffer(file_bytes, np.uint8)
        
        # Decode image
        image = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
        
        if image is None:
            return jsonify({'error': 'Invalid image file'}), 400
        
        # Analyze the eye image
        if eye_model:
            # Save the image temporarily
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            filename = f"eye_{timestamp}.jpg"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            cv2.imwrite(filepath, image)
            
            # Make prediction
            prediction_result = eye_model.predict(filepath)
            
            # Extract features
            features = eye_model.extract_features(image)
            
            # Determine risk level based on prediction probability
            prob = prediction_result['probability']
            if prob < 0.3:
                risk_level = "Low"
                risk_class = "risk-low"
            elif prob < 0.7:
                risk_level = "Medium"
                risk_class = "risk-medium"
            else:
                risk_level = "High"
                risk_class = "risk-high"
            
            # Generate possible substances based on features
            possible_substances = []
            
            # High pupil dilation suggests stimulants
            if features['pupil_dilation'] > 0.6:
                possible_substances.extend(["Amphetamines", "Cocaine", "MDMA"])
            
            # Low pupil dilation suggests opioids
            if features['pupil_dilation'] < 0.4:
                possible_substances.extend(["Heroin", "Opioids", "Morphine"])
            
            # Redness suggests marijuana
            if features['redness'] > 0.3:
                possible_substances.append("Marijuana")
            
            # Remove duplicates and limit to top 3
            possible_substances = list(set(possible_substances))[:3]
            
            if not possible_substances:
                possible_substances = ["None detected"]
            
            # Generate precautions based on risk level
            if risk_level == "Low":
                precautions = [
                    "Maintain a healthy lifestyle",
                    "Regular health check-ups are recommended",
                    "Avoid exposure to harmful substances"
                ]
            elif risk_level == "Medium":
                precautions = [
                    "Avoid driving or operating heavy machinery",
                    "Stay hydrated and get adequate rest",
                    "Consult a healthcare professional for proper diagnosis",
                    "Avoid mixing with other substances"
                ]
            else:  # High
                precautions = [
                    "Seek immediate medical attention",
                    "Do not drive or operate machinery",
                    "Stay in a safe environment with trusted individuals",
                    "Contact emergency services if experiencing severe symptoms",
                    "Avoid any additional substances"
                ]
            
            # Get drug information from the Kaggle dataset
            drug_info = {}
            drug_image = None
            
            # Try to get information for each possible substance
            for substance in possible_substances:
                if substance != "None detected":
                    # Get drug image from Kaggle dataset
                    drug_image_base64 = kaggle_integration.get_drug_image_base64(substance.lower())
                    if drug_image_base64:
                        drug_info[substance] = {
                            'image': drug_image_base64,
                            'description': f"Image of {substance}"
                        }
            
            # Prepare the response
            response = {
                'detection_status': 'Signs of drug use detected' if prob > 0.5 else 'No significant signs of drug use detected',
                'risk_level': risk_level,
                'risk_class': risk_class,
                'confidence': f"{int(prob * 100)}%",
                'possible_substances': ", ".join(possible_substances),
                'precautions': precautions,
                'features': {
                    'pupil_dilation': f"{features['pupil_dilation']:.2f}",
                    'redness': f"{features['redness']:.2f}",
                    'eye_detected': features['eye_detected']
                },
                'drug_info': drug_info
            }
            
            # Clean up the temporary file
            os.remove(filepath)
            
            return jsonify(response)
        else:
            return jsonify({'error': 'Eye analysis model not available'}), 500

@app.route('/api/diagnose-symptoms', methods=['POST'])
def diagnose_symptoms():
    """API endpoint for symptom-based diagnosis."""
    data = request.get_json()
    symptoms = data.get('symptoms', [])
    
    if not symptoms:
        return jsonify({'error': 'No symptoms provided'}), 400
    
    try:
        result = drug_api.diagnose(symptoms)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/symptoms', methods=['GET'])
def get_symptoms():
    """API endpoint to get all available symptoms."""
    return jsonify(drug_api.get_symptoms())

@app.route('/api/drug-types', methods=['GET'])
def get_drug_types():
    """API endpoint to get all available drug types."""
    return jsonify(drug_api.get_drug_types())

@app.route('/api/evaluate', methods=['GET'])
def evaluate_system():
    """API endpoint to evaluate system accuracy."""
    return jsonify(drug_api.evaluate_system())

@app.route('/api/generate-dataset', methods=['POST'])
def generate_dataset():
    """API endpoint to generate a synthetic dataset."""
    data = request.get_json()
    num_samples = data.get('num_samples', 1000)
    
    try:
        dataset = drug_api.generate_dataset(num_samples)
        
        # Convert to CSV and return
        csv_data = dataset.to_csv(index=False)
        
        # Create a response with the CSV data
        response = {
            'status': 'success',
            'message': f'Dataset with {num_samples} samples generated successfully',
            'data': csv_data
        }
        
        return jsonify(response)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/save-knowledge-base', methods=['POST'])
def save_knowledge_base():
    """API endpoint to save the knowledge base."""
    try:
        drug_api.save_knowledge_base('drug_knowledge_base.json')
        return jsonify({'status': 'success', 'message': 'Knowledge base saved successfully'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/load-knowledge-base', methods=['POST'])
def load_knowledge_base():
    """API endpoint to load the knowledge base."""
    try:
        drug_api.load_knowledge_base('drug_knowledge_base.json')
        return jsonify({'status': 'success', 'message': 'Knowledge base loaded successfully'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """API endpoint for health check."""
    return jsonify({'status': 'healthy', 'timestamp': datetime.now().isoformat()})

if __name__ == '__main__':
    app.run(debug=True)
    
    @app.route('/api/drug-info/<drug_name>', methods=['GET'])
    def get_drug_info(drug_name):
        """Get information about a drug from the Kaggle dataset."""
    try:
        # Get drug image from Kaggle dataset
        drug_image_base64 = kaggle_integration.get_drug_image_base64(drug_name.lower())
        
        if drug_image_base64:
            return jsonify({
                'drug_name': drug_name,
                'image': drug_image_base64,
                'description': f"Image of {drug_name}"
            })
        else:
            return jsonify({'error': f'No image found for {drug_name}'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500 
    
@app.route('/api/available-drugs', methods=['GET'])
def get_available_drugs():
    """Get list of available drugs in the Kaggle dataset."""
    try:
        available_drugs = kaggle_integration.get_available_drugs()
        return jsonify({'available_drugs': available_drugs})
    except Exception as e:
        return jsonify({'error': str(e)}), 500    