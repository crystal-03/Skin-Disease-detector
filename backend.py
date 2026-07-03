"""
Skin Disease Detection Backend - Flask API
File: app.py
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import io
import json
import os
from datetime import datetime

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend communication

# Configuration
MODEL_PATH = 'models/resnet50_skin_disease_final.keras'
CLASS_NAMES_PATH = 'class_names.json'
UPLOAD_FOLDER = 'uploads'
IMG_HEIGHT = 224
IMG_WIDTH = 224

# Create upload folder
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Disease information database
DISEASE_INFO = {
    'Acne': {
        'name': 'Acne',
        'description': 'Acne is a common skin condition caused by clogged hair follicles with oil and dead skin cells.',
        'symptoms': [
            'Whiteheads (closed plugged pores)',
            'Blackheads (open plugged pores)',
            'Small red, tender bumps (papules)',
            'Pimples with pus at their tips',
            'Painful lumps beneath the skin'
        ],
        'treatment': [
            'Topical retinoids (tretinoin, adapalene)',
            'Benzoyl peroxide',
            'Topical or oral antibiotics',
            'Salicylic acid cleansers',
            'Maintain good skincare routine'
        ],
        'prevention': [
            'Wash face twice daily with gentle cleanser',
            'Avoid touching face frequently',
            'Remove makeup before bed',
            'Use non-comedogenic products',
            'Avoid picking or squeezing pimples'
        ],
        'severity': 'Mild to Moderate',
        'urgency': 'low',
        'color': '#FFA726'
    },
    'Basal cell Carcinoma': {
        'name': 'Basal Cell Carcinoma',
        'description': 'The most common type of skin cancer, usually appearing on sun-exposed areas.',
        'symptoms': [
            'Pearly or waxy bump',
            'Flat, flesh-colored or brown lesion',
            'Bleeding or scabbing sore that heals and returns',
            'Pink growth with elevated border',
            'White, waxy, scar-like lesion'
        ],
        'treatment': [
            'Surgical excision',
            'Mohs surgery',
            'Radiation therapy',
            'Topical medications (imiquimod, 5-fluorouracil)',
            'Cryotherapy'
        ],
        'prevention': [
            'Avoid sun exposure during peak hours (10am-4pm)',
            'Use broad-spectrum sunscreen (SPF 30+)',
            'Wear protective clothing',
            'Regular skin self-examinations',
            'Annual dermatologist checkups'
        ],
        'severity': 'Moderate - Requires Medical Treatment',
        'urgency': 'medium',
        'color': '#EF5350'
    },
    'Contact Dermatitis': {
        'name': 'Contact Dermatitis',
        'description': 'Skin inflammation caused by contact with allergens or irritants.',
        'symptoms': [
            'Red rash or bumps',
            'Itching (can be severe)',
            'Dry, cracked, scaly skin',
            'Blisters and draining fluid',
            'Burning or tenderness'
        ],
        'treatment': [
            'Identify and avoid the irritant/allergen',
            'Topical corticosteroid creams',
            'Oral antihistamines for itching',
            'Cool, wet compresses',
            'Gentle, fragrance-free moisturizers'
        ],
        'prevention': [
            'Identify and avoid triggers',
            'Use hypoallergenic products',
            'Wear protective gloves when handling irritants',
            'Apply barrier creams',
            'Patch test new products'
        ],
        'severity': 'Mild to Moderate',
        'urgency': 'low',
        'color': '#66BB6A'
    },
    'Eczema': {
        'name': 'Eczema (Atopic Dermatitis)',
        'description': 'A chronic inflammatory skin condition causing itchy, red, and dry skin.',
        'symptoms': [
            'Dry, sensitive skin',
            'Intense itching',
            'Red, inflamed skin',
            'Dark colored patches',
            'Rough, leathery or scaly patches',
            'Oozing or crusting'
        ],
        'treatment': [
            'Daily moisturizing with thick creams',
            'Topical corticosteroids',
            'Topical calcineurin inhibitors',
            'Antihistamines for itching',
            'Phototherapy (light therapy)',
            'Avoid triggers and irritants'
        ],
        'prevention': [
            'Moisturize regularly (2-3 times daily)',
            'Take lukewarm baths',
            'Use mild, fragrance-free soaps',
            'Wear soft, breathable fabrics',
            'Manage stress levels',
            'Identify and avoid triggers'
        ],
        'severity': 'Mild to Severe',
        'urgency': 'low',
        'color': '#42A5F5'
    },
    'Melanoma': {
        'name': 'Melanoma',
        'description': 'The most serious type of skin cancer that develops in melanocytes.',
        'symptoms': [
            'New unusual growth or mole',
            'Change in existing mole',
            'Asymmetry in shape',
            'Irregular borders',
            'Multiple colors',
            'Diameter larger than 6mm',
            'Evolving size, shape, or color'
        ],
        'treatment': [
            'Surgical removal (primary treatment)',
            'Lymph node biopsy',
            'Immunotherapy',
            'Targeted therapy',
            'Radiation therapy',
            'Chemotherapy (advanced cases)'
        ],
        'prevention': [
            'Avoid UV exposure (sun and tanning beds)',
            'Use broad-spectrum sunscreen daily',
            'Wear protective clothing and hats',
            'Seek shade during peak sun hours',
            'Monthly skin self-examinations',
            'Annual professional skin checks'
        ],
        'severity': 'Severe - Requires IMMEDIATE Medical Attention',
        'urgency': 'critical',
        'color': '#D32F2F'
    },
    'Psoriasis': {
        'name': 'Psoriasis',
        'description': 'An autoimmune condition causing rapid buildup of skin cells forming scales and red patches.',
        'symptoms': [
            'Red patches covered with thick, silvery scales',
            'Dry, cracked skin that may bleed',
            'Itching, burning or soreness',
            'Thickened or ridged nails',
            'Swollen and stiff joints'
        ],
        'treatment': [
            'Topical corticosteroids',
            'Vitamin D analogues',
            'Topical retinoids',
            'Phototherapy (UVB light)',
            'Systemic medications (methotrexate, cyclosporine)',
            'Biologic drugs'
        ],
        'prevention': [
            'Avoid triggers (stress, infections, skin injuries)',
            'Moisturize regularly',
            'Avoid alcohol and smoking',
            'Maintain healthy weight',
            'Get adequate vitamin D',
            'Manage stress'
        ],
        'severity': 'Moderate to Severe',
        'urgency': 'medium',
        'color': '#AB47BC'
    },
    'Rosacea': {
        'name': 'Rosacea',
        'description': 'A chronic inflammatory skin condition causing facial redness and visible blood vessels.',
        'symptoms': [
            'Facial redness (flushing)',
            'Visible blood vessels',
            'Swollen, red bumps (may contain pus)',
            'Eye problems (dryness, irritation)',
            'Enlarged nose (rhinophyma)',
            'Burning or stinging sensation'
        ],
        'treatment': [
            'Topical medications (metronidazole, azelaic acid)',
            'Oral antibiotics (doxycycline)',
            'Laser therapy',
            'IPL (Intense Pulsed Light) therapy',
            'Avoid triggers',
            'Gentle skincare routine'
        ],
        'prevention': [
            'Identify and avoid triggers',
            'Protect skin from sun',
            'Use gentle, fragrance-free products',
            'Avoid hot beverages and spicy foods',
            'Manage stress',
            'Avoid alcohol'
        ],
        'severity': 'Mild to Moderate',
        'urgency': 'low',
        'color': '#FF7043'
    }
}

# Load model and class names
print("Loading model...")
try:
    model = load_model(MODEL_PATH)
    print(f"✓ Model loaded successfully from {MODEL_PATH}")
except Exception as e:
    print(f"✗ Error loading model: {e}")
    model = None

try:
    with open(CLASS_NAMES_PATH, 'r') as f:
        class_names = json.load(f)
    print(f"✓ Class names loaded: {class_names}")
except Exception as e:
    print(f"✗ Error loading class names: {e}")
    class_names = ['Acne', 'Basal cell Carcinoma', 'Contact Dermatitis', 
                   'Eczema', 'Melanoma', 'Psoriasis', 'Rosacea']

def preprocess_image(image):
    """Preprocess image for model prediction"""
    # Resize image
    image = image.resize((IMG_WIDTH, IMG_HEIGHT))
    
    # Convert to RGB if needed
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Convert to array and normalize
    img_array = np.array(image)
    img_array = img_array.astype('float32') / 255.0
    
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array

def get_prediction(image):
    """Get prediction from model"""
    if model is None:
        return None, None
    
    # Preprocess
    processed_image = preprocess_image(image)
    
    # Predict
    predictions = model.predict(processed_image, verbose=0)
    predicted_class_idx = np.argmax(predictions[0])
    confidence = float(predictions[0][predicted_class_idx])
    
    predicted_class = class_names[predicted_class_idx]
    
    # Get top 3 predictions
    top_3_idx = np.argsort(predictions[0])[-3:][::-1]
    top_3_predictions = [
        {
            'disease': class_names[idx],
            'confidence': float(predictions[0][idx])
        }
        for idx in top_3_idx
    ]
    
    return predicted_class, confidence, top_3_predictions, predictions[0]

@app.route('/')
def home():
    """Health check endpoint"""
    return jsonify({
        'status': 'running',
        'message': 'Skin Disease Detection API',
        'model_loaded': model is not None,
        'version': '1.0.0'
    })

@app.route('/api/predict', methods=['POST'])
def predict():
    """Main prediction endpoint"""
    try:
        # Check if image is in request
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400
        
        file = request.files['image']
        
        if file.filename == '':
            return jsonify({'error': 'No image selected'}), 400
        
        # Read and process image
        image_bytes = file.read()
        image = Image.open(io.BytesIO(image_bytes))
        
        # Save uploaded image (optional)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{timestamp}_{file.filename}"
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        image.save(filepath)
        
        # Get prediction
        predicted_class, confidence, top_3, all_predictions = get_prediction(image)
        
        if predicted_class is None:
            return jsonify({'error': 'Model not loaded'}), 500
        
        # Get disease info
        disease_info = DISEASE_INFO.get(predicted_class, {})
        
        # Prepare response
        response = {
            'success': True,
            'prediction': {
                'disease': predicted_class,
                'confidence': round(confidence * 100, 2),
                'confidence_score': round(confidence, 4)
            },
            'top_predictions': [
                {
                    'disease': pred['disease'],
                    'confidence': round(pred['confidence'] * 100, 2)
                }
                for pred in top_3
            ],
            'disease_info': disease_info,
            'all_probabilities': {
                class_names[i]: round(float(all_predictions[i]) * 100, 2)
                for i in range(len(class_names))
            },
            'timestamp': datetime.now().isoformat(),
            'uploaded_file': filename
        }
        
        return jsonify(response)
    
    except Exception as e:
        print(f"Error in prediction: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/diseases', methods=['GET'])
def get_diseases():
    """Get information about all diseases"""
    return jsonify({
        'success': True,
        'diseases': DISEASE_INFO
    })

@app.route('/api/disease/<disease_name>', methods=['GET'])
def get_disease_info(disease_name):
    """Get information about specific disease"""
    disease_info = DISEASE_INFO.get(disease_name)
    
    if disease_info:
        return jsonify({
            'success': True,
            'disease': disease_info
        })
    else:
        return jsonify({
            'success': False,
            'error': 'Disease not found'
        }), 404

@app.route('/api/classes', methods=['GET'])
def get_classes():
    """Get list of all disease classes"""
    return jsonify({
        'success': True,
        'classes': class_names,
        'count': len(class_names)
    })

if __name__ == '__main__':
    print("\n" + "="*60)
    print("🏥 SKIN DISEASE DETECTION API")
    print("="*60)
    print(f"Model: {'✓ Loaded' if model else '✗ Not Loaded'}")
    print(f"Classes: {len(class_names)}")
    print(f"Upload folder: {UPLOAD_FOLDER}")
    print("="*60)
    print("\nStarting server...")
    print("API will be available at: http://localhost:5000")
    print("="*60 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)