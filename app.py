import streamlit as st
import torch
import torch.nn.functional as F
from torchvision import models, transforms
from torch import nn
import numpy as np
from PIL import Image
import json
import os
import base64

# --- UTILITIES ---
# Define standard normalization for pre-trained models on ImageNet
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
TARGET_SIZE = 224

# Set device
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Configure page (CSS remains the same)
st.set_page_config(
    page_title="AI Flower Identification",
    page_icon="🌸",
    layout="centered",
    initial_sidebar_state="auto"
)

# Vibrant, colorful, modern CSS with functional drag & drop (CSS included for completeness)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=Inter:wght@400;500;600;700&display=swap');

    * {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Hide Streamlit Elements */
    .stDeployButton, footer, #MainMenu, .stDecoration {
        display: none;
    }
    
    /* Vibrant gradient background */
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 25%, #f093fb 50%, #4facfe 75%, #00f2fe 100%);
        background-size: 400% 400%;
        animation: gradientShift 15s ease infinite;
    }
    
    @keyframes gradientShift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    /* Main Container */
    .main .block-container {
        max-width: 950px;
        padding: 2.5rem 2rem 5rem 2rem;
    }

    /* Vibrant Header */
    .header {
        text-align: center;
        margin-bottom: 3rem;
        animation: fadeInDown 0.8s ease-out;
    }
    .header h1 {
        font-family: 'Space Grotesk', sans-serif;
        font-weight: 700;
        font-size: 4rem;
        color: #ffffff;
        text-shadow: 0 4px 20px rgba(0, 0, 0, 0.3), 0 0 40px rgba(255, 255, 255, 0.2);
        letter-spacing: -0.03em;
        margin-bottom: 1rem;
        line-height: 1.1;
        animation: glow 3s ease-in-out infinite;
    }
    @keyframes glow {
        0%, 100% { text-shadow: 0 4px 20px rgba(0, 0, 0, 0.3), 0 0 40px rgba(255, 255, 255, 0.3); }
        50% { text-shadow: 0 4px 20px rgba(0, 0, 0, 0.3), 0 0 60px rgba(255, 255, 255, 0.5); }
    }
    .header p {
        color: rgba(255, 255, 255, 0.95);
        font-size: 1.25rem;
        font-weight: 500;
        max-width: 650px;
        margin: 0 auto;
        line-height: 1.7;
        text-shadow: 0 2px 10px rgba(0, 0, 0, 0.2);
    }
    
    /* Upload Area */
    .upload-container {
        margin: 2rem 0;
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(20px);
        border: 3px solid rgba(102, 126, 234, 0.3);
        border-radius: 30px;
        padding: 3.5rem 2rem;
        text-align: center;
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: 0 20px 60px rgba(0, 0, 0, 0.2);
        position: relative;
        overflow: hidden;
    }
    .upload-container::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: conic-gradient(from 0deg, #667eea, #764ba2, #f093fb, #4facfe, #00f2fe, #667eea);
        opacity: 0;
        transition: opacity 0.4s ease;
        animation: rotate 4s linear infinite;
    }
    @keyframes rotate {
        100% { transform: rotate(360deg); }
    }
    .upload-container::after {
        content: '';
        position: absolute;
        inset: 3px;
        background: rgba(255, 255, 255, 0.95);
        border-radius: 27px;
        z-index: 1;
    }
    .upload-container:hover {
        border-color: #f093fb;
        transform: translateY(-5px) scale(1.02);
        box-shadow: 0 30px 80px rgba(0, 0, 0, 0.3);
    }
    .upload-container:hover::before {
        opacity: 1;
    }
    
    .upload-icon {
        font-size: 4rem;
        margin-bottom: 1.5rem;
        position: relative;
        z-index: 2;
        animation: bounce 2s ease-in-out infinite;
    }
    @keyframes bounce {
        0%, 100% { transform: translateY(0); }
        50% { transform: translateY(-15px); }
    }
    .upload-text {
        position: relative;
        z-index: 2;
        font-size: 1.3rem;
        font-weight: 600;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0.75rem;
    }

    /* Uploaded Image */
    .stImage {
        border-radius: 30px;
        overflow: hidden;
        box-shadow: 0 20px 50px rgba(0, 0, 0, 0.3);
        margin: 2.5rem 0;
        animation: zoomIn 0.6s ease-out;
        border: 4px solid rgba(255, 255, 255, 0.8);
    }
    .stImage img {
        border-radius: 30px;
    }
    @keyframes zoomIn {
        from {
            opacity: 0;
            transform: scale(0.8);
        }
        to {
            opacity: 1;
            transform: scale(1);
        }
    }

    /* Results Container */
    .results-container {
        margin-top: 3rem;
        animation: fadeInUp 0.7s ease-out;
    }
    
    /* Top Prediction - Vibrant Card */
    .prediction-card {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.98) 0%, rgba(255, 255, 255, 0.95) 100%);
        backdrop-filter: blur(25px);
        padding: 3.5rem 2.5rem;
        border-radius: 35px;
        margin-bottom: 2.5rem;
        border: 3px solid rgba(255, 255, 255, 0.6);
        box-shadow: 0 25px 70px rgba(0, 0, 0, 0.25), inset 0 0 50px rgba(255, 255, 255, 0.1);
        text-align: center;
        position: relative;
        overflow: hidden;
        transition: all 0.4s ease;
    }
    .prediction-card::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: conic-gradient(from 0deg, 
            rgba(102, 126, 234, 0.15), 
            rgba(118, 75, 162, 0.15), 
            rgba(240, 147, 251, 0.15), 
            rgba(79, 172, 254, 0.15), 
            rgba(0, 242, 254, 0.15),
            rgba(102, 126, 234, 0.15));
        animation: rotate 6s linear infinite;
        z-index: 0;
    }
    .prediction-card:hover {
        transform: translateY(-8px) scale(1.02);
        box-shadow: 0 35px 90px rgba(0, 0, 0, 0.35);
    }
    .prediction-card h2 {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 3.25rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 25%, #f093fb 50%, #4facfe 75%, #00f2fe 100%);
        background-size: 200% auto;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-transform: capitalize;
        margin-bottom: 2rem;
        letter-spacing: -0.02em;
        position: relative;
        z-index: 1;
        animation: gradientText 3s ease infinite;
    }
    @keyframes gradientText {
        0%, 100% { background-position: 0% center; }
        50% { background-position: 100% center; }
    }
    
    /* Colorful Confidence Bar */
    .confidence-bar-container {
        background: linear-gradient(90deg, #e0e7ff 0%, #fce7f3 50%, #dbeafe 100%);
        border-radius: 100px;
        height: 18px;
        margin: 2rem auto;
        width: 90%;
        box-shadow: inset 0 3px 8px rgba(0, 0, 0, 0.1);
        position: relative;
        z-index: 1;
        overflow: hidden;
        border: 2px solid rgba(255, 255, 255, 0.8);
    }
    .confidence-bar {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 20%, #f093fb 40%, #4facfe 60%, #00f2fe 80%, #43e97b 100%);
        background-size: 200% 100%;
        height: 100%;
        border-radius: 100px;
        box-shadow: 0 0 30px rgba(102, 126, 234, 0.6);
        transition: width 1.2s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        animation: gradientFlow 3s ease infinite;
    }
    @keyframes gradientFlow {
        0%, 100% { background-position: 0% center; }
        50% { background-position: 100% center; }
    }
    .confidence-bar::after {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.5), transparent);
        animation: shimmer 2.5s infinite;
    }
    @keyframes shimmer {
        100% { left: 100%; }
    }
    .confidence-text {
        font-weight: 700;
        font-size: 1.25rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-top: 1rem;
        position: relative;
        z-index: 1;
        letter-spacing: 0.02em;
    }
    
    /* Other Predictions - Colorful Cards */
    .other-predictions {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(20px);
        padding: 2.5rem;
        border-radius: 30px;
        border: 2px solid rgba(255, 255, 255, 0.6);
        box-shadow: 0 20px 60px rgba(0, 0, 0, 0.2);
    }
    .other-predictions h3 {
        text-align: center;
        font-family: 'Space Grotesk', sans-serif;
        font-weight: 700;
        font-size: 1.75rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 1.75rem;
        letter-spacing: -0.01em;
    }
    .prediction-item {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 1rem;
        padding: 1.25rem 1.5rem;
        border-radius: 20px;
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.9) 0%, rgba(255, 255, 255, 0.7) 100%);
        border: 2px solid transparent;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        cursor: pointer;
        position: relative;
        overflow: hidden;
    }
    .prediction-item::before {
        content: '';
        position: absolute;
        inset: 0;
        border-radius: 20px;
        padding: 2px;
        background: linear-gradient(135deg, #667eea, #764ba2, #f093fb, #4facfe, #00f2fe);
        -webkit-mask: linear-gradient(#fff 0 0) content-box, linear-gradient(#fff 0 0);
        -webkit-mask-composite: xor;
        mask-composite: exclude;
        opacity: 0;
        transition: opacity 0.3s ease;
    }
    .prediction-item:hover {
        transform: translateX(12px) scale(1.02);
        background: rgba(255, 255, 255, 0.98);
        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.3);
    }
    .prediction-item:hover::before {
        opacity: 1;
    }
    .prediction-item:last-child {
        margin-bottom: 0;
    }
    .prediction-name {
        font-weight: 600;
        color: #1e293b;
        text-transform: capitalize;
        font-size: 1.15rem;
    }
    .prediction-confidence {
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        min-width: 70px;
        text-align: right;
        font-size: 1.1rem;
    }
    
    /* Low Confidence Card Style */
    .low-confidence-card {
        background: #fef3c7; 
        border: 3px solid #fbbf24;
        padding: 3.5rem 2.5rem;
        border-radius: 35px;
        margin-bottom: 2.5rem;
        box-shadow: 0 25px 70px rgba(251, 191, 36, 0.3);
        text-align: center;
    }
    .low-confidence-card h2 {
        color: #d97706 !important; 
        font-size: 2.5rem !important;
        margin-bottom: 1rem !important;
    }
    .low-confidence-card p {
        color: #d97706; 
        font-size: 1.1rem; 
        font-weight: 500;
        line-height: 1.6;
    }

    /* Spinner */
    .stSpinner > div {
        border-color: rgba(255, 255, 255, 0.8) transparent transparent transparent !important;
    }
    
    /* File uploader styling */
    [data-testid="stFileUploader"] {
        position: relative;
        z-index: 2;
    }
    [data-testid="stFileUploader"] > div {
        padding: 0;
        border: none !important;
    }
    [data-testid="stFileUploader"] label {
        display: none;
    }
    [data-testid="stFileUploader"] > div > div {
        background: transparent !important;
        border: none !important;
    }
    
    /* Camera Input Styling */
    [data-testid="stCameraInput"] > div {
        padding: 0;
        border: none !important;
    }
    [data-testid="stCameraInput"] > div > div {
        background: transparent !important;
        border: none !important;
    }
</style>
""", unsafe_allow_html=True)

# --- MODEL LOADING (PYTORCH) ---
@st.cache_resource
def load_resources():
    """Load the model and class names, caching them for performance."""
    try:
        # 1. Load Class Names
        with open("cat_to_name.json", "r") as f:
            class_names_map = json.load(f)
        flower_classes = ["" for _ in range(102)]
        for key, name in class_names_map.items():
            index = int(key) - 1
            if 0 <= index < 102:
                flower_classes[index] = name
        
        for i, name in enumerate(flower_classes):
            if name == "":
                # Fallback for any missing indices in the original mapping
                flower_classes[i] = f"Unnamed Class {i+1}"
                
    except Exception as e:
        st.error(f"❌ Failed to load or process class names from 'cat_to_name.json': {e}")
        return None, None

    # 2. Load PyTorch Model
    model_path = 'densenet201_best_model.pth' # PyTorch standard suffix
        
    if os.path.exists(model_path):
        try:
            # Initialize a fresh DenseNet201 model
            model = models.densenet201(weights=None)
            
            # Replace the final fully connected layer (classifier) for 102 classes
            num_ftrs = model.classifier.in_features
            
            # Initialize classifier as a simple Linear layer
            model.classifier = torch.nn.Linear(num_ftrs, 102)

            # Load the state dictionary
            state_dict = torch.load(model_path, map_location=DEVICE)
            
            # 🔑 KEY MAPPING FIX: Rename keys from 'classifier.0.weight' to 'classifier.weight' 
            # and 'classifier.0.bias' to 'classifier.bias' to solve the error.
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith('classifier.0.'):
                    # Rename the keys to remove the '.0' index
                    new_key = k.replace('classifier.0.', 'classifier.')
                    new_state_dict[new_key] = v
                else:
                    new_state_dict[k] = v

            # Load the state dictionary with corrected keys
            model.load_state_dict(new_state_dict)
            
            # Set model to evaluation mode and move to device
            model = model.to(DEVICE)
            model.eval()
            
            return model, flower_classes
        except Exception as e:
            st.error(f"❌ Error loading PyTorch model file '{model_path}': {str(e)}")
            st.error("Please ensure the model architecture in app.py matches the saved model structure.")
            return None, None
    else:
        st.error(f"❌ Could not find the PyTorch model file: '{model_path}'.")
        st.error("Please ensure the model is placed in the project folder and is named 'densenet201_best_model.pth'.")
        return None, None

# --- IMAGE PROCESSING & PREDICTION (PYTORCH) ---
# PyTorch preprocessing pipeline
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(TARGET_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
])

def preprocess_image(image: Image.Image) -> torch.Tensor:
    """Preprocesses the image for the DenseNet201 model using torchvision transforms."""
    input_tensor = preprocess(image)
    input_batch = input_tensor.unsqueeze(0) 
    return input_batch.to(DEVICE)


def predict_flower(model, image, flower_classes):
    """Makes a prediction and returns the top 5 results using PyTorch."""
    processed_img = preprocess_image(image)
    
    with torch.no_grad(): # Disable gradient calculations during inference
        output = model(processed_img)
    
    # Apply softmax to get probabilities
    probabilities = F.softmax(output, dim=1).cpu().numpy()[0]
    
    # Get top 5 predictions
    top_indices = np.argsort(probabilities)[-5:][::-1]
    
    top_results = [
        (flower_classes[i], float(probabilities[i])) for i in top_indices
    ]
    return top_results

# --- MAIN APP LAYOUT ---

# Load model and class names once
model, flower_classes = load_resources()

if model is None or flower_classes is None:
    st.stop()

# Header
st.markdown("""
<div class="header">
    <h1>🌺 AI Flower Identifier</h1>
    <p>Upload a photo or use your camera to discover the species among 102 beautiful flowers with cutting-edge AI technology.</p>
</div>
""", unsafe_allow_html=True)

# Add Confidence Threshold Slider to Sidebar
confidence_threshold = st.sidebar.slider(
    'Prediction Confidence Threshold',
    min_value=0.50,
    max_value=1.00,
    value=0.80, # Default threshold as suggested by best practice
    step=0.01,
    help="Adjust the minimum confidence level required for the main prediction card to be displayed."
)

# Upload area with decorative styling
st.markdown("""
<div class="upload-container">
    <div class="upload-icon">🎨</div>
    <div class="upload-text">Upload File</div>
</div>
""", unsafe_allow_html=True)

# File Uploader
col1, = st.columns(1)

with col1:
    # 🌟 MODIFICATION HERE: Added 'webp' to the allowed types 🌟
    uploaded_file = st.file_uploader(
        "Choose an image",
        type=['jpg', 'jpeg', 'png', 'webp'],
        label_visibility="collapsed"
    )

# Determine which input to use
input_file = uploaded_file if uploaded_file is not None else None

# Perform prediction and display results
if input_file is not None:
    try:
        # Open and ensure the image is RGB before processing
        # PIL (Pillow) handles most common image formats, including webp, and converts to RGB.
        image_to_process = Image.open(input_file).convert("RGB")
        st.image(image_to_process, caption="✨ Your Beautiful Flower", use_container_width=True)
        
        with st.spinner("🎨 AI is analyzing your stunning flower..."):
            top_results = predict_flower(model, image_to_process, flower_classes)

        st.markdown('<div class="results-container">', unsafe_allow_html=True)

        # Apply Confidence Threshold Logic
        top_flower, top_confidence = top_results[0]
        
        if top_confidence >= confidence_threshold:
            # Display the top prediction (High Confidence)
            st.markdown(f"""
            <div class="prediction-card">
                <h2>{top_flower}</h2>
                <div class="confidence-bar-container">
                    <div class="confidence-bar" style="width: {top_confidence:.1%};"></div>
                </div>
                <div class="confidence-text">Confidence: {top_confidence:.1%}</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            # Display Low Confidence Message
            st.markdown(f"""
            <div class="low-confidence-card">
                <h2>🤔 Prediction Too Low</h2>
                <p>The highest prediction confidence of **{top_confidence:.1%}** is below your set threshold of **{confidence_threshold:.1%}**.</p>
                <p>Please try a clearer image or adjust the **Confidence Threshold** slider in the sidebar.</p>
            </div>
            """, unsafe_allow_html=True)

        # Display other predictions (Ranks 2 to 5)
        if len(top_results) > 1: 
            st.markdown('<div class="other-predictions"><h3>🌈 Other Top Predictions (Ranks 2-5)</h3>', unsafe_allow_html=True)
            # Iterate through results starting from the second element (index 1)
            for flower_name, confidence in top_results[1:]: 
                st.markdown(f"""
                <div class="prediction-item">
                    <span class="prediction-name">{flower_name}</span>
                    <span class="prediction-confidence">{confidence:.1%}</span>
                </div>
                """, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)
        
    # Add Robust Error Handling
    except Exception as e:
        st.error(f"❌ An error occurred during image processing or prediction.")
        st.exception(e)
        st.warning("Please ensure the input file is a valid image (JPG, JPEG, PNG, or WEBP) and the PyTorch model is correctly loaded.")