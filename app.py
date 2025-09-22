import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
import numpy as np
from PIL import Image
import cv2
import os

# Configure page
st.set_page_config(
    page_title="Flower Detection AI", 
    page_icon="🌸", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for beautiful UI
st.markdown("""
<style>
    .main-header {
        text-align: center;
        color: #ff6b6b;
        font-size: 3.5rem;
        font-weight: bold;
        margin-bottom: 1rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    
    .subtitle {
        text-align: center;
        color: #4ecdc4;
        font-size: 1.3rem;
        margin-bottom: 2rem;
    }
    
    .prediction-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 20px;
        color: white;
        text-align: center;
        margin: 1rem 0;
        box-shadow: 0 15px 35px rgba(0,0,0,0.1);
        border: 2px solid rgba(255,255,255,0.1);
    }
    
    .prediction-title {
        font-size: 1.8rem;
        font-weight: bold;
        margin-bottom: 1rem;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.3);
    }
    
    .confidence-display {
        font-size: 1.2rem;
        margin-top: 1rem;
        background: rgba(255,255,255,0.2);
        padding: 0.8rem;
        border-radius: 10px;
    }
    
    .stButton > button {
        background: linear-gradient(45deg, #ff6b6b, #4ecdc4);
        color: white;
        border: none;
        border-radius: 30px;
        padding: 0.8rem 3rem;
        font-weight: bold;
        font-size: 1.1rem;
        transition: all 0.3s ease;
        box-shadow: 0 5px 15px rgba(0,0,0,0.2);
        width: 100%;
    }
    
    .stButton > button:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.3);
    }
    
    .status-success {
        background: linear-gradient(135deg, #00b09b, #96c93d);
        color: white;
        padding: 1rem;
        border-radius: 15px;
        text-align: center;
        font-weight: bold;
        margin: 1rem 0;
        box-shadow: 0 5px 15px rgba(0,0,0,0.2);
    }
    
    .main-container {
        max-width: 1200px;
        margin: 0 auto;
        padding: 2rem;
    }
    
    /* Hide streamlit elements */
    .stDeployButton {display:none;}
    footer {visibility: hidden;}
    .stDecoration {display:none;}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_flower_class_names():
    try:
        with open("cat_to_name.json", "r") as f:
            class_names = json.load(f)
        sorted_items = sorted(class_names.items(), key=lambda x: int(x[0]))
        flower_classes = [name for _, name in sorted_items]
        return flower_classes
    except Exception as e:
        st.error(f"Failed to load class names from JSON: {e}")
        return None
    
@st.cache_resource
def load_model():
    """Load PyTorch DenseNet121 model"""
    
    with st.spinner("🤖 Loading DenseNet121 AI model..."):
        try:
            # Define flower classes
            flower_classes = load_flower_class_names()
            if not flower_classes:
                raise ValueError("Flower classes list is empty.")
            
            # Create DenseNet121 model
            model = models.densenet121(weights=None)  # Don't load pretrained weights
            
            # Modify the final layer for 5 flower classes
            num_features = model.classifier.in_features
            model.classifier = nn.Linear(num_features, 5)
            
            # Load the trained weights
            model.load_state_dict(torch.load('densenet201_oxfordflowers.h5', map_location='cpu'))
            
            # Set to evaluation mode
            model.eval()
            
            return model, flower_classes
            
        except Exception as e:
            st.error(f"❌ Could not load DenseNet121 model: {str(e)}")
            st.error("Please make sure 'flower_classification.pth' exists in the project folder.")
            return None, None

def preprocess_image(image):
    """Preprocess image for PyTorch DenseNet121"""
    
    # Define transforms (standard ImageNet preprocessing for DenseNet121)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Convert PIL image to RGB if needed
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Apply transforms
    img_tensor = transform(image)
    
    # Add batch dimension
    img_tensor = img_tensor.unsqueeze(0)
    
    return img_tensor

def predict_flower(model, flower_classes, image, confidence_threshold=0.7):
    """Make prediction with PyTorch model"""
    # Preprocess image
    processed_img = preprocess_image(image)
    
    # Make prediction
    with torch.no_grad():
        outputs = model(processed_img)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        
    # Convert to numpy for easier handling
    probabilities = probabilities.numpy()[0]
    
    # Get predicted class
    predicted_class_idx = np.argmax(probabilities)
    confidence = float(probabilities[predicted_class_idx])
    predicted_class = flower_classes[predicted_class_idx]
    
    # Check if confidence meets threshold
    meets_threshold = confidence >= confidence_threshold
    
    return predicted_class, confidence, meets_threshold, probabilities

# Main app
st.markdown('<div class="main-container">', unsafe_allow_html=True)
st.markdown('<h1 class="main-header">🌸 Flower Detection AI 🌸</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Powered by DenseNet121 Deep Learning Model</p>', unsafe_allow_html=True)

# Load model
model_result = load_model()
if model_result[0] is None:
    st.stop()

model, flower_classes = model_result

# Success message
st.markdown('<div class="status-success">🎉 AI Model Ready! Choose your input method below.</div>', unsafe_allow_html=True)

# Sidebar with settings
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    
    # Confidence threshold slider
    confidence_threshold = st.slider(
        "🎯 Confidence Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.7,  # Default to 70% for more accurate results
        step=0.05,
        help="Minimum confidence required for prediction. Higher values = more accurate but may reject some predictions."
    )
    
    st.markdown(f"**Current Setting:** {confidence_threshold:.0%}")
    
    if confidence_threshold >= 0.8:
        st.success("🎯 High Accuracy Mode")
    elif confidence_threshold >= 0.6:
        st.info("⚖️ Balanced Mode")
    else:
        st.warning("🤔 Low Confidence Mode")
    
    st.markdown("---")
    
    st.markdown("### 🌸 Flower Types")
    st.markdown("This AI can identify:")
    for i, flower in enumerate(flower_classes, 1):
        st.markdown(f"{i}. **{flower.title()}** 🌺")
    
    st.markdown("---")
    st.markdown("### 💡 Tips for Best Results")
    st.markdown("""
    - Use clear, well-lit images
    - Ensure the flower is the main subject
    - Avoid blurry or dark photos
    - Try different angles if needed
    - Higher confidence = more accurate predictions
    """)
    
    st.markdown("---")
    st.markdown("### 🔧 Model Info")
    st.markdown("""
    - **Architecture**: DenseNet121 CNN
    - **Input Size**: 224x224 pixels
    - **Model Type**: Transfer Learning
    - **Classes**: 5 flower types
    - **Depth**: 121 layers
    - **Key Feature**: Dense connections
    """)

# Input method selection
st.markdown("### 🔧 Choose Input Method")
input_method = st.radio("Select input method", ["📤 Upload Image", "📹 Use Webcam"], horizontal=True, label_visibility="collapsed")

st.markdown("---")  # Clean separator

# Main content area
main_col1, main_col2 = st.columns([2, 1], gap="large")

with main_col1:
    if input_method == "📤 Upload Image":
        st.markdown("### 📤 Upload Your Flower Image")
        st.markdown("*Drag and drop or click to browse for the best results*")
        
        uploaded_file = st.file_uploader(
            "Choose an image file", 
            type=['jpg', 'jpeg', 'png', 'bmp'], 
            help="Upload a clear image of a flower for accurate prediction",
            label_visibility="collapsed"
        )
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="🌸 Your Uploaded Image", use_container_width=True)
            
            col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
            with col_btn2:
                if st.button("🔍 Analyze Flower", use_container_width=True):
                    with st.spinner("🤖 AI is analyzing your flower..."):
                        flower_type, confidence, meets_threshold, all_predictions = predict_flower(
                            model, flower_classes, image, confidence_threshold
                        )
                        
                        # Store result in session state for display
                        st.session_state.prediction_result = (flower_type, confidence, meets_threshold, all_predictions)
                        st.rerun()
    
    elif input_method == "📹 Use Webcam":
        st.markdown("### 📹 Capture with Webcam")
        st.markdown("*Click the camera button to take a photo*")
        
        camera_image = st.camera_input("Take a picture of a flower", label_visibility="collapsed")
        
        if camera_image is not None:
            image = Image.open(camera_image)
            st.image(image, caption="📸 Your Captured Image", use_container_width=True)
            
            col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
            with col_btn2:
                if st.button("🔍 Analyze Flower", use_container_width=True):
                    with st.spinner("🤖 AI is analyzing your flower..."):
                        flower_type, confidence, meets_threshold, all_predictions = predict_flower(
                            model, flower_classes, image, confidence_threshold
                        )
                        
                        # Store result in session state for display
                        st.session_state.prediction_result = (flower_type, confidence, meets_threshold, all_predictions)
                        st.rerun()

# Display prediction result in the right column
with main_col2:
    if hasattr(st.session_state, 'prediction_result') and len(st.session_state.prediction_result) == 4:
        flower_type, confidence, meets_threshold, all_predictions = st.session_state.prediction_result
        
        if meets_threshold:
            # High confidence prediction
            st.markdown(f"""
            <div class="prediction-card">
                <div class="prediction-title">🎯 Prediction Result</div>
                <h2 style="margin: 1rem 0; font-size: 2.5rem;">🌺 {flower_type.title()}</h2>
                <div class="confidence-display" style="background: rgba(0,176,155,0.3);">
                    <strong>Confidence: {confidence:.1%}</strong>
                    <br><small>✅ Meets accuracy threshold</small>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Show top predictions
            st.markdown("### 🏆 All Predictions")
            top_indices = np.argsort(all_predictions)[-3:][::-1]
            
            for i, idx in enumerate(top_indices):
                conf = all_predictions[idx]
                is_selected = idx == np.argmax(all_predictions)
                icon = "🥇" if i == 0 else "🥈" if i == 1 else "🥉"
                
                if is_selected and meets_threshold:
                    st.success(f"{icon} **{flower_classes[idx].title()}**: {conf:.1%}")
                else:
                    st.info(f"{icon} {flower_classes[idx].title()}: {conf:.1%}")
        else:
            # Low confidence prediction
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #ff6b6b 0%, #ee5a52 100%); 
                        padding: 2rem; border-radius: 20px; color: white; text-align: center; 
                        margin: 1rem 0; box-shadow: 0 15px 35px rgba(0,0,0,0.1);">
                <div style="font-size: 1.8rem; font-weight: bold; margin-bottom: 1rem;">⚠️ Low Confidence</div>
                <h3 style="margin: 1rem 0;">Best Guess: {flower_type.title()}</h3>
                <div style="background: rgba(255,255,255,0.2); padding: 0.8rem; border-radius: 10px; margin-top: 1rem;">
                    <strong>Confidence: {confidence:.1%}</strong>
                    <br><small>Below {confidence_threshold:.0%} threshold</small>
                </div>
                <p style="margin-top: 1rem; font-size: 0.9rem;">
                    Try a clearer image or adjust the confidence threshold.
                </p>
            </div>
            """, unsafe_allow_html=True)
    else:
        # Placeholder when no prediction yet
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%); 
                    padding: 2rem; border-radius: 20px; text-align: center; 
                    color: #666; border: 2px dashed #ccc;">
            <h3>🔮 Prediction will appear here</h3>
            <p>Upload an image or use webcam to get started!</p>
            <p><small>Current threshold: {confidence_threshold:.0%}</small></p>
        </div>
        """, unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)  # Close main container

# Footer
st.markdown("---")
st.markdown(
    '<div style="text-align: center; color: #666; margin-top: 2rem;">'
    '🌸 Flower Detection AI - Powered by Deep Learning 🌸'
    '</div>', 
    unsafe_allow_html=True
)