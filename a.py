import streamlit as st
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import io
import os
import cv2
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import json

# Try to import TensorFlow
tf_available = True
try:
    import tensorflow as tf
except ImportError:
    tf_available = False

# Enhanced class names with emojis and descriptions
CLASS_INFO = {
    'Daisy': {'emoji': '🌼', 'description': 'Simple white petals with yellow center', 'color': '#FFD700'},
    'Dandelion': {'emoji': '🌻', 'description': 'Bright yellow composite flower', 'color': '#FFA500'},
    'Rose': {'emoji': '🌹', 'description': 'Classic layered petals, often red or pink', 'color': '#FF69B4'},
    'Sunflower': {'emoji': '🌻', 'description': 'Large yellow petals with dark center', 'color': '#FFD700'},
    'Tulip': {'emoji': '🌷', 'description': 'Cup-shaped flower with smooth petals', 'color': '#FF1493'}
}

CLASS_NAMES = list(CLASS_INFO.keys())

# Enhanced page configuration
st.set_page_config(
    page_title="🌼 Blossom AI",
    page_icon="🌼",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(90deg, #ff9a9e 0%, #fecfef 50%, #fecfef 100%);
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .prediction-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
        border-left: 4px solid #ff9a9e;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        margin: 0.5rem;
    }
    .info-box {
        background: #f0f8ff;
        border: 1px solid #e6f3ff;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Load model with caching
@st.cache_resource
def load_model():
    if not tf_available:
        return None
    
    try:
        if os.path.exists('flower_classifier.h5'):
            return tf.keras.models.load_model('flower_classifier.h5')
        else:
            return None
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Enhanced image preprocessing with augmentation options
def preprocess_image(image, augment=False):
    if not tf_available:
        return None
    
    # Convert to RGB if needed
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Optional augmentation for better predictions
    if augment:
        # Slightly enhance the image
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(1.2)
        
        enhancer = ImageEnhance.Sharpness(image)
        image = enhancer.enhance(1.1)
    
    # Resize and normalize
    image = image.resize((128, 128))
    image_array = np.array(image) / 255.0
    return np.expand_dims(image_array, axis=0)

# Create confidence visualization
def create_confidence_chart(predictions, class_names):
    fig = go.Figure(data=[
        go.Bar(
            x=[f"{CLASS_INFO[name]['emoji']} {name}" for name in class_names],
            y=predictions * 100,
            marker_color=[CLASS_INFO[name]['color'] for name in class_names],
            text=[f"{pred:.1f}%" for pred in predictions * 100],
            textposition='auto',
        )
    ])
    
    fig.update_layout(
        title="Prediction Confidence",
        xaxis_title="Flower Type",
        yaxis_title="Confidence (%)",
        yaxis=dict(range=[0, 100]),
        template="plotly_white",
        height=400
    )
    
    return fig

# Gradient analysis (simulated feature importance)
def analyze_image_features(image):
    """Simulate feature analysis for educational purposes"""
    image_array = np.array(image.resize((128, 128)))
    
    # Simulate color analysis
    avg_colors = np.mean(image_array, axis=(0, 1))
    color_dominance = {
        'Red': avg_colors[0] / 255.0,
        'Green': avg_colors[1] / 255.0,
        'Blue': avg_colors[2] / 255.0
    }
    
    # Simulate texture analysis (using edge detection)
    gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    edge_density = np.sum(edges > 0) / (128 * 128)
    
    return color_dominance, edge_density

# Main app
def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🌼 Blossom AI – Smart Flower Classifier</h1>
        <p>Advanced AI-powered flower identification with detailed analysis</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load model
    model = load_model()
    
    if model is None:
        st.error("⚠️ Model not available")
        if not tf_available:
            st.info("📦 TensorFlow not installed. Install with: `pip install tensorflow`")
        else:
            st.info("🏃‍♂️ Train the model first: `python train_flower_classifier.py`")
        return
    
    # Sidebar configuration
    st.sidebar.markdown("## 🎛️ Configuration")
    enhance_image = st.sidebar.checkbox("🔧 Enhance image quality", value=True)
    show_analysis = st.sidebar.checkbox("📊 Show detailed analysis", value=True)
    confidence_threshold = st.sidebar.slider("🎯 Confidence threshold", 0.0, 1.0, 0.5)
    
    # File upload section
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 📁 Upload Your Flower Image")
        uploaded_file = st.file_uploader(
            "Choose a flower image...", 
            type=["jpg", "jpeg", "png"],
            help="Upload a clear image of a flower for best results"
        )
    
    with col2:
        if uploaded_file:
            st.markdown("### 📊 Quick Stats")
            file_size = len(uploaded_file.getvalue())
            st.markdown(f"""
            <div class="info-box">
                <strong>File:</strong> {uploaded_file.name}<br>
                <strong>Size:</strong> {file_size/1024:.1f} KB<br>
                <strong>Type:</strong> {uploaded_file.type}
            </div>
            """, unsafe_allow_html=True)
    
    if uploaded_file is not None:
        try:
            # Load and display image
            image = Image.open(uploaded_file)
            
            # Image display section
            st.markdown("### 🖼️ Uploaded Image")
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                st.image(image, caption='Your Flower Image', use_column_width=True)
            
            # Prediction section
            st.markdown("### 🔮 AI Prediction")
            
            with st.spinner("🤖 Analyzing your flower..."):
                # Preprocess image
                processed_image = preprocess_image(image, augment=enhance_image)
                
                # Make prediction
                predictions = model.predict(processed_image, verbose=0)[0]
                
                # Get sorted predictions
                pred_indices = np.argsort(predictions)[::-1]
                
                # Main prediction
                top_pred_idx = pred_indices[0]
                top_class = CLASS_NAMES[top_pred_idx]
                top_confidence = predictions[top_pred_idx]
                
                # Display main result
                if top_confidence >= confidence_threshold:
                    st.success(f"🎉 I'm {top_confidence*100:.1f}% confident this is a **{top_class}**!")
                else:
                    st.warning(f"🤔 I think this might be a **{top_class}** ({top_confidence*100:.1f}% confidence), but I'm not very sure.")
                
                # Detailed predictions
                st.markdown("### 📈 Detailed Predictions")
                
                # Create two columns for predictions
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("#### Top 3 Predictions")
                    for i, idx in enumerate(pred_indices[:3]):
                        class_name = CLASS_NAMES[idx]
                        confidence = predictions[idx]
                        emoji = CLASS_INFO[class_name]['emoji']
                        desc = CLASS_INFO[class_name]['description']
                        
                        st.markdown(f"""
                        <div class="prediction-card">
                            <h4>{emoji} {class_name}</h4>
                            <p><strong>Confidence:</strong> {confidence*100:.2f}%</p>
                            <p><small>{desc}</small></p>
                        </div>
                        """, unsafe_allow_html=True)
                
                with col2:
                    # Confidence chart
                    fig = create_confidence_chart(predictions, CLASS_NAMES)
                    st.plotly_chart(fig, use_container_width=True)
                
                # Advanced analysis
                if show_analysis:
                    st.markdown("### 🔬 Advanced Analysis")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("#### 🎨 Color Analysis")
                        color_dom, edge_density = analyze_image_features(image)
                        
                        # Color dominance chart
                        fig_color = px.bar(
                            x=list(color_dom.keys()),
                            y=list(color_dom.values()),
                            color=list(color_dom.keys()),
                            color_discrete_map={'Red': 'red', 'Green': 'green', 'Blue': 'blue'},
                            title="Color Dominance"
                        )
                        st.plotly_chart(fig_color, use_container_width=True)
                    
                    with col2:
                        st.markdown("#### 🔍 Image Properties")
                        st.markdown(f"""
                        <div class="info-box">
                            <strong>Image Size:</strong> {image.size[0]} × {image.size[1]} pixels<br>
                            <strong>Mode:</strong> {image.mode}<br>
                            <strong>Edge Density:</strong> {edge_density:.3f}<br>
                            <strong>Texture:</strong> {"High detail" if edge_density > 0.1 else "Smooth"}
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Model confidence metrics
                        entropy = -np.sum(predictions * np.log(predictions + 1e-10))
                        max_conf = np.max(predictions)
                        
                        st.markdown("#### 🎯 Prediction Metrics")
                        st.markdown(f"""
                        <div class="info-box">
                            <strong>Prediction Entropy:</strong> {entropy:.3f}<br>
                            <strong>Max Confidence:</strong> {max_conf:.3f}<br>
                            <strong>Certainty:</strong> {"High" if max_conf > 0.8 else "Medium" if max_conf > 0.5 else "Low"}
                        </div>
                        """, unsafe_allow_html=True)
            
            # Save prediction history (in session state)
            if 'prediction_history' not in st.session_state:
                st.session_state.prediction_history = []
            
            st.session_state.prediction_history.append({
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'prediction': top_class,
                'confidence': float(top_confidence),
                'filename': uploaded_file.name
            })
            
        except Exception as e:
            st.error(f"❌ Error processing image: {e}")
            st.info("Please try with a different image or check the file format.")
    
    # Model evaluation section
    st.markdown("---")
    st.markdown("## 📊 Model Performance")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if os.path.exists('training_history.png'):
            st.markdown("### 📈 Training History")
            st.image('training_history.png', use_column_width=True)
        else:
            st.info("📈 Training history not available. Train the model first.")
    
    with col2:
        if os.path.exists('confusion_matrix.png'):
            st.markdown("### 🔄 Confusion Matrix")
            st.image('confusion_matrix.png', use_column_width=True)
        else:
            st.info("🔄 Confusion matrix not available. Train the model first.")
    
    # Prediction history
    if 'prediction_history' in st.session_state and st.session_state.prediction_history:
        st.markdown("### 📝 Prediction History")
        history_df = st.session_state.prediction_history
        
        # Convert to display format
        for record in history_df[-5:]:  # Show last 5 predictions
            st.markdown(f"**{record['timestamp']}** - {record['filename']}: **{record['prediction']}** ({record['confidence']*100:.1f}%)")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; padding: 2rem; background: #f8f9fa; border-radius: 10px;">
        <h4>🌼 Blossom AI</h4>
        <p>Built with ❤️ using Streamlit, TensorFlow, and Plotly</p>
        <p><small>Supports: Daisy, Dandelion, Rose, Sunflower, Tulip</small></p>
    </div>
    """, unsafe_allow_html=True)

# Sidebar information
def setup_sidebar():
    st.sidebar.markdown("## 🚀 How to Use")
    st.sidebar.markdown("""
    1. **Train the model**: `python train_flower_classifier.py`
    2. **Upload image**: Click 'Browse files' and select a flower image
    3. **Get predictions**: View AI analysis and confidence scores
    4. **Explore**: Enable detailed analysis for more insights
    """)
    
    st.sidebar.markdown("## 🌸 Supported Flowers")
    for name, info in CLASS_INFO.items():
        st.sidebar.markdown(f"**{info['emoji']} {name}**")
        st.sidebar.markdown(f"*{info['description']}*")
    
    st.sidebar.markdown("## 💡 Tips")
    st.sidebar.markdown("""
    - Use clear, well-lit images
    - Ensure the flower is the main subject
    - Try different angles if confidence is low
    - Enable image enhancement for better results
    """)

if __name__ == "__main__":
    setup_sidebar()
    main()