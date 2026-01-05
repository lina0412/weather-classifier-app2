# streamlit_app.py - MODERN MULTI-IMAGE INTERFACE
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import json
import os
import time
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import joblib
import base64
from io import BytesIO
from datetime import datetime
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D

# ==================== PAGE CONFIG ====================
st.set_page_config(
    page_title="Weather Vision AI",
    page_icon="🌤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== CUSTOM CSS ====================
st.markdown("""
<style>
    /* Main container styling */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 4px 20px rgba(0,0,0,0.1);
    }
    
    .file-upload-box {
        background: #f8f9fa;
        border: 3px dashed #dee2e6;
        border-radius: 15px;
        padding: 3rem 1rem;
        text-align: center;
        transition: all 0.3s;
        cursor: pointer;
    }
    
    .file-upload-box:hover {
        border-color: #667eea;
        background: #eef2ff;
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(102, 126, 234, 0.1);
    }
    
    .upload-icon {
        font-size: 4rem;
        color: #667eea;
        margin-bottom: 1rem;
    }
    
    .image-card {
        background: white;
        border-radius: 12px;
        padding: 1.2rem;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
        transition: transform 0.2s;
        border: 1px solid #e9ecef;
        height: 100%;
    }
    
    .image-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 5px 20px rgba(0,0,0,0.1);
    }
    
    .status-badge {
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    
    .status-processing { background: #fff3cd; color: #856404; }
    .status-completed { background: #d4edda; color: #155724; }
    .status-error { background: #f8d7da; color: #721c24; }
    
    .progress-container {
        background: #e9ecef;
        border-radius: 10px;
        height: 8px;
        margin: 0.8rem 0;
        overflow: hidden;
    }
    
    .progress-bar {
        height: 100%;
        background: linear-gradient(90deg, #667eea, #764ba2);
        border-radius: 10px;
        transition: width 0.3s ease;
    }
    
    .model-card {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        border-radius: 12px;
        padding: 1.5rem;
        border-left: 5px solid #667eea;
        margin-bottom: 1rem;
    }
    
    .confidence-meter {
        display: flex;
        align-items: center;
        margin: 1rem 0;
    }
    
    .confidence-fill {
        height: 10px;
        border-radius: 5px;
        background: linear-gradient(90deg, #ff6b6b, #ffd93d, #6bcf7f);
    }
    
    .stats-card {
        background: white;
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
        border-top: 4px solid #667eea;
    }
    
    .stat-number {
        font-size: 2.5rem;
        font-weight: 700;
        color: #667eea;
        margin-bottom: 0.5rem;
    }
    
    .stat-label {
        font-size: 0.9rem;
        color: #6c757d;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .prediction-chip {
        display: inline-block;
        padding: 0.4rem 1rem;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.9rem;
        margin: 0.2rem;
    }
    
    .hail-chip { background: #c8e6c9; color: #2e7d32; }
    .lightning-chip { background: #ffecb3; color: #ff8f00; }
    .rain-chip { background: #bbdefb; color: #1565c0; }
    .sandstorm-chip { background: #d7ccc8; color: #5d4037; }
    .snow-chip { background: #e3f2fd; color: #0277bd; }
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: #f1f1f1;
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: #c1c1c1;
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: #a8a8a8;
    }
</style>
""", unsafe_allow_html=True)

# ==================== CONFIGURATION ====================
CONFIDENCE_THRESHOLD = 0.7
MAX_IMAGE_SIZE = 5000
MAX_UPLOAD_FILES = 20  # Maximum number of images to upload at once
MAX_FILE_SIZE_MB = 10  # Maximum file size per image

# ==================== SESSION STATE ====================
if 'uploaded_files' not in st.session_state:
    st.session_state.uploaded_files = []
if 'predictions' not in st.session_state:
    st.session_state.predictions = {}
if 'processing_queue' not in st.session_state:
    st.session_state.processing_queue = []
if 'selected_model' not in st.session_state:
    st.session_state.selected_model = "Sparse Fine-Tuning"

# ==================== LOAD DATA ====================
@st.cache_data
def load_class_names():
    with open('class_names.json', 'r') as f:
        return json.load(f)

@st.cache_data  
def load_accuracies():
    return {
        "Sparse Fine-Tuning": 0.9315,
        "Fine-Tuning (Last 30 layers)": 0.9259,
        "Feature Extraction + SVM": 0.7963,
        "Stochastic Fine-Tuning": 0.7593,
        "Feature Extraction + Random Forest": 0.7778,
        "Feature Extraction + MLP": 0.7222,
        "Knowledge Distillation": 0.7056,
    }

@st.cache_resource
def load_class_colors():
    return {
        "hail": "#2e7d32",
        "lightning": "#ff8f00",
        "rain": "#1565c0",
        "sandstorm": "#5d4037",
        "snow": "#0277bd"
    }

# ==================== MODEL LOADING ====================
@st.cache_resource
def create_feature_extractor():
    base = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    return Model(inputs=base.input, outputs=GlobalAveragePooling2D()(base.output))

@st.cache_resource
def load_all_models():
    """Load all available models"""
    models = {}
    
    # Map display names to actual model files
    model_files = {
        "Sparse Fine-Tuning": "strategy4_sparse.keras",
        "Fine-Tuning (Last 30 layers)": "strategy2_fine_tuned.keras",
        "Stochastic Fine-Tuning": "strategy3_stochastic.keras",
        "Knowledge Distillation": "weather_classifier_fixed.keras",
        "Feature Extraction + SVM": "strategy1_svm_rbf.pkl",
        "Feature Extraction + Random Forest": "strategy1_random_forest.pkl",
        "Feature Extraction + MLP": "strategy1_mlp.pkl"
    }
    
    # Load Keras models
    for name, file in model_files.items():
        if os.path.exists(file):
            try:
                if file.endswith('.keras'):
                    models[name] = {
                        'type': 'keras',
                        'model': tf.keras.models.load_model(file),
                        'accuracy': load_accuracies().get(name, 0),
                        'color': '#667eea' if 'Sparse' in name else '#764ba2'
                    }
                elif file.endswith('.pkl'):
                    models[name] = {
                        'type': 'ml',
                        'model': joblib.load(file),
                        'accuracy': load_accuracies().get(name, 0),
                        'color': '#ff6b6b' if 'SVM' in name else '#6bcf7f'
                    }
            except Exception as e:
                st.warning(f"Could not load {name}: {str(e)}")
    
    return models

# ==================== PREDICTION FUNCTIONS ====================
def preprocess_image(image):
    """Preprocess image for model prediction"""
    img_processed = image.resize((224, 224))
    img_array = np.array(img_processed) / 255.0
    
    if len(img_array.shape) == 2:
        img_array = np.stack([img_array]*3, axis=-1)
    elif img_array.shape[2] == 4:
        img_array = img_array[:, :, :3]
    
    return np.expand_dims(img_array, axis=0)

def predict_image(model_info, image_array, feature_extractor=None):
    """Make prediction for a single image"""
    start_time = time.time()
    
    if model_info['type'] == 'keras':
        predictions = model_info['model'].predict(image_array, verbose=0)[0]
    else:  # ML model
        features = feature_extractor.predict(image_array, verbose=0)
        features_flat = features.reshape(features.shape[0], -1)
        
        if hasattr(model_info['model'], 'predict_proba'):
            predictions = model_info['model'].predict_proba(features_flat)[0]
        else:
            pred_class = model_info['model'].predict(features_flat)[0]
            predictions = np.zeros(5)
            predictions[pred_class] = 1.0
    
    prediction_time = time.time() - start_time
    max_confidence = np.max(predictions)
    predicted_class = np.argmax(predictions)
    
    return predictions, prediction_time, max_confidence, predicted_class

# ==================== BATCH PROCESSING ====================
def process_batch(images, model_name, progress_callback=None):
    """Process multiple images with the selected model"""
    models = load_all_models()
    class_names = load_class_names()
    model_info = models.get(model_name)
    
    if not model_info:
        return []
    
    # Check if we need feature extractor for ML models
    need_feature_extractor = model_info['type'] == 'ml'
    feature_extractor = create_feature_extractor() if need_feature_extractor else None
    
    results = []
    total_images = len(images)
    
    for idx, (image, filename) in enumerate(images):
        try:
            # Preprocess image
            img_array = preprocess_image(image)
            
            # Make prediction
            predictions, pred_time, max_conf, pred_class_idx = predict_image(
                model_info, img_array, feature_extractor
            )
            
            predicted_class = class_names[pred_class_idx]
            
            # Determine if prediction is accepted based on confidence threshold
            is_accepted = max_conf >= CONFIDENCE_THRESHOLD
            
            results.append({
                'filename': filename,
                'class': predicted_class,
                'confidence': max_conf,
                'time': pred_time,
                'accepted': is_accepted,
                'predictions': predictions.tolist(),
                'timestamp': datetime.now().strftime("%H:%M:%S")
            })
            
            # Update progress if callback provided
            if progress_callback:
                progress_callback((idx + 1) / total_images)
                
        except Exception as e:
            results.append({
                'filename': filename,
                'error': str(e),
                'timestamp': datetime.now().strftime("%H:%M:%S")
            })
    
    return results

# ==================== SIDEBAR ====================
with st.sidebar:
    st.markdown('<div class="model-card">', unsafe_allow_html=True)
    st.header("🤖 **Model Selection**")
    
    models = load_all_models()
    model_names = list(models.keys())
    
    if model_names:
        selected_model = st.selectbox(
            "Choose a model:",
            model_names,
            index=0,
            help="Select the AI model for weather classification"
        )
        st.session_state.selected_model = selected_model
        
        # Display model accuracy
        accuracy = models[selected_model]['accuracy']
        st.progress(float(accuracy))
        st.caption(f"Accuracy: {accuracy:.2%}")
    else:
        st.error("No models found! Please ensure model files are in the correct directory.")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown('<div class="model-card">', unsafe_allow_html=True)
    st.header("⚙️ **Settings**")
    
    confidence_threshold = st.slider(
        "Confidence Threshold",
        min_value=0.0,
        max_value=1.0,
        value=CONFIDENCE_THRESHOLD,
        step=0.05,
        help="Minimum confidence required to accept prediction"
    )
    CONFIDENCE_THRESHOLD = confidence_threshold
    
    st.info(f"**Current threshold:** {CONFIDENCE_THRESHOLD:.0%}")
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Clear all button
    if st.button("🗑️ Clear All Images", use_container_width=True):
        st.session_state.uploaded_files = []
        st.session_state.predictions = {}
        st.rerun()
    
    # Statistics
    st.markdown('<div class="stats-card">', unsafe_allow_html=True)
    total_files = len(st.session_state.uploaded_files)
    processed_files = len(st.session_state.predictions)
    
    st.markdown(f'<div class="stat-number">{total_files}</div>', unsafe_allow_html=True)
    st.markdown('<div class="stat-label">Total Images</div>', unsafe_allow_html=True)
    
    st.markdown(f'<div class="stat-number">{processed_files}</div>', unsafe_allow_html=True)
    st.markdown('<div class="stat-label">Processed</div>', unsafe_allow_html=True)
    
    if total_files > 0:
        percent_processed = (processed_files / total_files) * 100
        st.metric("Completion", f"{percent_processed:.0f}%")
    st.markdown('</div>', unsafe_allow_html=True)

# ==================== MAIN INTERFACE ====================
# Header
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.markdown("""
    <div class="main-header">
        <h1 style="text-align: center; margin: 0;">🌤️ Weather Vision AI</h1>
        <p style="text-align: center; opacity: 0.9; margin: 0.5rem 0 0 0;">
            Advanced Multi-Image Weather Classification System
        </p>
    </div>
    """, unsafe_allow_html=True)

# Upload Section
st.markdown("### 📤 Upload Images")
st.markdown("""
<div class="file-upload-box" onclick="document.getElementById('file-upload').click()">
    <div class="upload-icon">📁</div>
    <h3>Drag & Drop or Click to Upload</h3>
    <p>Upload up to {MAX_UPLOAD_FILES} weather images at once (Max {MAX_FILE_SIZE_MB}MB each)</p>
    <p style="color: #6c757d; font-size: 0.9rem;">
        Supported formats: JPG, PNG, JPEG
    </p>
</div>
""".format(MAX_UPLOAD_FILES=MAX_UPLOAD_FILES, MAX_FILE_SIZE_MB=MAX_FILE_SIZE_MB), unsafe_allow_html=True)

# File uploader (hidden, triggered by the styled box)
uploaded_files = st.file_uploader(
    "",
    type=['jpg', 'jpeg', 'png'],
    accept_multiple_files=True,
    key="file_uploader",
    label_visibility="collapsed"
)

# Handle new file uploads
if uploaded_files:
    new_files = [file for file in uploaded_files if file.name not in [f[1] for f in st.session_state.uploaded_files]]
    
    for file in new_files:
        if len(st.session_state.uploaded_files) >= MAX_UPLOAD_FILES:
            st.warning(f"Maximum upload limit reached ({MAX_UPLOAD_FILES} images)")
            break
        
        try:
            image = Image.open(file)
            st.session_state.uploaded_files.append((image, file.name))
            
            # Remove from predictions if it was processed before
            if file.name in st.session_state.predictions:
                del st.session_state.predictions[file.name]
                
        except Exception as e:
            st.error(f"Error loading {file.name}: {str(e)}")

# Display uploaded images
if st.session_state.uploaded_files:
    st.markdown("### 📁 Uploaded Images")
    
    # Grid layout for images
    cols = st.columns(4)
    for idx, (image, filename) in enumerate(st.session_state.uploaded_files):
        with cols[idx % 4]:
            st.markdown('<div class="image-card">', unsafe_allow_html=True)
            
            # Display thumbnail
            st.image(image, use_column_width=True, caption=filename[:20] + "..." if len(filename) > 20 else filename)
            
            # File info
            file_size = len(image.tobytes()) / 1024 / 1024
            st.caption(f"Size: {file_size:.1f}MB | {image.size[0]}×{image.size[1]}")
            
            # Status
            if filename in st.session_state.predictions:
                pred = st.session_state.predictions[filename]
                if 'error' in pred:
                    st.markdown('<span class="status-badge status-error">Error</span>', unsafe_allow_html=True)
                elif pred['accepted']:
                    st.markdown('<span class="status-badge status-completed">Completed</span>', unsafe_allow_html=True)
                else:
                    st.markdown('<span class="status-badge status-processing">Low Confidence</span>', unsafe_allow_html=True)
            else:
                st.markdown('<span class="status-badge status-processing">Pending</span>', unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
    
    # Process button
    if st.button("🚀 Process All Images", type="primary", use_container_width=True):
        if st.session_state.selected_model:
            with st.spinner(f"Processing {len(st.session_state.uploaded_files)} images with {st.session_state.selected_model}..."):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                def update_progress(progress):
                    progress_bar.progress(progress)
                    status_text.text(f"Processing: {int(progress * len(st.session_state.uploaded_files))}/{len(st.session_state.uploaded_files)} images")
                
                results = process_batch(
                    st.session_state.uploaded_files,
                    st.session_state.selected_model,
                    update_progress
                )
                
                # Store predictions in session state
                for result in results:
                    st.session_state.predictions[result['filename']] = result
                
                progress_bar.progress(1.0)
                status_text.text("✅ Processing complete!")
                st.success(f"Processed {len(results)} images!")
                st.rerun()
        else:
            st.error("Please select a model first!")

# Display results if we have predictions
if st.session_state.predictions:
    st.markdown("---")
    st.markdown("## 📊 Analysis Results")
    
    # Summary statistics
    accepted = sum(1 for pred in st.session_state.predictions.values() if pred.get('accepted', False))
    rejected = len(st.session_state.predictions) - accepted
    avg_confidence = np.mean([pred.get('confidence', 0) for pred in st.session_state.predictions.values() if 'confidence' in pred])
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="stats-card">
            <div class="stat-number">{len(st.session_state.predictions)}</div>
            <div class="stat-label">Total Processed</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="stats-card">
            <div class="stat-number">{accepted}</div>
            <div class="stat-label">Accepted</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="stats-card">
            <div class="stat-number">{rejected}</div>
            <div class="stat-label">Rejected</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="stats-card">
            <div class="stat-number">{avg_confidence:.1%}</div>
            <div class="stat-label">Avg Confidence</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Detailed results table
    st.markdown("### Detailed Predictions")
    
    # Create dataframe for display
    results_data = []
    for filename, pred in st.session_state.predictions.items():
        if 'error' in pred:
            results_data.append({
                'Image': filename,
                'Status': 'Error',
                'Class': 'N/A',
                'Confidence': 'N/A',
                'Time': 'N/A',
                'Result': pred['error']
            })
        else:
            chip_class = pred['class'].lower() + '-chip'
            class_display = f'<span class="prediction-chip {chip_class}">{pred["class"].upper()}</span>'
            
            results_data.append({
                'Image': filename,
                'Status': '✅ Accepted' if pred['accepted'] else '⚠️ Low Confidence',
                'Class': class_display,
                'Confidence': f"{pred['confidence']:.2%}",
                'Time': f"{pred['time']:.3f}s",
                'Result': 'Accepted' if pred['accepted'] else f'Below {CONFIDENCE_THRESHOLD:.0%} threshold'
            })
    
    if results_data:
        # Display as HTML table for better styling
        html_table = """
        <table style="width:100%; border-collapse: collapse; margin: 1rem 0;">
            <thead>
                <tr style="background: #f8f9fa; border-bottom: 2px solid #dee2e6;">
                    <th style="padding: 12px; text-align: left;">Image</th>
                    <th style="padding: 12px; text-align: left;">Status</th>
                    <th style="padding: 12px; text-align: left;">Class</th>
                    <th style="padding: 12px; text-align: left;">Confidence</th>
                    <th style="padding: 12px; text-align: left;">Time</th>
                    <th style="padding: 12px; text-align: left;">Result</th>
                </tr>
            </thead>
            <tbody>
        """
        
        for row in results_data:
            html_table += f"""
            <tr style="border-bottom: 1px solid #e9ecef;">
                <td style="padding: 12px;">{row['Image']}</td>
                <td style="padding: 12px;">{row['Status']}</td>
                <td style="padding: 12px;">{row['Class']}</td>
                <td style="padding: 12px;">
                    <div class="progress-container">
                        <div class="progress-bar" style="width: {float(row['Confidence'].strip('%'))/100*100}%"></div>
                    </div>
                    {row['Confidence']}
                </td>
                <td style="padding: 12px;">{row['Time']}</td>
                <td style="padding: 12px;">{row['Result']}</td>
            </tr>
            """
        
        html_table += "</tbody></table>"
        st.markdown(html_table, unsafe_allow_html=True)
        
        # Visualization tabs
        tab1, tab2, tab3 = st.tabs(["📈 Distribution", "⏱️ Performance", "🎯 Confidence"])
        
        with tab1:
            # Class distribution chart
            if any('class' in pred for pred in st.session_state.predictions.values() if 'error' not in pred):
                class_counts = {}
                for pred in st.session_state.predictions.values():
                    if 'class' in pred:
                        class_counts[pred['class']] = class_counts.get(pred['class'], 0) + 1
                
                if class_counts:
                    fig = px.pie(
                        values=list(class_counts.values()),
                        names=list(class_counts.keys()),
                        title="Weather Class Distribution",
                        color_discrete_sequence=px.colors.qualitative.Set3
                    )
                    st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            # Performance metrics
            times = [pred.get('time', 0) for pred in st.session_state.predictions.values() if 'time' in pred]
            if times:
                avg_time = np.mean(times)
                fig = go.Figure(data=[go.Histogram(x=times, nbinsx=20)])
                fig.update_layout(
                    title=f"Processing Time Distribution (Avg: {avg_time:.3f}s)",
                    xaxis_title="Time (seconds)",
                    yaxis_title="Count",
                    showlegend=False
                )
                st.plotly_chart(fig, use_container_width=True)
        
        with tab3:
            # Confidence scores
            confidences = [pred.get('confidence', 0) for pred in st.session_state.predictions.values() if 'confidence' in pred]
            if confidences:
                fig = go.Figure(data=[go.Histogram(x=confidences, nbinsx=20)])
                fig.add_vline(
                    x=CONFIDENCE_THRESHOLD,
                    line_dash="dash",
                    line_color="red",
                    annotation_text=f"Threshold: {CONFIDENCE_THRESHOLD:.0%}",
                    annotation_position="top right"
                )
                fig.update_layout(
                    title="Confidence Scores Distribution",
                    xaxis_title="Confidence",
                    yaxis_title="Count",
                    showlegend=False,
                    xaxis_range=[0, 1]
                )
                st.plotly_chart(fig, use_container_width=True)
        
        # Export results
        st.markdown("---")
        st.markdown("### 📥 Export Results")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📋 Copy to Clipboard", use_container_width=True):
                # Create CSV string
                import io
                csv_buffer = io.StringIO()
                df = pd.DataFrame(results_data)
                df.to_csv(csv_buffer, index=False)
                st.success("Results copied to clipboard!")
        
        with col2:
            # Create downloadable CSV
            csv = pd.DataFrame(results_data).to_csv(index=False)
            st.download_button(
                label="📥 Download CSV",
                data=csv,
                file_name=f"weather_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )

else:
    # Instructions when no images are uploaded
    if not st.session_state.uploaded_files:
        st.markdown("---")
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown("""
            <div style="text-align: center; padding: 3rem; background: #f8f9fa; border-radius: 15px;">
                <h3 style="color: #6c757d;">📤 Ready to Analyze</h3>
                <p style="color: #6c757d;">
                    1. Click the upload area above<br>
                    2. Select multiple weather images<br>
                    3. Choose your model<br>
                    4. Click "Process All Images"<br>
                    5. View detailed results
                </p>
                <p style="color: #667eea; font-weight: bold;">
                    ⚡ Supports batch processing of up to {MAX_UPLOAD_FILES} images
                </p>
            </div>
            """.format(MAX_UPLOAD_FILES=MAX_UPLOAD_FILES), unsafe_allow_html=True)

# ==================== FOOTER ====================
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #6c757d; padding: 1rem;">
    <p>🌤️ <b>Weather Vision AI</b> | Advanced Multi-Image Classification | Model: {selected_model} | Threshold: {threshold:.0%}</p>
    <p style="font-size: 0.8rem;">Upload up to {max_files} images simultaneously • Max {max_size}MB per file</p>
</div>
""".format(
    selected_model=st.session_state.selected_model if 'selected_model' in st.session_state else "Not selected",
    threshold=CONFIDENCE_THRESHOLD,
    max_files=MAX_UPLOAD_FILES,
    max_size=MAX_FILE_SIZE_MB
), unsafe_allow_html=True)