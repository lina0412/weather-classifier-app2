# streamlit_app.py - MULTI-MODEL COMPARISON VERSION
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
    
    .model-comparison-card {
        background: white;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
        border-left: 5px solid #667eea;
    }
    
    .prediction-row {
        display: flex;
        align-items: center;
        padding: 0.8rem;
        border-bottom: 1px solid #f1f3f4;
        transition: background-color 0.2s;
    }
    
    .prediction-row:hover {
        background-color: #f8f9fa;
    }
    
    .model-name {
        font-weight: 600;
        width: 200px;
        color: #2d3748;
    }
    
    .prediction-chip {
        display: inline-block;
        padding: 0.4rem 1rem;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.9rem;
        margin: 0.2rem;
        min-width: 100px;
        text-align: center;
    }
    
    .hail-chip { background: #c8e6c9; color: #2e7d32; }
    .lightning-chip { background: #ffecb3; color: #ff8f00; }
    .rain-chip { background: #bbdefb; color: #1565c0; }
    .sandstorm-chip { background: #d7ccc8; color: #5d4037; }
    .snow-chip { background: #e3f2fd; color: #0277bd; }
    
    .confidence-bar {
        flex-grow: 1;
        margin: 0 1rem;
    }
    
    .progress-container {
        background: #e9ecef;
        border-radius: 10px;
        height: 8px;
        margin: 0.3rem 0;
        overflow: hidden;
    }
    
    .progress-bar {
        height: 100%;
        background: linear-gradient(90deg, #667eea, #764ba2);
        border-radius: 10px;
        transition: width 0.3s ease;
    }
    
    .confidence-value {
        width: 80px;
        text-align: right;
        font-weight: 600;
        color: #2d3748;
    }
    
    .time-value {
        width: 80px;
        text-align: right;
        color: #6c757d;
        font-size: 0.9rem;
    }
    
    .consensus-badge {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: 600;
        margin: 0.5rem 0;
        display: inline-block;
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
</style>
""", unsafe_allow_html=True)

# ==================== CONFIGURATION ====================
CONFIDENCE_THRESHOLD = 0.7
MAX_IMAGE_SIZE = 5000
MAX_UPLOAD_FILES = 20
MAX_FILE_SIZE_MB = 10

# ==================== SESSION STATE ====================
if 'uploaded_files' not in st.session_state:
    st.session_state.uploaded_files = []
if 'predictions' not in st.session_state:
    st.session_state.predictions = {}
if 'selected_models' not in st.session_state:
    st.session_state.selected_models = []
if 'comparison_data' not in st.session_state:
    st.session_state.comparison_data = {}

# ==================== LOAD DATA ====================
@st.cache_data
def load_class_names():
    with open('class_names.json', 'r') as f:
        return json.load(f)

@st.cache_data  
def load_model_accuracies():
    return {
        "Sparse Fine-Tuning": 0.9315,
        "Fine-Tuning": 0.9259,
        "Stochastic Fine-Tuning": 0.7593,
        "Knowledge Distillation": 0.7056,
        "SVM": 0.7963,
        "Random Forest": 0.7778,
        "MLP": 0.7222,
    }

@st.cache_resource
def create_feature_extractor():
    base = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    return Model(inputs=base.input, outputs=GlobalAveragePooling2D()(base.output))

@st.cache_resource
def load_all_models():
    """Load all available models"""
    models = {}
    
    # Model file mapping
    model_files = {
        "Sparse Fine-Tuning": "strategy4_sparse.keras",
        "Fine-Tuning": "strategy2_fine_tuned.keras",
        "Stochastic Fine-Tuning": "strategy3_stochastic.keras",
        "Knowledge Distillation": "weather_classifier_fixed.keras",
        "SVM": "strategy1_svm_rbf.pkl",
        "Random Forest": "strategy1_random_forest.pkl",
        "MLP": "strategy1_mlp.pkl"
    }
    
    accuracies = load_model_accuracies()
    
    for name, file in model_files.items():
        if os.path.exists(file):
            try:
                if file.endswith('.keras'):
                    models[name] = {
                        'type': 'keras',
                        'model': tf.keras.models.load_model(file),
                        'accuracy': accuracies.get(name, 0),
                        'color': '#667eea',
                        'file': file
                    }
                elif file.endswith('.pkl'):
                    models[name] = {
                        'type': 'ml',
                        'model': joblib.load(file),
                        'accuracy': accuracies.get(name, 0),
                        'color': '#10b981',
                        'file': file
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

def predict_with_model(model_info, image_array, feature_extractor=None):
    """Make prediction with a single model"""
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

# ==================== MULTI-MODEL PROCESSING ====================
def process_image_with_models(image, filename, model_names):
    """Process a single image with multiple models"""
    models = load_all_models()
    class_names = load_class_names()
    
    # Check which models need feature extractor
    need_feature_extractor = any(models.get(name, {}).get('type') == 'ml' for name in model_names)
    feature_extractor = create_feature_extractor() if need_feature_extractor else None
    
    results = {}
    
    for model_name in model_names:
        if model_name in models:
            model_info = models[model_name]
            
            try:
                # Preprocess image
                img_array = preprocess_image(image)
                
                # Make prediction
                predictions, pred_time, max_conf, pred_class_idx = predict_with_model(
                    model_info, img_array, feature_extractor
                )
                
                predicted_class = class_names[pred_class_idx]
                is_accepted = max_conf >= CONFIDENCE_THRESHOLD
                
                results[model_name] = {
                    'class': predicted_class,
                    'confidence': float(max_conf),
                    'time': float(pred_time),
                    'accepted': is_accepted,
                    'predictions': predictions.tolist(),
                    'accuracy': model_info['accuracy'],
                    'type': model_info['type']
                }
                
            except Exception as e:
                results[model_name] = {
                    'error': str(e),
                    'class': 'Error',
                    'confidence': 0,
                    'time': 0,
                    'accepted': False
                }
    
    return results

# ==================== SIDEBAR ====================
with st.sidebar:
    st.markdown("""
    <div style="padding: 1rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                border-radius: 12px; color: white; margin-bottom: 1.5rem;">
        <h3 style="margin: 0; color: white;">🤖 Model Selection</h3>
        <p style="margin: 0.5rem 0 0 0; opacity: 0.9;">Select 1-7 models to compare</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load available models
    models = load_all_models()
    model_names = list(models.keys())
    
    if model_names:
        # Multi-select for models
        selected_models = st.multiselect(
            "Choose models to compare:",
            model_names,
            default=model_names[:3] if len(model_names) >= 3 else model_names,
            help="Select multiple models to compare their predictions"
        )
        
        st.session_state.selected_models = selected_models
        
        # Display selected models with accuracy
        if selected_models:
            st.markdown("**Selected Models:**")
            for model_name in selected_models:
                accuracy = models[model_name]['accuracy']
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.write(f"• {model_name}")
                with col2:
                    st.write(f"{accuracy:.1%}")
    
    st.markdown("---")
    
    # Confidence threshold
    st.markdown("### ⚙️ Confidence Settings")
    confidence_threshold = st.slider(
        "Minimum confidence required:",
        min_value=0.0,
        max_value=1.0,
        value=CONFIDENCE_THRESHOLD,
        step=0.05,
        format="%.2f"
    )
    CONFIDENCE_THRESHOLD = confidence_threshold
    
    # Upload stats
    st.markdown("---")
    st.markdown("### 📊 Upload Stats")
    
    col1, col2 = st.columns(2)
    with col1:
        total_files = len(st.session_state.uploaded_files)
        st.metric("Total Images", total_files)
    
    with col2:
        processed = len(st.session_state.comparison_data)
        st.metric("Processed", processed)
    
    # Clear button
    if st.button("🗑️ Clear All", use_container_width=True, type="secondary"):
        st.session_state.uploaded_files = []
        st.session_state.comparison_data = {}
        st.rerun()

# ==================== MAIN INTERFACE ====================
# Header
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.markdown("""
    <div class="main-header">
        <h1 style="text-align: center; margin: 0;">🌤️ Weather Vision AI</h1>
        <p style="text-align: center; opacity: 0.9; margin: 0.5rem 0 0 0;">
            Multi-Model Comparison & Analysis
        </p>
    </div>
    """, unsafe_allow_html=True)

# Upload Section
st.markdown("### 📤 Upload Images for Comparison")
st.markdown(f"""
<div class="file-upload-box" onclick="document.getElementById('file-upload').click()">
    <div style="font-size: 4rem; color: #667eea; margin-bottom: 1rem;">📁</div>
    <h3>Drag & Drop or Click to Upload</h3>
    <p>Upload up to {MAX_UPLOAD_FILES} weather images at once</p>
    <p style="color: #6c757d; font-size: 0.9rem;">
        Compare predictions across {len(model_names)} different models
    </p>
</div>
""", unsafe_allow_html=True)

# File uploader
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
        except Exception as e:
            st.error(f"Error loading {file.name}: {str(e)}")

# Display uploaded images
if st.session_state.uploaded_files:
    st.markdown("### 📁 Uploaded Images")
    
    # Grid layout
    cols = st.columns(4)
    for idx, (image, filename) in enumerate(st.session_state.uploaded_files):
        with cols[idx % 4]:
            with st.container():
                st.markdown('<div class="image-card">', unsafe_allow_html=True)
                st.image(image, use_column_width=True, 
                        caption=filename[:20] + "..." if len(filename) > 20 else filename)
                
                # Check if processed
                if filename in st.session_state.comparison_data:
                    num_models = len(st.session_state.comparison_data[filename])
                    st.caption(f"✅ Processed with {num_models} models")
                else:
                    st.caption("⏳ Ready for processing")
                
                st.markdown('</div>', unsafe_allow_html=True)
    
    # Process button
    if st.session_state.selected_models:
        if st.button(f"🚀 Process All with {len(st.session_state.selected_models)} Models", 
                    type="primary", use_container_width=True):
            
            with st.spinner(f"Processing {len(st.session_state.uploaded_files)} images with {len(st.session_state.selected_models)} models..."):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                for idx, (image, filename) in enumerate(st.session_state.uploaded_files):
                    status_text.text(f"Processing {filename}...")
                    
                    # Process image with all selected models
                    results = process_image_with_models(
                        image, 
                        filename, 
                        st.session_state.selected_models
                    )
                    
                    st.session_state.comparison_data[filename] = results
                    progress_bar.progress((idx + 1) / len(st.session_state.uploaded_files))
                
                progress_bar.progress(1.0)
                status_text.text("✅ Processing complete!")
                st.success(f"Processed {len(st.session_state.uploaded_files)} images!")
                st.rerun()
    else:
        st.warning("⚠️ Please select at least one model from the sidebar to proceed.")

# Display comparison results
if st.session_state.comparison_data and st.session_state.selected_models:
    st.markdown("---")
    st.markdown("## 📊 Multi-Model Comparison Results")
    
    # Summary statistics
    total_images = len(st.session_state.comparison_data)
    total_predictions = total_images * len(st.session_state.selected_models)
    
    # Calculate consensus statistics
    consensus_count = 0
    all_confidences = []
    all_times = []
    
    for filename, model_results in st.session_state.comparison_data.items():
        # Check if all models agree
        if len(model_results) > 1:
            predictions = [results.get('class') for results in model_results.values() 
                          if 'error' not in results]
            if len(set(predictions)) == 1 and len(predictions) == len(st.session_state.selected_models):
                consensus_count += 1
        
        # Collect confidence and time data
        for model_name, results in model_results.items():
            if 'error' not in results:
                all_confidences.append(results.get('confidence', 0))
                all_times.append(results.get('time', 0))
    
    avg_confidence = np.mean(all_confidences) if all_confidences else 0
    avg_time = np.mean(all_times) if all_times else 0
    
    # Display summary cards
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="stats-card">
            <div class="stat-number">{total_images}</div>
            <div class="stat-label">Images</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="stats-card">
            <div class="stat-number">{len(st.session_state.selected_models)}</div>
            <div class="stat-label">Models</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="stats-card">
            <div class="stat-number">{consensus_count}/{total_images}</div>
            <div class="stat-label">Consensus</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="stats-card">
            <div class="stat-number">{avg_confidence:.1%}</div>
            <div class="stat-label">Avg Confidence</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Image selector for detailed view
    st.markdown("### 🔍 Detailed Analysis")
    
    selected_image = st.selectbox(
        "Select an image to view detailed model predictions:",
        list(st.session_state.comparison_data.keys())
    )
    
    if selected_image:
        model_results = st.session_state.comparison_data[selected_image]
        
        # Display the selected image
        col_img, col_stats = st.columns([1, 1])
        
        with col_img:
            # Find the image in uploaded files
            for image, filename in st.session_state.uploaded_files:
                if filename == selected_image:
                    st.image(image, use_column_width=True, caption=selected_image)
                    break
        
        with col_stats:
            # Calculate consensus
            predictions = []
            for model_name, results in model_results.items():
                if 'error' not in results:
                    predictions.append(results.get('class'))
            
            if predictions:
                unique_predictions = set(predictions)
                if len(unique_predictions) == 1:
                    st.markdown(f'<div class="consensus-badge">✅ Full Consensus: {list(unique_predictions)[0].upper()}</div>', 
                               unsafe_allow_html=True)
                else:
                    # Find most common prediction
                    from collections import Counter
                    pred_counter = Counter(predictions)
                    most_common = pred_counter.most_common(1)[0]
                    st.markdown(f'<div class="consensus-badge">⚠️ Partial Consensus: {most_common[0].upper()} ({most_common[1]}/{len(predictions)} models)</div>', 
                               unsafe_allow_html=True)
            
            # Display image stats
            st.metric("Models Applied", len(model_results))
            
            if all_confidences:
                st.metric("Average Confidence", f"{np.mean([r.get('confidence', 0) for r in model_results.values() if 'error' not in r]):.1%}")
        
        # Display model-by-model predictions
        st.markdown("#### Model Predictions")
        
        for model_name, results in model_results.items():
            if 'error' in results:
                st.error(f"**{model_name}**: Error - {results['error']}")
            else:
                confidence = results.get('confidence', 0)
                predicted_class = results.get('class', 'Unknown')
                prediction_time = results.get('time', 0)
                
                # Determine chip class
                chip_class = predicted_class.lower().replace(' ', '-') + '-chip'
                
                html = f"""
                <div class="model-comparison-card">
                    <div class="prediction-row">
                        <div class="model-name">{model_name}</div>
                        <span class="prediction-chip {chip_class}">{predicted_class.upper()}</span>
                        <div class="confidence-bar">
                            <div class="progress-container">
                                <div class="progress-bar" style="width: {confidence*100}%"></div>
                            </div>
                        </div>
                        <div class="confidence-value">{confidence:.1%}</div>
                        <div class="time-value">{prediction_time:.3f}s</div>
                    </div>
                    <div style="display: flex; justify-content: space-between; margin-top: 0.5rem; font-size: 0.9rem;">
                        <span style="color: #6c757d;">Accuracy: {results.get('accuracy', 0):.1%}</span>
                        <span style="color: {'#10b981' if confidence >= CONFIDENCE_THRESHOLD else '#ef4444'}">
                            {'✅ Accepted' if confidence >= CONFIDENCE_THRESHOLD else '⚠️ Low Confidence'}
                        </span>
                    </div>
                </div>
                """
                st.markdown(html, unsafe_allow_html=True)
        
        # Model performance comparison chart
        if len(model_results) > 1:
            st.markdown("#### 📈 Model Performance Comparison")
            
            # Create comparison data
            chart_data = []
            for model_name, results in model_results.items():
                if 'error' not in results:
                    chart_data.append({
                        'Model': model_name,
                        'Confidence': results.get('confidence', 0),
                        'Time (s)': results.get('time', 0),
                        'Accuracy': results.get('accuracy', 0),
                        'Type': results.get('type', 'Unknown')
                    })
            
            if chart_data:
                df = pd.DataFrame(chart_data)
                
                tab1, tab2, tab3 = st.tabs(["Confidence", "Speed", "Accuracy"])
                
                with tab1:
                    fig = px.bar(df, x='Model', y='Confidence', 
                                color='Type',
                                title=f"Confidence Scores by Model for {selected_image}",
                                color_discrete_map={'keras': '#667eea', 'ml': '#10b981'})
                    fig.add_hline(y=CONFIDENCE_THRESHOLD, line_dash="dash", 
                                 line_color="red", 
                                 annotation_text=f"Threshold: {CONFIDENCE_THRESHOLD:.0%}")
                    st.plotly_chart(fig, use_container_width=True)
                
                with tab2:
                    fig = px.bar(df, x='Model', y='Time (s)', 
                                color='Type',
                                title="Prediction Time by Model",
                                color_discrete_map={'keras': '#667eea', 'ml': '#10b981'})
                    st.plotly_chart(fig, use_container_width=True)
                
                with tab3:
                    fig = px.bar(df, x='Model', y='Accuracy',
                                title="Model Training Accuracy",
                                color='Type',
                                color_discrete_map={'keras': '#667eea', 'ml': '#10b981'})
                    st.plotly_chart(fig, use_container_width=True)
    
    # Batch analysis across all images
    st.markdown("---")
    st.markdown("## 📋 Batch Analysis Summary")
    
    # Create summary table
    summary_data = []
    for filename, model_results in st.session_state.comparison_data.items():
        for model_name, results in model_results.items():
            if 'error' not in results:
                summary_data.append({
                    'Image': filename,
                    'Model': model_name,
                    'Prediction': results.get('class', 'Error'),
                    'Confidence': results.get('confidence', 0),
                    'Time (s)': results.get('time', 0),
                    'Accepted': '✅' if results.get('confidence', 0) >= CONFIDENCE_THRESHOLD else '⚠️'
                })
    
    if summary_data:
        df_summary = pd.DataFrame(summary_data)
        
        # Pivot table for better visualization
        pivot_df = df_summary.pivot_table(
            index='Image',
            columns='Model',
            values=['Prediction', 'Confidence'],
            aggfunc='first'
        )
        
        st.dataframe(
            df_summary.style
            .background_gradient(subset=['Confidence'], cmap='RdYlGn', vmin=0, vmax=1)
            .format({'Confidence': '{:.1%}'}),
            use_container_width=True,
            height=400
        )
        
        # Export options
        st.markdown("### 📥 Export Results")
        col1, col2 = st.columns(2)
        
        csv = df_summary.to_csv(index=False)
        with col1:
            st.download_button(
                label="📊 Download CSV",
                data=csv,
                file_name=f"model_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        with col2:
            # Create Excel with multiple sheets
            excel_buffer = BytesIO()
            with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
                df_summary.to_excel(writer, sheet_name='Summary', index=False)
                
                # Add per-image details
                for filename in st.session_state.comparison_data.keys():
                    if filename in st.session_state.comparison_data:
                        img_data = []
                        for model_name, results in st.session_state.comparison_data[filename].items():
                            if 'error' not in results:
                                img_data.append({
                                    'Model': model_name,
                                    'Prediction': results.get('class', 'Error'),
                                    'Confidence': results.get('confidence', 0),
                                    'Time (s)': results.get('time', 0),
                                    'Accepted': 'Yes' if results.get('confidence', 0) >= CONFIDENCE_THRESHOLD else 'No'
                                })
                        if img_data:
                            pd.DataFrame(img_data).to_excel(
                                writer, 
                                sheet_name=filename[:30],  # Limit sheet name length
                                index=False
                            )
            
            excel_data = excel_buffer.getvalue()
            st.download_button(
                label="📈 Download Excel",
                data=excel_data,
                file_name=f"detailed_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )

# ==================== FOOTER ====================
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #6c757d; padding: 1rem; font-size: 0.9rem;">
    <p>🌤️ <b>Weather Vision AI</b> | Multi-Model Comparison System | 
    Threshold: {threshold:.0%} | {num_models} Models Available</p>
    <p>Upload up to {max_files} images • Compare 1-{max_compare} models simultaneously</p>
</div>
""".format(
    threshold=CONFIDENCE_THRESHOLD,
    num_models=len(model_names),
    max_files=MAX_UPLOAD_FILES,
    max_compare=len(model_names)
), unsafe_allow_html=True)
