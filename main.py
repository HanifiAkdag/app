import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import io
import base64
from typing import Dict, Any, List, Optional
import cv2
import sys
import os
from datetime import datetime

# Add the src directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Import our utility modules with error handling
try:
    from src.preprocessing import correct_illumination, denoise_image
    from src.phase_analysis import PhaseAnalyzer, get_default_phase_analysis_parameters, create_visualization_plots as create_phase_plots
    from src.line_analysis import LineAnalyzer, get_default_parameters as get_default_line_parameters, create_visualization_plots as create_line_plots
    from src.artifact_removal import ArtifactRemover, get_default_parameters as get_artifact_default_parameters, create_visualization_plots as create_artifact_plots
except ImportError as e:
    st.error(f"Import error: {e}")
    st.error("Please ensure all required modules are in the src/ directory")
    st.stop()

# Configure Streamlit page
st.set_page_config(
    page_title="Functional Materials Image Analysis",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
def initialize_session_state():
    """Initialize session state variables."""
    if 'original_image' not in st.session_state:
        st.session_state.original_image = None
    if 'processing_history' not in st.session_state:
        st.session_state.processing_history = []
    if 'selected_image_index' not in st.session_state:
        st.session_state.selected_image_index = 0  # 0 = original
    if 'material_mask' not in st.session_state:
        st.session_state.material_mask = None
    if 'last_operation' not in st.session_state:
        st.session_state.last_operation = None
    if 'results_cache' not in st.session_state:
        st.session_state.results_cache = {}
    if 'image_filename' not in st.session_state:
        st.session_state.image_filename = None
    if 'output_dir' not in st.session_state:
        st.session_state.output_dir = None

def create_output_directory(image_filename: str) -> str:
    """Create timestamped output directory for results - FIXED: Only create once."""
    # Check if directory already exists and is valid
    if (st.session_state.get('output_dir') and 
        os.path.exists(st.session_state.output_dir) and 
        st.session_state.get('image_filename') == image_filename):
        return st.session_state.output_dir
    
    # Create new directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = os.path.splitext(image_filename)[0]
    output_dir = os.path.join("outputs", f"{timestamp}_{base_name}")
    os.makedirs(output_dir, exist_ok=True)
    return output_dir

def save_image(image: np.ndarray, filename: str, output_dir: str) -> str:
    """Save image to output directory."""
    if image.dtype == np.float32 or image.dtype == np.float64:
        # Convert float [0,1] to uint8 [0,255]
        image_save = (np.clip(image, 0, 1) * 255).astype(np.uint8)
    else:
        image_save = image
    
    filepath = os.path.join(output_dir, filename)
    cv2.imwrite(filepath, image_save)
    return filepath

def get_current_image():
    """Get the currently selected image from processing history."""
    if st.session_state.original_image is None:
        return None
    
    if st.session_state.selected_image_index == 0:
        return st.session_state.original_image
    
    if st.session_state.selected_image_index <= len(st.session_state.processing_history):
        idx = st.session_state.selected_image_index - 1
        return st.session_state.processing_history[idx]['result']
    
    return st.session_state.original_image

def get_image_options():
    """Get list of available images for selection."""
    options = ["Original Image"]
    
    for i, step in enumerate(st.session_state.processing_history):
        operation_name = step['operation'].replace('_', ' ').title()
        options.append(f"Step {i+1}: {operation_name}")
    
    return options

def load_and_normalize_image(uploaded_file):
    """
    Properly load and normalize microscopy images.
    
    Args:
        uploaded_file: Streamlit uploaded file object
        
    Returns:
        Normalized numpy array [0,1] or None if failed
    """
    try:
        # Read the file
        file_bytes = uploaded_file.read()
        
        # Try to open with PIL first
        try:
            pil_image = Image.open(io.BytesIO(file_bytes))
            
            # Get image info
            st.info(f"📊 Image mode: {pil_image.mode}, Size: {pil_image.size}")
            
            # Convert to grayscale if needed
            if pil_image.mode == 'RGBA':
                # Convert RGBA to RGB first, then to grayscale
                pil_image = pil_image.convert('RGB').convert('L')
            elif pil_image.mode == 'RGB':
                pil_image = pil_image.convert('L')
            elif pil_image.mode == 'P':
                # Palette mode - convert to RGB first
                pil_image = pil_image.convert('RGB').convert('L')
            # If already 'L' (grayscale), keep as is
            
            # Convert to numpy array
            image_array = np.array(pil_image)
            
        except Exception as pil_error:
            st.warning(f"PIL failed: {pil_error}, trying OpenCV...")
            
            # Try with OpenCV as fallback
            nparr = np.frombuffer(file_bytes, np.uint8)
            image_array = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
            
            if image_array is None:
                raise ValueError("Could not decode image with OpenCV")
        
        # Check if we got a valid image
        if image_array is None or image_array.size == 0:
            raise ValueError("Invalid image data")
        
        # Get original data type and range info
        original_dtype = image_array.dtype
        original_min = np.min(image_array)
        original_max = np.max(image_array)
        original_range = original_max - original_min
        
        st.info(f"🔍 Original: dtype={original_dtype}, min={original_min}, max={original_max}, range={original_range}")
        
        # Normalize based on data type and range
        if original_dtype == np.uint16:
            # 16-bit image
            if original_max <= 255:
                # Actually 8-bit data stored as 16-bit
                image_normalized = image_array.astype(np.float32) / 255.0
            else:
                # True 16-bit data
                image_normalized = image_array.astype(np.float32) / 65535.0
                
        elif original_dtype == np.uint8:
            # 8-bit image
            image_normalized = image_array.astype(np.float32) / 255.0
            
        else:
            # Float or other types
            image_float = image_array.astype(np.float32)
            if original_max <= 1.0:
                # Already normalized
                image_normalized = image_float
            else:
                # Scale to [0,1]
                if original_range > 0:
                    image_normalized = (image_float - original_min) / original_range
                else:
                    image_normalized = image_float
        
        # Ensure values are in [0,1] range
        image_normalized = np.clip(image_normalized, 0.0, 1.0)
        
        # Check for unusual patterns that might indicate loading issues
        unique_values = np.unique(image_normalized)
        if len(unique_values) <= 2:
            st.warning(f"⚠️ Image appears to be binary (only {len(unique_values)} unique values). This might indicate a loading issue.")
            st.write(f"Unique values: {unique_values}")
        
        # Show normalization results
        final_min = np.min(image_normalized)
        final_max = np.max(image_normalized)
        final_mean = np.mean(image_normalized)
        final_std = np.std(image_normalized)
        
        st.success(f"✅ Normalized: min={final_min:.3f}, max={final_max:.3f}, mean={final_mean:.3f}, std={final_std:.3f}")
        
        return image_normalized
        
    except Exception as e:
        st.error(f"❌ Error loading image: {str(e)}")
        return None

def upload_image_section():
    """Handle image upload and display."""
    st.header("📤 Image Upload")
    
    uploaded_file = st.file_uploader(
        "Choose an image file",
        type=['png', 'jpg', 'jpeg', 'tiff', 'tif'],
        help="Upload a microscopy image for analysis"
    )
    
    if uploaded_file is not None:
        # Reset the file pointer
        uploaded_file.seek(0)
        
        # Show file info
        st.write(f"**Filename:** {uploaded_file.name}")
        st.write(f"**File size:** {uploaded_file.size:,} bytes")
        st.write(f"**File type:** {uploaded_file.type}")
        
        # Load and normalize the image
        image_normalized = load_and_normalize_image(uploaded_file)
        
        if image_normalized is not None:
            # Store in session state
            st.session_state.original_image = image_normalized
            st.session_state.processing_history = []
            st.session_state.selected_image_index = 0
            st.session_state.material_mask = None
            st.session_state.results_cache = {}
            st.session_state.image_filename = uploaded_file.name
            st.session_state.output_dir = create_output_directory(uploaded_file.name)
            
            # Save original image
            save_image(image_normalized, "original.png", st.session_state.output_dir)
            st.success(f"📁 Output directory created: {st.session_state.output_dir}")
            
            # Display image info
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Width", image_normalized.shape[1])
            with col2:
                st.metric("Height", image_normalized.shape[0])
            with col3:
                st.metric("Data Type", str(image_normalized.dtype))
            with col4:
                st.metric("Unique Values", len(np.unique(image_normalized)))
            
            # Display the image with proper scaling
            st.subheader("Uploaded Image")
            
            # Create two columns for different display modes
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Auto-scaled display**")
                st.image(image_normalized, caption="Auto-scaled", use_container_width=True, clamp=True)
            
            with col2:
                st.write("**Histogram**")
                fig, ax = plt.subplots(figsize=(6, 4))
                ax.hist(image_normalized.flatten(), bins=50, alpha=0.7, edgecolor='black')
                ax.set_xlabel('Pixel Intensity')
                ax.set_ylabel('Frequency')
                ax.set_title('Pixel Intensity Distribution')
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)
                plt.close(fig)
            
            # Advanced display options
            with st.expander("🔧 Advanced Display Options", expanded=False):
                # Manual scaling
                use_manual_scaling = st.checkbox("Use manual intensity scaling")
                if use_manual_scaling:
                    min_val = st.slider("Min display value", 0.0, 1.0, 0.0, step=0.01)
                    max_val = st.slider("Max display value", 0.0, 1.0, 1.0, step=0.01)
                    
                    if max_val > min_val:
                        # Apply manual scaling
                        scaled_image = np.clip((image_normalized - min_val) / (max_val - min_val), 0, 1)
                        st.image(scaled_image, caption=f"Manual scale [{min_val:.2f}, {max_val:.2f}]", use_container_width=True, clamp=True)
                
                # Colormap options
                colormap = st.selectbox("Colormap", ["gray", "viridis", "plasma", "inferno", "hot", "jet"], index=0)
                if colormap != "gray":
                    fig, ax = plt.subplots(figsize=(8, 6))
                    im = ax.imshow(image_normalized, cmap=colormap)
                    ax.set_title(f"Image with {colormap} colormap")
                    ax.axis('off')
                    plt.colorbar(im, ax=ax, fraction=0.046)
                    st.pyplot(fig)
                    plt.close(fig)
            
            return True
        else:
            return False
    
    return False

def show_current_image_status():
    """Show status of current image and processing history."""
    if st.session_state.original_image is not None:
        st.sidebar.markdown("---")
        st.sidebar.subheader("📸 Image Selection")
        
        # Image selection dropdown
        image_options = get_image_options()
        
        # Update selection - FIXED: Remove the key parameter and handle state change properly
        selected_option = st.sidebar.selectbox(
            "Choose image to work with:",
            options=range(len(image_options)),
            format_func=lambda x: image_options[x],
            index=st.session_state.selected_image_index
        )
        
        # Update selected index - FIXED: Direct assignment
        st.session_state.selected_image_index = selected_option
        
        # Show current selection info
        current_image = get_current_image()
        if current_image is not None:
            if st.session_state.selected_image_index == 0:
                st.sidebar.write("🔵 **Using Original Image**")
            else:
                step_info = st.session_state.processing_history[st.session_state.selected_image_index - 1]
                st.sidebar.write(f"🟢 **Using:** {step_info['operation'].replace('_', ' ').title()}")
            
            # Show image stats
            current_min = np.min(current_image)
            current_max = np.max(current_image)
            current_mean = np.mean(current_image)
            
            st.sidebar.write(f"**Stats:** min={current_min:.3f}, max={current_max:.3f}, mean={current_mean:.3f}")
        
        # Show processing history count
        num_steps = len(st.session_state.processing_history)
        if num_steps > 0:
            st.sidebar.write(f"📋 **Available Images:** {num_steps + 1} total")
        
        # Quick reset option
        if num_steps > 0:
            if st.sidebar.button("🗑️ Clear All Processing", help="Remove all processed images"):
                st.session_state.processing_history = []
                st.session_state.selected_image_index = 0
                st.session_state.material_mask = None
                st.session_state.results_cache = {}
                st.rerun()

def create_preprocessing_ui() -> Dict[str, Any]:
    """Create UI for preprocessing parameters."""
    st.subheader("🔧 Preprocessing Parameters")
    
    # Show which image will be processed
    current_image = get_current_image()
    if current_image is not None:
        if st.session_state.selected_image_index == 0:
            st.info("🔄 **Processing Original Image**")
        else:
            step_info = st.session_state.processing_history[st.session_state.selected_image_index - 1]
            st.info(f"🔄 **Processing:** {step_info['operation'].replace('_', ' ').title()}")
    
    # Illumination Correction
    with st.expander("💡 Illumination Correction", expanded=False):
        apply_illumination = st.checkbox("Apply Illumination Correction", value=False)
        illumination_method = st.selectbox(
            "Method",
            ["blur_subtract", "blur_divide"],
            index=1,
            disabled=not apply_illumination,
            help="blur_divide: good for multiplicative variations; blur_subtract: good for additive variations"
        )
        illumination_kernel = st.slider(
            "Kernel Size", 
            min_value=5, max_value=201, value=65, step=10,
            disabled=not apply_illumination,
            help="Size of Gaussian blur kernel for background estimation"
        )
    
    # Denoising
    with st.expander("🧹 Denoising", expanded=False):
        apply_denoising = st.checkbox("Apply Denoising", value=False)
        denoise_method = st.selectbox(
            "Method",
            ["median", "gaussian", "bilateral", "nlm"],
            index=1,
            disabled=not apply_denoising,
            help="median: good for salt-pepper noise; gaussian: general smoothing; bilateral: edge-preserving"
        )
        
        # Method-specific parameters
        denoise_params = {}
        if apply_denoising:
            if denoise_method == "median":
                denoise_params["median_k_size"] = st.slider("Kernel Size", 1, 15, 3, step=2)
            elif denoise_method == "gaussian":
                denoise_params["gaussian_sigma"] = st.slider("Sigma", 0.1, 5.0, 1.0, step=0.1)
            elif denoise_method == "bilateral":
                denoise_params["bilateral_d"] = st.slider("Diameter", 1, 15, 5)
                denoise_params["bilateral_sigma_color"] = st.slider("Sigma Color", 10, 150, 75)
                denoise_params["bilateral_sigma_space"] = st.slider("Sigma Space", 10, 150, 75)
            elif denoise_method == "nlm":
                denoise_params["nlm_h"] = st.slider("Filter Strength", 0.1, 1.0, 0.5, step=0.1)
                denoise_params["nlm_template_window_size"] = st.slider("Template Window Size", 3, 21, 7, step=2)
                denoise_params["nlm_search_window_size"] = st.slider("Search Window Size", 3, 21, 7, step=2)
    
    return {
        'apply_illumination_correction': apply_illumination,
        'illumination_method': illumination_method if apply_illumination else None,
        'illumination_kernel_size': illumination_kernel,
        'apply_denoising': apply_denoising,
        'denoise_method': denoise_method if apply_denoising else None,
        'denoise_params': denoise_params if apply_denoising else {}
    }

def create_phase_analysis_ui() -> Dict[str, Dict[str, Any]]:
    """Create UI for phase analysis parameters."""
    st.subheader("🧪 Phase Analysis Parameters")
    
    # Show which image will be analyzed
    current_image = get_current_image()
    if current_image is not None:
        if st.session_state.selected_image_index == 0:
            st.info("🔬 **Analyzing Original Image**")
        else:
            step_info = st.session_state.processing_history[st.session_state.selected_image_index - 1]
            st.info(f"🔬 **Analyzing:** {step_info['operation'].replace('_', ' ').title()}")
    
    # Preprocessing (note: these will be applied ON TOP of any existing preprocessing)
    with st.expander("🔧 Additional Preprocessing", expanded=False):
        st.warning("⚠️ These settings will be applied ON TOP of any existing preprocessing")
        
        initial_median_size = st.slider("Initial Median Size", 0, 10, 1)
        
        apply_illum = st.checkbox("Apply Illumination Correction", value=False)
        illum_method = st.selectbox("Method", ["blur_subtract", "blur_divide"], disabled=not apply_illum)
        illum_kernel = st.slider("Kernel Size", 5, 201, 65, step=10, disabled=not apply_illum)
        
        apply_denoise = st.checkbox("Apply Denoising", value=False)
        denoise_method = st.selectbox("Denoise Method", ["median", "gaussian", "bilateral"], disabled=not apply_denoise)
    
    # Masking
    with st.expander("🎭 Material Masking", expanded=True):
        masking_strategy = st.selectbox(
            "Strategy",
            ["fill_holes", "bright_phases"],
            index=0,
            help="fill_holes: fills internal holes in material; bright_phases: detects bright regions"
        )
        background_threshold = st.slider("Background Threshold", 0.0, 0.5, 0.08, step=0.01)
        cleanup_area = st.slider("Cleanup Area", 0, 2000, 500)
    
    # Artifact Removal
    with st.expander("🧹 Artifact Removal", expanded=False):
        dark_spot_area = st.slider("Dark Spot Fill Area", 0, 200, 50)
        dark_spot_method = st.selectbox("Dark Spot Threshold", ["otsu", "li", "triangle"], index=0)
        
        bright_spot_method = st.selectbox("Bright Spot Method", ["opening", "clipping"], index=0)
        bright_opening_size = st.slider("Opening Size", 0, 30, 15, disabled=bright_spot_method != "opening")
        bright_clip_offset = st.slider("Clip Offset", 0.0, 0.2, 0.05, step=0.01, disabled=bright_spot_method != "clipping")
    
    # Phase Detection
    with st.expander("🔬 Phase Detection", expanded=True):
        auto_detect = st.checkbox("Auto-detect Number of Phases", value=True)
        manual_phases = st.slider("Manual Phase Count", 1, 10, 3, disabled=auto_detect)
        
        if auto_detect:
            histogram_bins = st.slider("Histogram Bins", 50, 500, 256)
            min_distance = st.slider("Min Distance Between Peaks", 1, 20, 5)
            min_prominence = st.slider("Min Prominence Ratio", 0.01, 0.2, 0.05, step=0.01)
    
    # Segmentation
    with st.expander("✂️ Segmentation", expanded=True):
        seg_method = st.selectbox(
            "Method",
            ["auto", "kmeans", "otsu", "multiotsu", "percentile", "manual"],
            index=1,
            help="auto: automatic selection; kmeans: cluster-based; otsu/multiotsu: threshold-based"
        )
        
        if seg_method == "kmeans":
            kmeans_random_state = st.slider("Random State", 0, 100, 42)
        elif seg_method == "manual":
            st.write("Manual thresholds (comma-separated):")
            manual_thresholds_str = st.text_input("Thresholds", "0.3, 0.7")
            try:
                manual_thresholds = [float(x.strip()) for x in manual_thresholds_str.split(',')]
            except:
                manual_thresholds = None
                st.warning("Invalid threshold format")
    
    # Visualization
    with st.expander("🎨 Visualization", expanded=False):
        color_mode = st.selectbox("Color Mode", ["palette", "intensity"], index=0)
        palette_name = st.selectbox("Palette", ["viridis", "plasma", "inferno", "magma", "tab10"], index=0)
    
    # Compile parameters
    params = {
        'preprocessing': {
            'apply_illumination_correction': apply_illum,
            'illumination_method': illum_method if apply_illum else None,
            'illumination_kernel_size': illum_kernel,
            'apply_denoising': apply_denoise,
            'denoise_method': denoise_method if apply_denoise else None,
            'denoise_params': {},
            'initial_median_size': initial_median_size
        },
        'masking': {
            'strategy': masking_strategy,
            'background_threshold': background_threshold,
            'cleanup_area': cleanup_area
        },
        'artifact_removal': {
            'dark_spot_fill_area': dark_spot_area,
            'dark_spot_threshold_method': dark_spot_method,
            'bright_spot_method': bright_spot_method,
            'bright_spot_opening_size': bright_opening_size,
            'bright_spot_clip_offset': bright_clip_offset
        },
        'phase_detection': {
            'auto_detect_phases': auto_detect,
            'histogram_bins': histogram_bins if auto_detect else 256,
            'min_distance_bins': min_distance if auto_detect else 5,
            'min_prominence_ratio': min_prominence if auto_detect else 0.05,
            'default_phases': manual_phases
        },
        'segmentation': {
            'num_phases': manual_phases,
            'method': seg_method,
            'manual_thresholds': manual_thresholds if seg_method == "manual" else None,
            'kmeans_random_state': kmeans_random_state if seg_method == "kmeans" else 42
        },
        'visualization': {
            'color_mode': color_mode,
            'palette_name': palette_name
        }
    }
    
    return params

def create_line_analysis_ui() -> Dict[str, Dict[str, Any]]:
    """Create UI for line analysis parameters."""
    st.subheader("📏 Line Analysis Parameters")
    
    # Show which image will be analyzed
    current_image = get_current_image()
    if current_image is not None:
        if st.session_state.selected_image_index == 0:
            st.info("📏 **Analyzing Original Image**")
        else:
            step_info = st.session_state.processing_history[st.session_state.selected_image_index - 1]
            st.info(f"📏 **Analyzing:** {step_info['operation'].replace('_', ' ').title()}")
    
    # Preprocessing (note: these will be applied ON TOP of any existing preprocessing)
    with st.expander("🔧 Additional Preprocessing", expanded=False):
        st.warning("⚠️ These settings will be applied ON TOP of any existing preprocessing")
        
        apply_illum = st.checkbox("Apply Illumination Correction", value=False)
        illum_method = st.selectbox("Method", ["blur_subtract", "blur_divide"], disabled=not apply_illum)
        illum_kernel = st.slider("Kernel Size", 5, 201, 65, step=10, disabled=not apply_illum)
        
        apply_denoise = st.checkbox("Apply Denoising", value=False)
        denoise_method = st.selectbox("Denoise Method", ["median", "gaussian", "bilateral"], disabled=not apply_denoise)
    
    # Frangi Filter
    with st.expander("🌊 Frangi Filter", expanded=True):
        sigma_min = st.slider("Sigma Min", 1, 10, 1, help="Minimum scale for line detection")
        sigma_max = st.slider("Sigma Max", 2, 20, 8, help="Maximum scale for line detection")
        sigma_step = st.slider("Sigma Step", 1, 5, 2, help="Step size between scales")
        black_ridges = st.checkbox("Detect Dark Lines", value=False)
    
    # Skeletonization
    with st.expander("🦴 Skeletonization", expanded=False):
        apply_skeleton = st.checkbox("Apply Skeletonization", value=True)
    
    # Boundary Exclusion
    with st.expander("🚧 Boundary Exclusion", expanded=False):
        boundary_erosion = st.slider("Boundary Erosion Size", 0, 50, 10)
    
    # Hough Transform
    with st.expander("📏 Hough Transform", expanded=True):
        hough_threshold = st.slider("Threshold", 1, 20, 5)
        hough_min_length = st.slider("Min Line Length", 5, 100, 20)
        hough_max_gap = st.slider("Max Line Gap", 1, 50, 10)
        hough_bins = st.slider("Histogram Bins", 30, 180, 60)
    
    # Sobel Analysis
    with st.expander("🔍 Sobel Analysis", expanded=False):
        analyze_sobel = st.checkbox("Analyze Sobel Edges", value=True)
        sobel_magnitude_threshold = st.slider("Magnitude Threshold", 0.001, 0.1, 0.01, step=0.001, disabled=not analyze_sobel)
        sobel_bins = st.slider("Sobel Histogram Bins", 30, 180, 90, disabled=not analyze_sobel)
    
    # Visualization
    with st.expander("🎨 Visualization", expanded=False):
        colormap = st.selectbox("Colormap", ["hsv", "viridis", "plasma", "rainbow", "jet"], index=0)
    
    # Material mask info
    if st.session_state.material_mask is None:
        st.info("💡 Material mask will be automatically created using Otsu thresholding")
    else:
        st.success("✅ Using existing material mask from previous analysis")
    
    # Compile parameters
    params = {
        'preprocessing': {
            'apply_illumination_correction': apply_illum,
            'illumination_method': illum_method if apply_illum else None,
            'illumination_kernel_size': illum_kernel,
            'apply_denoising': apply_denoise,
            'denoise_method': denoise_method if apply_denoise else None,
            'denoise_params': {}
        },
        'frangi': {
            'sigmas': range(sigma_min, sigma_max + 1, sigma_step),
            'black_ridges': black_ridges
        },
        'hough': {
            'threshold': hough_threshold,
            'min_length': hough_min_length,
            'max_gap': hough_max_gap
        },
        'analysis': {
            'apply_skeletonization': apply_skeleton,
            'boundary_erosion_size': boundary_erosion,
            'analyze_sobel': analyze_sobel,
            'hough_histogram_bins': hough_bins,
            'sobel_histogram_bins': sobel_bins,
            'sobel_magnitude_threshold': sobel_magnitude_threshold,
            'colormap': colormap
        }
    }
    
    return params

def create_artifact_removal_ui() -> Dict[str, Any]:
    """Create UI for artifact removal parameters."""
    st.subheader("🎯 Artifact Removal Parameters")
    
    # Show which image will be processed
    current_image = get_current_image()
    if current_image is not None:
        if st.session_state.selected_image_index == 0:
            st.info("🎯 **Processing Original Image**")
        else:
            step_info = st.session_state.processing_history[st.session_state.selected_image_index - 1]
            st.info(f"🎯 **Processing:** {step_info['operation'].replace('_', ' ').title()}")
    
    # Detection parameters
    with st.expander("🔍 Detection Parameters", expanded=True):
        detection_method = st.selectbox(
            "Detection Method",
            ["threshold", "adaptive", "blob"],
            index=0,
            help="Method for detecting artifacts in the image"
        )
        
        if detection_method == "threshold":
            threshold_value = st.slider("Threshold Value", 0.0, 1.0, 0.8, step=0.01)
            threshold_mode = st.selectbox("Threshold Mode", ["binary", "binary_inv"], index=0)
        elif detection_method == "adaptive":
            block_size = st.slider("Block Size", 3, 51, 11, step=2)
            c_value = st.slider("C Value", -20, 20, 2)
        elif detection_method == "blob":
            min_area = st.slider("Min Blob Area", 10, 1000, 100)
            circularity = st.slider("Min Circularity", 0.0, 1.0, 0.8, step=0.05)
    
    # Processing parameters
    with st.expander("🛠️ Processing Parameters", expanded=True):
        processing_method = st.selectbox(
            "Processing Method",
            ["remove", "inpaint", "filter"],
            index=1,
            help="Method for handling detected artifacts"
        )
        
        if processing_method == "inpaint":
            inpaint_method = st.selectbox("Inpainting Method", ["NS", "TELEA"], index=0)
            inpaint_radius = st.slider("Inpainting Radius", 1, 10, 3)
        elif processing_method == "filter":
            filter_size = st.slider("Filter Size", 3, 15, 5, step=2)
            filter_type = st.selectbox("Filter Type", ["median", "gaussian", "mean"], index=0)
    
    # Post-processing
    with st.expander("🔧 Post-processing", expanded=False):
        apply_smoothing = st.checkbox("Apply Smoothing", value=True)
        if apply_smoothing:
            smooth_method = st.selectbox("Smoothing Method", ["gaussian", "median", "bilateral"], index=0)
            smooth_kernel = st.slider("Kernel Size", 3, 15, 5, step=2)
    
    # Compile parameters
    params = get_artifact_default_parameters()
    
    # Update with UI values
    params.update({
        'detection': {
            'method': detection_method,
            'threshold_value': threshold_value if detection_method == "threshold" else 0.8,
            'threshold_mode': threshold_mode if detection_method == "threshold" else "binary",
            'block_size': block_size if detection_method == "adaptive" else 11,
            'c_value': c_value if detection_method == "adaptive" else 2,
            'min_area': min_area if detection_method == "blob" else 100,
            'circularity': circularity if detection_method == "blob" else 0.8,
        },
        'processing': {
            'method': processing_method,
            'inpaint_method': inpaint_method if processing_method == "inpaint" else "NS",
            'inpaint_radius': inpaint_radius if processing_method == "inpaint" else 3,
            'filter_size': filter_size if processing_method == "filter" else 5,
            'filter_type': filter_type if processing_method == "filter" else "median",
        },
        'post_processing': {
            'apply_smoothing': apply_smoothing,
            'smooth_method': smooth_method if apply_smoothing else "gaussian",
            'smooth_kernel': smooth_kernel if apply_smoothing else 5,
        }
    })
    
    return params

def apply_preprocessing(image: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
    """Apply preprocessing operations."""
    processed_image = image.copy()
    
    # Convert to uint8 for OpenCV operations
    if processed_image.dtype != np.uint8:
        processed_image = (processed_image * 255).astype(np.uint8)
    
    if params['apply_illumination_correction']:
        processed_image = correct_illumination(
            processed_image,
            params['illumination_method'],
            params['illumination_kernel_size']
        )
    
    if params['apply_denoising']:
        processed_image = denoise_image(
            processed_image,
            params['denoise_method'],
            **params['denoise_params']
        )
    
    # Convert back to float [0,1]
    return processed_image.astype(np.float32) / 255.0

def apply_phase_analysis(image: np.ndarray, params: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Apply phase analysis - uses current processed image."""
    analyzer = PhaseAnalyzer()
    results = analyzer.run_full_analysis(image, **params)
    return results

def apply_line_analysis(image: np.ndarray, material_mask: np.ndarray, params: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Apply line analysis - uses current processed image."""
    analyzer = LineAnalyzer()
    results = analyzer.run_full_analysis(image, material_mask, **params)
    return results

def apply_artifact_removal(image: np.ndarray, params: Dict[str, Any]) -> Dict[str, Any]:
    """Apply artifact removal using OOP interface."""
    analyzer = ArtifactRemover()
    results = analyzer.run_full_analysis(image, **params)
    return results

def create_auto_material_mask(image: np.ndarray) -> np.ndarray:
    """Create a simple material mask using Otsu thresholding."""
    if image.dtype != np.uint8:
        image_uint8 = (image * 255).astype(np.uint8)
    else:
        image_uint8 = image
    
    # Apply Otsu thresholding
    _, mask = cv2.threshold(image_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return mask.astype(bool)

def display_results(operation: str, results: Any, params: Dict = None):
    """Display results based on operation type."""
    
    if operation == "preprocessing":
        st.subheader("📊 Preprocessing Results")
        
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Original Image**")
            st.image(st.session_state.original_image, use_container_width=True, clamp=True)
        
        with col2:
            st.write("**Processed Image**")
            st.image(results, use_container_width=True, clamp=True)
        
        # Show histogram comparison
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        ax1.hist(st.session_state.original_image.flatten(), bins=50, alpha=0.7, color='blue', edgecolor='black')
        ax1.set_title('Original Image Histogram')
        ax1.set_xlabel('Pixel Intensity')
        ax1.set_ylabel('Frequency')
        ax1.grid(True, alpha=0.3)
        
        ax2.hist(results.flatten(), bins=50, alpha=0.7, color='red', edgecolor='black')
        ax2.set_title('Processed Image Histogram')
        ax2.set_xlabel('Pixel Intensity')
        ax2.set_ylabel('Frequency')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
    
    elif operation == "phase_analysis":
        st.subheader("🧪 Phase Analysis Results")
        
        if results['success']:
            # Display key metrics
            col1, col2, col3 = st.columns(3)
            
            analysis_results = results['analysis_results']
            phase_stats = results['phase_statistics']
            
            with col1:
                st.metric("Phases Detected", analysis_results['num_phases_detected'])
            with col2:
                st.metric("Material Pixels", analysis_results['total_material_pixels'])
            with col3:
                st.metric("Segmentation Method", analysis_results['segmentation_method'])
            
            # Display phase statistics
            if phase_stats:
                st.subheader("📊 Phase Statistics")
                stats_data = []
                for phase_name, stats in phase_stats.items():
                    stats_data.append({
                        "Phase": phase_name.replace('_', ' ').title(),
                        "Pixels": stats['pixels'],
                        "Area Fraction": f"{stats['fraction']:.4f}"
                    })
                st.table(stats_data)
            
            # Create and display visualizations
            try:
                analyzer = PhaseAnalyzer()
                analyzer.intermediate_images = results['intermediate_images']
                figures = create_phase_plots(analyzer, results)
                
                if figures:
                    st.subheader("🎨 Visualizations")
                    
                    for fig_name, fig in figures.items():
                        st.write(f"**{fig_name.replace('_', ' ').title()}**")
                        st.pyplot(fig)
                        plt.close(fig)  # Clean up
            except Exception as e:
                st.warning(f"Could not create visualizations: {e}")
        else:
            st.error(f"❌ Phase analysis failed: {results['error_message']}")
    
    elif operation == "line_analysis":
        st.subheader("📏 Line Analysis Results")
        
        if results['success']:
            # Display key metrics
            col1, col2, col3 = st.columns(3)
            
            analysis_results = results['analysis_results']
            
            with col1:
                st.metric("Lines Detected", analysis_results['num_lines_detected'])
            with col2:
                if analysis_results['dominant_hough_angle'] is not None:
                    st.metric("Dominant Hough Angle", f"{analysis_results['dominant_hough_angle']:.1f}°")
                else:
                    st.metric("Dominant Hough Angle", "N/A")
            with col3:
                if analysis_results['dominant_sobel_angle'] is not None:
                    st.metric("Dominant Sobel Angle", f"{analysis_results['dominant_sobel_angle']:.1f}°")
                else:
                    st.metric("Dominant Sobel Angle", "N/A")
            
            # Create and display visualizations
            try:
                analyzer = LineAnalyzer()
                analyzer.intermediate_images = results['intermediate_images']
                figures = create_line_plots(analyzer, results)
                
                if figures:
                    st.subheader("🎨 Visualizations")
                    
                    for fig_name, fig in figures.items():
                        st.write(f"**{fig_name.replace('_', ' ').title()}**")
                        st.pyplot(fig)
                        plt.close(fig)  # Clean up
            except Exception as e:
                st.warning(f"Could not create visualizations: {e}")
        else:
            st.error(f"❌ Line analysis failed: {results['error_message']}")
    
    elif operation == "artifact_removal":
        st.subheader("🎯 Artifact Removal Results")
        
        if results['success']:
            # Display key metrics
            col1, col2, col3 = st.columns(3)
            
            analysis_results = results['analysis_results']
            artifact_stats = results['artifact_statistics']
            
            with col1:
                st.metric("Artifacts Detected", "Yes" if analysis_results['artifacts_detected'] else "No")
            with col2:
                st.metric("Artifacts Processed", "Yes" if analysis_results['artifacts_processed'] else "No")
            with col3:
                st.metric("Inpaint Method", analysis_results['inpainting_method'].upper())
            
            # Show detailed statistics if artifacts were found
            if analysis_results['artifacts_detected']:
                st.subheader("📊 Artifact Statistics")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Initial Detection:**")
                    st.metric("Components", artifact_stats['initial_artifacts']['num_components'])
                    st.metric("Coverage", f"{artifact_stats['initial_artifacts']['coverage_percent']:.3f}%")
                    st.metric("Largest Component", f"{artifact_stats['initial_artifacts']['max_component_size']} px")
                
                with col2:
                    st.write("**Final Processing:**")
                    st.metric("Components", artifact_stats['final_artifacts']['num_components'])
                    st.metric("Coverage", f"{artifact_stats['final_artifacts']['coverage_percent']:.3f}%")
                    st.metric("Filtered Out", f"{artifact_stats['filtered_out']['components']} components")
            
            # Create and display visualizations
            try:
                analyzer = ArtifactRemover()
                analyzer.intermediate_images = results['intermediate_images']
                figures = create_artifact_plots(analyzer, results)
                
                if figures:
                    st.subheader("🎨 Visualizations")
                    
                    for fig_name, fig in figures.items():
                        st.write(f"**{fig_name.replace('_', ' ').title()}**")
                        st.pyplot(fig)
                        plt.close(fig)  # Clean up
            except Exception as e:
                st.warning(f"Could not create visualizations: {e}")
            
            # Display images in columns
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.write("**Input Image**")
                input_image = get_current_image()
                if input_image is not None:
                    st.image(input_image, use_container_width=True, clamp=True)
            
            with col2:
                st.write("**Processed Image**")
                if results.get('processed_image') is not None:
                    # Convert back to [0,1] for display
                    processed_display = results['processed_image'].astype(np.float32) / 255.0
                    st.image(processed_display, use_container_width=True, clamp=True)
            
            with col3:
                st.write("**Detection Mask**")
                if results.get('final_mask') is not None:
                    st.image(results['final_mask'].astype(np.uint8) * 255, use_container_width=True, clamp=True)
        else:
            st.error(f"❌ Artifact removal failed: {results['error_message']}")

def processing_history_section():
    """Display and manage processing history."""
    if st.session_state.processing_history:
        st.subheader("📋 Processing History")
        
        for i, step in enumerate(st.session_state.processing_history):
            with st.expander(f"Step {i+1}: {step['operation'].replace('_', ' ').title()}", expanded=False):
                st.write(f"**Operation:** {step['operation']}")
                if 'timestamp' in step:
                    st.write(f"**Timestamp:** {step['timestamp']}")
                if 'saved_path' in step:
                    st.write(f"**Saved to:** {step['saved_path']}")
                if 'params' in step:
                    st.write("**Parameters:**")
                    st.json(step['params'])
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 Remove Last Step", type="secondary"):
                if len(st.session_state.processing_history) > 0:
                    st.session_state.processing_history.pop()
                    # Update selected index if needed
                    if st.session_state.selected_image_index > len(st.session_state.processing_history):
                        st.session_state.selected_image_index = len(st.session_state.processing_history)
                    st.rerun()
        
        with col2:
            if st.button("🗑️ Clear All History", type="secondary"):
                st.session_state.processing_history = []
                st.session_state.selected_image_index = 0
                st.session_state.material_mask = None
                st.rerun()

def main():
    """Main Streamlit application."""
    # Initialize session state
    initialize_session_state()
    
    # Header
    st.title("🔬 Functional Materials Image Analysis")
    st.markdown("*Advanced microscopy image processing and analysis toolkit*")
    
    # Sidebar for operations
    st.sidebar.header("🛠️ Operations")
    
    # Show current image status in sidebar
    show_current_image_status()
    
    # Image upload
    image_uploaded = upload_image_section()
    
    if not image_uploaded or st.session_state.original_image is None:
        st.warning("⚠️ Please upload an image to begin analysis")
        return
    
    # Operation selection
    operation = st.sidebar.selectbox(
        "Choose Operation",
        [
            "preprocessing",
            "artifact_removal", 
            "phase_analysis",
            "line_analysis"
        ],
        format_func=lambda x: {
            "preprocessing": "🔧 Preprocessing",
            "artifact_removal": "🎯 Artifact Removal",
            "phase_analysis": "🧪 Phase Analysis", 
            "line_analysis": "📏 Line Analysis"
        }[x]
    )
    
    # Parameter configuration based on operation
    params = None
    
    if operation == "preprocessing":
        params = create_preprocessing_ui()
    elif operation == "phase_analysis":
        params = create_phase_analysis_ui()
    elif operation == "line_analysis":
        params = create_line_analysis_ui()
    elif operation == "artifact_removal":
        params = create_artifact_removal_ui()
    
    # Process button
    if st.sidebar.button(f"🚀 Run {operation.replace('_', ' ').title()}", type="primary"):
        with st.spinner(f"Processing {operation.replace('_', ' ')}..."):
            try:
                # Get current image to process
                current_image = get_current_image()
                if current_image is None:
                    st.error("No image available for processing")
                    return
                
                # Apply the selected operation
                if operation == "preprocessing":
                    result = apply_preprocessing(current_image, params)
                    
                elif operation == "phase_analysis":
                    result = apply_phase_analysis(current_image, params)
                    
                elif operation == "line_analysis":
                    # Create material mask if not available
                    if st.session_state.material_mask is None:
                        st.session_state.material_mask = create_auto_material_mask(current_image)
                        st.info("📝 Created automatic material mask using Otsu thresholding")
                    
                    result = apply_line_analysis(current_image, st.session_state.material_mask, params)
                
                elif operation == "artifact_removal":
                    result = apply_artifact_removal(current_image, params)
                
                # Store results
                st.session_state.last_operation = operation
                st.session_state.results_cache[operation] = result
                
                # For operations that modify the image, update processing history
                if operation in ["preprocessing", "artifact_removal"]:
                    if operation == "artifact_removal":
                        if result.get('success', False) and result.get('artifacts_processed', False):
                            # Convert processed image back to [0,1]
                            processed_image = result['processed_image'].astype(np.float32) / 255.0
                        else:
                            processed_image = current_image.copy()
                    else:
                        processed_image = result
                    
                    # Ensure output directory exists
                    if not st.session_state.output_dir:
                        st.session_state.output_dir = create_output_directory(st.session_state.image_filename)
                    
                    # Save processed image
                    timestamp = datetime.now().strftime("%H%M%S")
                    filename = f"{operation}_{timestamp}.png"
                    saved_path = save_image(processed_image, filename, st.session_state.output_dir)
                    
                    # Add to history with metadata
                    step_data = {
                        'operation': operation,
                        'params': params,
                        'result': processed_image.copy(),
                        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        'saved_path': saved_path
                    }
                    
                    st.session_state.processing_history.append(step_data)
                    
                    # Auto-select the newly processed image
                    st.session_state.selected_image_index = len(st.session_state.processing_history)
                    
                    st.success(f"💾 Image saved: {saved_path}")
                
                # Save analysis results
                elif operation in ["phase_analysis", "line_analysis"]:
                    # Ensure output directory exists
                    if not st.session_state.output_dir:
                        st.session_state.output_dir = create_output_directory(st.session_state.image_filename)
                    
                    timestamp = datetime.now().strftime("%H%M%S")
                    
                    # Save analysis results as JSON
                    import json
                    results_file = f"{operation}_results_{timestamp}.json"
                    results_path = os.path.join(st.session_state.output_dir, results_file)
                    
                    # Prepare results for JSON (remove non-serializable items)
                    results_to_save = {}
                    for key, value in result.items():
                        if isinstance(value, (str, int, float, bool, list, dict)) or value is None:
                            results_to_save[key] = value
                        elif hasattr(value, 'tolist'):  # numpy arrays
                            results_to_save[key] = value.tolist()
                        else:
                            results_to_save[key] = str(value)
                    
                    with open(results_path, 'w') as f:
                        json.dump(results_to_save, f, indent=2)
                    
                    st.success(f"📊 Results saved: {results_path}")
                
                # Update material mask for phase analysis
                if operation == "phase_analysis" and result.get('success', False):
                    st.session_state.material_mask = result.get('material_mask')
                
                st.success(f"✅ {operation.replace('_', ' ').title()} completed successfully!")
                
            except Exception as e:
                st.error(f"❌ Error during {operation}: {str(e)}")
                st.exception(e)
    
    # Display results
    if st.session_state.last_operation and st.session_state.last_operation in st.session_state.results_cache:
        display_results(
            st.session_state.last_operation, 
            st.session_state.results_cache[st.session_state.last_operation],
            params
        )
    
    # Processing history
    processing_history_section()
    
    # Current image display
    current_image = get_current_image()
    if current_image is not None:
        st.subheader("📸 Current Selected Image")
        
        # Show current image with diagnostics
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Show which image is selected
            if st.session_state.selected_image_index == 0:
                caption = "Original Image"
            else:
                step_info = st.session_state.processing_history[st.session_state.selected_image_index - 1]
                caption = f"Step {st.session_state.selected_image_index}: {step_info['operation'].replace('_', ' ').title()}"
            
            st.image(current_image, caption=caption, use_container_width=True, clamp=True)
        
        with col2:
            # Image diagnostics
            st.write("**Image Diagnostics:**")
            current_min = np.min(current_image)
            current_max = np.max(current_image)
            current_mean = np.mean(current_image)
            current_std = np.std(current_image)
            unique_count = len(np.unique(current_image))
            
            st.metric("Min Value", f"{current_min:.3f}")
            st.metric("Max Value", f"{current_max:.3f}")
            st.metric("Mean", f"{current_mean:.3f}")
            st.metric("Std Dev", f"{current_std:.3f}")
            st.metric("Unique Values", unique_count)
    
    # Show output directory info
    if st.session_state.output_dir:
        st.sidebar.markdown("---")
        st.sidebar.subheader("📁 Output Directory")
        st.sidebar.code(st.session_state.output_dir)
        
        # List saved files
        if os.path.exists(st.session_state.output_dir):
            files = os.listdir(st.session_state.output_dir)
            if files:
                st.sidebar.write(f"**Files saved:** {len(files)}")
                with st.sidebar.expander("View files"):
                    for file in sorted(files):
                        st.write(f"• {file}")

if __name__ == "__main__":
    main()