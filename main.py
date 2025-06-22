import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import io
import cv2
import sys
import os
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
import json

# Add the src directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Import our modules
try:
    from src.preprocessing import correct_illumination, denoise_image
    from src.phase_analysis import PhaseAnalyzer, create_visualization_plots as create_phase_plots
    from src.line_analysis import LineAnalyzer, create_visualization_plots as create_line_plots
    from src.artifact_removal import ArtifactRemover, create_visualization_plots as create_artifact_plots
    from src.utils import create_material_mask
except ImportError as e:
    st.error(f"Import error: {e}")
    st.stop()

# Configure Streamlit
st.set_page_config(
    page_title="Custom Pipeline Builder",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Initialize session state
def initialize_session_state():
    """Initialize session state variables."""
    if 'original_image' not in st.session_state:
        st.session_state.original_image = None
    if 'pipeline_steps' not in st.session_state:
        st.session_state.pipeline_steps = []
    if 'pipeline_results' not in st.session_state:
        st.session_state.pipeline_results = []
    if 'current_image' not in st.session_state:
        st.session_state.current_image = None
    if 'material_mask' not in st.session_state:
        st.session_state.material_mask = None
    if 'output_dir' not in st.session_state:
        st.session_state.output_dir = None

def load_and_normalize_image(uploaded_file):
    """Load and normalize an image to [0,1] float32."""
    try:
        file_bytes = uploaded_file.read()
        
        # Try PIL first
        try:
            pil_image = Image.open(io.BytesIO(file_bytes))
            if pil_image.mode in ['RGBA', 'RGB', 'P']:
                pil_image = pil_image.convert('L')
            image_array = np.array(pil_image)
        except Exception:
            # Fallback to OpenCV
            nparr = np.frombuffer(file_bytes, np.uint8)
            image_array = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
            
        if image_array is None:
            return None
            
        # Normalize to [0,1]
        if image_array.dtype == np.uint8:
            return image_array.astype(np.float32) / 255.0
        elif image_array.dtype == np.uint16:
            return image_array.astype(np.float32) / 65535.0
        else:
            # Already float, normalize to [0,1]
            img_min, img_max = image_array.min(), image_array.max()
            if img_max > img_min:
                return (image_array - img_min) / (img_max - img_min)
            else:
                return image_array.astype(np.float32)
                
    except Exception as e:
        st.error(f"Error loading image: {e}")
        return None

def create_output_directory(image_filename: str) -> str:
    """Create output directory for results."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = os.path.splitext(image_filename)[0] if image_filename else "pipeline"
    output_dir = os.path.join("outputs", f"{timestamp}_{base_name}")
    os.makedirs(output_dir, exist_ok=True)
    return output_dir

def save_image(image: np.ndarray, filename: str, output_dir: str) -> str:
    """Save image to output directory."""
    if image.dtype in [np.float32, np.float64]:
        image_save = (np.clip(image, 0, 1) * 255).astype(np.uint8)
    else:
        image_save = image
    
    filepath = os.path.join(output_dir, filename)
    cv2.imwrite(filepath, image_save)
    return filepath

# Operation parameter UIs
def create_preprocessing_params_ui(step_id: str) -> Dict[str, Any]:
    """Create UI for preprocessing parameters."""
    with st.expander(f"🔧 Preprocessing Parameters (Step {step_id})", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Illumination Correction")
            st.info("💡 Corrects uneven lighting across the image. Use when background brightness varies significantly.")
            
            apply_illum = st.checkbox("Apply Illumination Correction", key=f"illum_{step_id}", value=True)
            
            illum_method = st.selectbox("Method", ["blur_subtract", "blur_divide"], 
                                      key=f"illum_method_{step_id}", disabled=not apply_illum,
                                      help="blur_subtract: Subtracts blurred background (better for mild variations)\nblur_divide: Divides by blurred background (better for strong variations)")
            
            illum_kernel = st.slider("Kernel Size", 5, 201, 65, step=10, 
                                   key=f"illum_kernel_{step_id}", disabled=not apply_illum,
                                   help="Size of the blur kernel. Larger values = smoother background estimation. Should be larger than the largest features you want to preserve.")
        
        with col2:
            st.subheader("Denoising")
            st.info("🔧 Reduces image noise while preserving important features. Choose method based on noise type.")
            
            apply_denoise = st.checkbox("Apply Denoising", key=f"denoise_{step_id}", value=True)
            denoise_method = st.selectbox("Method", ["median", "gaussian", "bilateral", "nlm"],
                                        key=f"denoise_method_{step_id}", disabled=not apply_denoise,
                                        help="median: Best for salt-and-pepper noise\ngaussian: General smoothing\nbilateral: Edge-preserving smoothing\nnlm: Advanced non-local means (slowest but best quality)")
            
            # Method-specific parameters
            denoise_params = {}
            if apply_denoise:
                if denoise_method == "median":
                    denoise_params["median_k_size"] = st.slider("Kernel Size", 1, 15, 3, step=2, 
                                                              key=f"median_k_{step_id}",
                                                              help="Size of median filter. Larger values remove more noise but blur fine details. Must be odd.")
                elif denoise_method == "gaussian":
                    denoise_params["gaussian_sigma"] = st.slider("Sigma", 0.1, 5.0, 1.0, step=0.1,
                                                               key=f"gauss_sigma_{step_id}",
                                                               help="Standard deviation of Gaussian kernel. Higher values = more smoothing but less detail preservation.")
                elif denoise_method == "bilateral":
                    denoise_params["bilateral_d"] = st.slider("Diameter", 1, 15, 5, key=f"bil_d_{step_id}",
                                                            help="Diameter of pixel neighborhood. Larger values = stronger filtering but slower processing.")
                    denoise_params["bilateral_sigma_color"] = st.slider("Sigma Color", 10, 150, 75, 
                                                                      key=f"bil_sc_{step_id}",
                                                                      help="Color similarity threshold. Higher values = more aggressive smoothing of different colors.")
                    denoise_params["bilateral_sigma_space"] = st.slider("Sigma Space", 10, 150, 75,
                                                                      key=f"bil_ss_{step_id}",
                                                                      help="Spatial distance threshold. Higher values = larger neighborhood considered for smoothing.")
                elif denoise_method == "nlm":
                    denoise_params["nlm_h"] = st.slider("Filter Strength", 0.1, 1.0, 0.5, step=0.1,
                                                      key=f"nlm_h_{step_id}",
                                                      help="Denoising strength. Higher values remove more noise but may over-smooth textures.")
    
    return {
        'apply_illumination_correction': apply_illum,
        'illumination_method': illum_method if apply_illum else None,
        'illumination_kernel_size': illum_kernel,
        'apply_denoising': apply_denoise,
        'denoise_method': denoise_method if apply_denoise else None,
        'denoise_params': denoise_params
    }

def create_artifact_removal_params_ui(step_id: str) -> Dict[str, Any]:
    """Create UI for artifact removal parameters."""
    with st.expander(f"🎯 Artifact Removal Parameters (Step {step_id})", expanded=True):
        st.info("💡 This step processes both bright and dark artifacts. Adjust detection sensitivity and processing parameters for your specific artifacts.")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Bright Spot Detection")
            st.info("🔆 Detects bright artifacts like dust, scratches, or overexposed regions.")
            
            apply_bright = st.checkbox("Remove Bright Spots", value=True, key=f"apply_bright_{step_id}")
            bright_detection_method = st.selectbox("Detection Method", ["percentile", "otsu", "absolute"],
                                                  key=f"bright_detect_method_{step_id}", disabled=not apply_bright,
                                                  help="percentile: Statistical threshold (most flexible)\notsu: Automatic threshold (good for bimodal images)\nabsolute: Fixed intensity threshold (most predictable)")
            
            if bright_detection_method == "percentile":
                bright_threshold_percentile = st.slider("Threshold Percentile", 0.0, 100.0, 95.0,
                                                       key=f"bright_thresh_perc_{step_id}", disabled=not apply_bright,
                                                       help="Pixels above this percentile are considered artifacts. Higher values = less sensitive (fewer artifacts detected). 95% means only the brightest 5% of pixels are considered.")
            elif bright_detection_method == "absolute":
                bright_absolute_threshold = st.slider("Absolute Threshold", 0, 255, 200,
                                                     key=f"bright_abs_thresh_{step_id}", disabled=not apply_bright,
                                                     help="Fixed intensity threshold. Pixels above this value are artifacts. Higher values = less sensitive. 255 = white, 0 = black.")
            
            st.subheader("Dark Spot Detection")
            st.info("🔻 Detects dark artifacts like debris, shadows, or underexposed regions.")
            
            apply_dark = st.checkbox("Remove Dark Spots", value=True, key=f"apply_dark_{step_id}")
            dark_detection_method = st.selectbox("Detection Method", ["percentile", "otsu", "absolute"],
                                                key=f"dark_detect_method_{step_id}", disabled=not apply_dark,
                                                help="Same methods as bright spots but for dark artifacts.")
            
            if dark_detection_method == "percentile":
                dark_threshold_percentile = st.slider("Threshold Percentile", 0.0, 100.0, 5.0,
                                                     key=f"dark_thresh_perc_{step_id}", disabled=not apply_dark,
                                                     help="Pixels below this percentile are considered artifacts. Lower values = less sensitive. 5% means only the darkest 5% of pixels are considered.")
            elif dark_detection_method == "absolute":
                dark_absolute_threshold = st.slider("Absolute Threshold", 0, 255, 50,
                                                   key=f"dark_abs_thresh_{step_id}", disabled=not apply_dark,
                                                   help="Fixed intensity threshold. Pixels below this value are artifacts. Lower values = less sensitive.")
        
        with col2:
            st.subheader("Processing Parameters")
            st.info("⚙️ Controls how detected artifacts are cleaned and filtered before removal.")
            
            apply_opening = st.checkbox("Apply Opening", value=True, key=f"opening_{step_id}",
                                      help="Morphological opening removes small noise pixels and separates connected artifacts.")
            opening_size = st.slider("Opening Size", 1, 15, 3, key=f"open_size_{step_id}", 
                                   disabled=not apply_opening,
                                   help="Size of opening operation. Larger values remove smaller artifacts but may break up larger ones. Must be odd.")
            
            min_area = st.slider("Min Area", 1, 1000, 10, key=f"min_area_{step_id}",
                               help="Minimum artifact size in pixels. Smaller artifacts are ignored. Increase to ignore tiny noise, decrease to catch small artifacts.")
            max_area = st.slider("Max Area", 100, 10000, 5000, key=f"max_area_{step_id}",
                               help="Maximum artifact size in pixels. Larger artifacts are ignored (may be actual features). Decrease if large features are being mistakenly removed.")
            
            st.subheader("Inpainting")
            st.info("🎨 How to fill in the removed artifact regions.")
            
            inpaint_method = st.selectbox("Inpainting Method", ["telea", "ns"], key=f"inpaint_{step_id}",
                                        help="telea: Fast marching method (faster, good for most cases)\nns: Navier-Stokes method (slower, better for complex textures)")
            inpaint_radius = st.slider("Inpainting Radius", 1, 10, 3, key=f"inpaint_r_{step_id}",
                                     help="How far around each artifact to use for inpainting. Larger values use more context but may introduce blurring.")
            
            # Dilation parameters
            dilation_size = st.slider("Mask Dilation Size", 0, 10, 2, key=f"dilation_{step_id}",
                                    help="Expands artifact masks before inpainting. Larger values ensure complete artifact removal but may remove more surrounding pixels.")
    
    # Compile parameters for both bright and dark spots
    params = {
        'process_both_types': True,
        'bright_spots': {
            'enabled': apply_bright,
            'detection_params': {
                'method': bright_detection_method,
            },
        },
        'dark_spots': {
            'enabled': apply_dark,
            'detection_params': {
                'method': dark_detection_method,
            },
        },
        'filtering_params': {
            'apply_opening': apply_opening,
            'opening_size': opening_size,
            'min_area': min_area,
            'max_area': max_area,
            'dilation_size': dilation_size,
            'dilation_shape': 'ellipse'
        },
        'inpainting_params': {
            'method': inpaint_method,
            'radius': inpaint_radius
        }
    }
    
    # Add threshold parameters based on detection method
    if apply_bright:
        if bright_detection_method == "percentile":
            params['bright_spots']['detection_params']['threshold_percentile'] = bright_threshold_percentile
        elif bright_detection_method == "absolute":
            params['bright_spots']['detection_params']['absolute_threshold'] = bright_absolute_threshold
    
    if apply_dark:
        if dark_detection_method == "percentile":
            params['dark_spots']['detection_params']['threshold_percentile'] = dark_threshold_percentile
        elif dark_detection_method == "absolute":
            params['dark_spots']['detection_params']['absolute_threshold'] = dark_absolute_threshold
    
    return params

def create_phase_analysis_params_ui(step_id: str) -> Dict[str, Dict[str, Any]]:
    """Create UI for phase analysis parameters."""
    with st.expander(f"🧪 Phase Analysis Parameters (Step {step_id})", expanded=True):
        st.info("🔬 Segments the material into different phases based on intensity patterns. Adjust parameters based on your material's characteristics.")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Masking")
            st.info("🎭 Creates material mask to separate sample from background.")
            
            masking_strategy = st.selectbox("Strategy", ["fill_holes", "bright_phases"], 
                                          key=f"mask_strat_{step_id}",
                                          help="fill_holes: Good for continuous materials with dark background\nbright_phases: Better when material phases are generally brighter than background")
            background_threshold = st.slider("Background Threshold", 0.0, 0.5, 0.08, step=0.01,
                                           key=f"bg_thresh_{step_id}",
                                           help="Intensity threshold to separate background from material. Lower values include more dark regions as background. Adjust if background/material separation is poor.")
            cleanup_area = st.slider("Cleanup Area", 0, 2000, 500, key=f"cleanup_{step_id}",
                                   help="Removes small disconnected regions from material mask. Larger values remove bigger noise regions but may remove small material features.")
            
            st.subheader("Artifact Removal")
            st.info("🧹 Cleans up the material before phase segmentation.")
            
            dark_spot_area = st.slider("Dark Spot Fill Area", 0, 200, 50, key=f"dark_area_{step_id}",
                                     help="Fills dark spots smaller than this size within material regions. Helps with imaging artifacts like pinholes or debris.")
            bright_spot_method = st.selectbox("Bright Spot Method", ["opening", "clipping"],
                                            key=f"bright_method_{step_id}",
                                            help="opening: Morphological opening to remove bright spots\nclipping: Intensity clipping to reduce bright outliers")
        
        with col2:
            st.subheader("Phase Detection")
            st.info("🔍 Determines how many phases are present in the material.")
            
            auto_detect = st.checkbox("Auto-detect Phases", value=True, key=f"auto_detect_{step_id}",
                                    help="Automatically detects number of phases using histogram analysis. Disable to manually specify phase count.")
            manual_phases = st.slider("Manual Phase Count", 1, 10, 3, disabled=auto_detect,
                                    key=f"manual_phases_{step_id}",
                                    help="Number of phases to segment when auto-detection is disabled. Should match the number of distinct materials in your sample.")
            
            st.subheader("Segmentation")
            st.info("✂️ How to separate the detected phases.")
            
            seg_method = st.selectbox("Method", ["auto", "kmeans", "otsu", "multiotsu", "percentile"],
                                    key=f"seg_method_{step_id}",
                                    help="auto: Automatically chooses best method\nkmeans: Clustering based on intensity\notsu: Automatic thresholding (2 phases)\nmultiotsu: Multiple threshold levels\npercentile: Statistical intensity-based separation")
            
            if seg_method == "kmeans":
                kmeans_random_state = st.slider("Random State", 0, 100, 42, key=f"kmeans_rs_{step_id}",
                                               help="Random seed for reproducible clustering results. Change if you want different initialization.")
    
    return {
        'masking': {
            'strategy': masking_strategy,
            'background_threshold': background_threshold,
            'cleanup_area': cleanup_area
        },
        'artifact_removal': {
            'dark_spot_fill_area': dark_spot_area,
            'bright_spot_method': bright_spot_method,
            'bright_spot_opening_size': 15,
            'bright_spot_clip_offset': 0.05
        },
        'phase_detection': {
            'auto_detect_phases': auto_detect,
            'histogram_bins': 256,
            'min_distance_bins': 5,
            'min_prominence_ratio': 0.05,
            'default_phases': manual_phases
        },
        'segmentation': {
            'num_phases': manual_phases,
            'method': seg_method,
            'kmeans_random_state': kmeans_random_state if seg_method == "kmeans" else 42
        }
    }

def create_line_analysis_params_ui(step_id: str) -> Dict[str, Dict[str, Any]]:
    """Create UI for line analysis parameters."""
    with st.expander(f"📏 Line Analysis Parameters (Step {step_id})", expanded=True):
        st.info("📐 Detects and analyzes linear features like grain boundaries, cracks, or fiber orientations. Requires a material mask from previous phase analysis.")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Frangi Filter")
            st.info("🔍 Enhances linear structures using multiscale filtering.")
            
            sigma_min = st.slider("Sigma Min", 1, 10, 1, key=f"sigma_min_{step_id}",
                                help="Minimum scale for line detection. Smaller values detect thinner lines. Start with 1 for fine details.")
            sigma_max = st.slider("Sigma Max", 2, 20, 8, key=f"sigma_max_{step_id}",
                                help="Maximum scale for line detection. Larger values detect thicker lines. Should be larger than your widest lines of interest.")
            sigma_step = st.slider("Sigma Step", 1, 5, 2, key=f"sigma_step_{step_id}",
                                 help="Step size between scales. Smaller steps = more thorough detection but slower processing.")
            black_ridges = st.checkbox("Detect Dark Lines", key=f"black_ridges_{step_id}",
                                     help="Enable to detect dark lines (e.g., grain boundaries). Disable for bright lines (e.g., cracks filled with bright material).")
            
            st.subheader("Boundary Exclusion")
            st.info("🚫 Excludes edge artifacts from analysis.")
            
            boundary_erosion = st.slider("Boundary Erosion Size", 0, 50, 10, key=f"boundary_{step_id}",
                                        help="Excludes this many pixels from material edges. Larger values remove more edge artifacts but may miss lines near boundaries.")
        
        with col2:
            st.subheader("Hough Transform")
            st.info("📊 Detects straight line segments geometrically.")
            
            hough_threshold = st.slider("Threshold", 1, 20, 5, key=f"hough_thresh_{step_id}",
                                      help="Minimum number of edge pixels required to form a line. Higher values = fewer, more confident line detections.")
            hough_min_length = st.slider("Min Line Length", 5, 100, 20, key=f"hough_len_{step_id}",
                                        help="Minimum length of detected lines in pixels. Shorter lines are ignored. Increase to focus on major features.")
            hough_max_gap = st.slider("Max Line Gap", 1, 50, 10, key=f"hough_gap_{step_id}",
                                    help="Maximum gap in pixels to connect line segments. Larger values connect more broken lines but may merge separate features.")
            
            st.subheader("Analysis")
            st.info("📈 Additional analysis options.")
            
            apply_skeleton = st.checkbox("Apply Skeletonization", value=True, key=f"skeleton_{step_id}",
                                       help="Reduces lines to single-pixel width for cleaner analysis. Recommended for most cases.")
            analyze_sobel = st.checkbox("Analyze Sobel Edges", value=True, key=f"sobel_{step_id}",
                                      help="Performs edge-based orientation analysis in addition to line detection. Provides complementary information about directional patterns.")
    
    return {
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
            'hough_histogram_bins': 60,
            'sobel_histogram_bins': 90,
            'sobel_magnitude_threshold': 0.01,
            'colormap': 'hsv'
        }
    }

# Operation execution functions
def execute_preprocessing(image: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
    """Execute preprocessing operation."""
    processed_image = image.copy()
    
    # Convert to uint8 for OpenCV operations if needed
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

def execute_artifact_removal(image: np.ndarray, params: Dict[str, Any]) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Execute artifact removal operation for both bright and dark spots."""
    
    # Check if we're using the new combined format
    if params.get('process_both_types', False):
        return execute_combined_artifact_removal(image, params)
    else:
        # Legacy single spot type format
        analyzer = ArtifactRemover()
        results = analyzer.run_full_analysis(image, **params)
        
        if results['success'] and results.get('artifacts_processed', False):
            processed_image = results['processed_image'].astype(np.float32) / 255.0
            return processed_image, results
        else:
            return image.copy(), results

def execute_combined_artifact_removal(image: np.ndarray, params: Dict[str, Any]) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Execute artifact removal for both bright and dark spots."""
    
    # Initialize results
    combined_results = {
        'success': True,
        'artifacts_processed': False,
        'bright_spots_processed': False,
        'dark_spots_processed': False,
        'error_message': None,
        'artifact_statistics': {},
        'analysis_results': {}
    }
    
    current_image = image.copy()
    total_artifacts_processed = False
    
    try:
        # Process bright spots if enabled
        if params['bright_spots']['enabled']:
            bright_params = {
                'spot_type': 'bright',
                'detection_params': params['bright_spots']['detection_params'],
                'filtering_params': params['filtering_params'],
                'inpainting_params': params['inpainting_params']
            }
            
            analyzer_bright = ArtifactRemover()
            bright_results = analyzer_bright.run_full_analysis(current_image, **bright_params)
            
            if bright_results['success'] and bright_results.get('artifacts_processed', False):
                current_image = bright_results['processed_image'].astype(np.float32) / 255.0
                combined_results['bright_spots_processed'] = True
                total_artifacts_processed = True
                
                # Store bright spot statistics
                combined_results['artifact_statistics']['bright_spots'] = bright_results.get('artifact_statistics', {})
        
        # Process dark spots if enabled
        if params['dark_spots']['enabled']:
            # Convert current image back to uint8 for processing
            if current_image.dtype != np.uint8:
                current_image_uint8 = (current_image * 255).astype(np.uint8)
            else:
                current_image_uint8 = current_image
            
            dark_params = {
                'spot_type': 'dark',
                'detection_params': params['dark_spots']['detection_params'],
                'filtering_params': params['filtering_params'],
                'inpainting_params': params['inpainting_params']
            }
            
            analyzer_dark = ArtifactRemover()
            dark_results = analyzer_dark.run_full_analysis(current_image_uint8, **dark_params)
            
            if dark_results['success'] and dark_results.get('artifacts_processed', False):
                current_image = dark_results['processed_image'].astype(np.float32) / 255.0
                combined_results['dark_spots_processed'] = True
                total_artifacts_processed = True
                
                # Store dark spot statistics
                combined_results['artifact_statistics']['dark_spots'] = dark_results.get('artifact_statistics', {})
        
        # Update combined results
        combined_results['artifacts_processed'] = total_artifacts_processed
        combined_results['processed_image'] = current_image
        
        # Combine analysis results
        combined_results['analysis_results'] = {
            'bright_spots_enabled': params['bright_spots']['enabled'],
            'dark_spots_enabled': params['dark_spots']['enabled'],
            'bright_spots_processed': combined_results['bright_spots_processed'],
            'dark_spots_processed': combined_results['dark_spots_processed'],
            'total_artifacts_processed': total_artifacts_processed,
            'inpainting_method': params['inpainting_params']['method']
        }
        
    except Exception as e:
        combined_results['success'] = False
        combined_results['error_message'] = str(e)
        current_image = image.copy()
    
    return current_image, combined_results

def execute_phase_analysis(image: np.ndarray, params: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Execute phase analysis operation."""
    analyzer = PhaseAnalyzer()
    
    # Convert nested parameter structure to the expected format
    analysis_params = {
        'preprocessing_params': params.get('preprocessing', {}),
        'artifact_removal_params': params.get('artifact_removal', {}),
        'masking_params': params.get('masking', {}),
        'phase_detection_params': {
            # Filter out the UI-specific parameter and only pass the method's expected parameters
            'histogram_bins': params.get('phase_detection', {}).get('histogram_bins', 256),
            'min_distance_bins': params.get('phase_detection', {}).get('min_distance_bins', 5),
            'min_prominence_ratio': params.get('phase_detection', {}).get('min_prominence_ratio', 0.05),
            'default_phases': params.get('phase_detection', {}).get('default_phases', 3)
        },
        'segmentation_params': params.get('segmentation', {}),
        'visualization_params': params.get('visualization', {})
    }
    
    results = analyzer.run_full_analysis(image, **analysis_params)
    return results

def execute_line_analysis(image: np.ndarray, material_mask: np.ndarray, 
                         params: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Execute line analysis operation."""
    analyzer = LineAnalyzer()
    results = analyzer.run_full_analysis(image, material_mask, **params)
    return results

def create_auto_material_mask(image: np.ndarray) -> np.ndarray:
    """Create a simple material mask using Otsu thresholding."""
    if image.dtype != np.uint8:
        image_uint8 = (image * 255).astype(np.uint8)
    else:
        image_uint8 = image
    
    _, mask = cv2.threshold(image_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return mask.astype(bool)

def display_pipeline_results():
    """Display results from pipeline execution."""
    if not st.session_state.pipeline_results:
        return
    
    st.header("📊 Pipeline Results")
    
    # Show pipeline summary
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Steps", len(st.session_state.pipeline_steps))
    with col2:
        processing_steps = sum(1 for step in st.session_state.pipeline_steps 
                             if step['operation'] in ['preprocessing', 'artifact_removal'])
        st.metric("Processing Steps", processing_steps)
    with col3:
        analysis_steps = sum(1 for step in st.session_state.pipeline_steps 
                           if step['operation'] in ['phase_analysis', 'line_analysis'])
        st.metric("Analysis Steps", analysis_steps)
    
    # Display results for each step
    for i, (step, result) in enumerate(zip(st.session_state.pipeline_steps, st.session_state.pipeline_results)):
        operation = step['operation']
        
        with st.expander(f"Step {i+1}: {operation.replace('_', ' ').title()}", expanded=True):
            if operation in ['preprocessing', 'artifact_removal']:
                # Show before/after for processing steps
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Input Image**")
                    if i == 0:
                        st.image(st.session_state.original_image, use_container_width=True)
                    else:
                        # Find the last processing step's output
                        prev_image = st.session_state.original_image
                        for j in range(i):
                            if st.session_state.pipeline_steps[j]['operation'] in ['preprocessing', 'artifact_removal']:
                                if isinstance(st.session_state.pipeline_results[j], tuple):
                                    prev_image = st.session_state.pipeline_results[j][0]
                                else:
                                    prev_image = st.session_state.pipeline_results[j]
                        st.image(prev_image, use_container_width=True)
                
                with col2:
                    st.write("**Output Image**")
                    if isinstance(result, tuple):
                        output_image = result[0]
                    else:
                        output_image = result
                    st.image(output_image, use_container_width=True)
                
                # Show additional info for artifact removal
                if operation == 'artifact_removal' and isinstance(result, tuple):
                    artifact_results = result[1]
                    
                    # Handle combined artifact removal results
                    if artifact_results.get('bright_spots_processed', False) or artifact_results.get('dark_spots_processed', False):
                        st.success("✅ Artifacts were detected and processed")
                        
                        # Show processing status for each type
                        col1, col2 = st.columns(2)
                        with col1:
                            if artifact_results.get('bright_spots_processed', False):
                                st.success("🔆 Bright spots processed")
                                bright_stats = artifact_results.get('artifact_statistics', {}).get('bright_spots', {})
                                if bright_stats:
                                    st.metric("Bright Components", 
                                            bright_stats.get('initial_artifacts', {}).get('num_components', 0))
                            else:
                                st.info("🔆 Bright spots: not processed")
                        
                        with col2:
                            if artifact_results.get('dark_spots_processed', False):
                                st.success("🔻 Dark spots processed")
                                dark_stats = artifact_results.get('artifact_statistics', {}).get('dark_spots', {})
                                if dark_stats:
                                    st.metric("Dark Components", 
                                            dark_stats.get('initial_artifacts', {}).get('num_components', 0))
                            else:
                                st.info("🔻 Dark spots: not processed")
                    
                    # Handle legacy single-type artifact removal
                    elif artifact_results.get('artifacts_processed', False):
                        st.success("✅ Artifacts were detected and processed")
                        stats = artifact_results.get('artifact_statistics', {})
                        if stats:
                            col1, col2 = st.columns(2)
                            with col1:
                                st.metric("Components Detected", 
                                        stats.get('initial_artifacts', {}).get('num_components', 0))
                            with col2:
                                st.metric("Coverage %", 
                                        f"{stats.get('final_artifacts', {}).get('coverage_percent', 0):.3f}")
                    else:
                        st.info("ℹ️ No artifacts were detected or processed")
            
            elif operation in ['phase_analysis', 'line_analysis']:
                # Show analysis results
                if result.get('success', False):
                    st.success("✅ Analysis completed successfully")
                    
                    if operation == 'phase_analysis':
                        analysis_results = result['analysis_results']
                        phase_stats = result['phase_statistics']
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Phases Detected", analysis_results['num_phases_detected'])
                        with col2:
                            st.metric("Material Pixels", analysis_results['total_material_pixels'])
                        with col3:
                            st.metric("Method", analysis_results['segmentation_method'])
                        
                        # Show phase statistics
                        if phase_stats:
                            st.write("**Phase Statistics:**")
                            stats_data = []
                            for phase_name, stats in phase_stats.items():
                                stats_data.append({
                                    "Phase": phase_name.replace('_', ' ').title(),
                                    "Pixels": stats['pixels'],
                                    "Area Fraction": f"{stats['fraction']:.4f}"
                                })
                            st.table(stats_data)
                    
                    elif operation == 'line_analysis':
                        analysis_results = result['analysis_results']
                        
                        col1, col2, col3 = st.columns(3)
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
                else:
                    st.error(f"❌ Analysis failed: {result.get('error_message', 'Unknown error')}")

def main():
    """Main Streamlit application."""
    initialize_session_state()
    
    st.title("🔬 Custom Pipeline Builder")
    st.markdown("*Build custom analysis workflows by chaining operations*")
    
    # Image upload section
    st.header("📤 Image Upload")
    col1, col2 = st.columns([2, 1])
    
    with col1:
        uploaded_file = st.file_uploader(
            "Choose an image file",
            type=['png', 'jpg', 'jpeg', 'tiff', 'tif']
        )
        
        if uploaded_file is not None:
            image = load_and_normalize_image(uploaded_file)
            if image is not None:
                st.session_state.original_image = image
                st.session_state.current_image = image
                st.session_state.output_dir = create_output_directory(uploaded_file.name)
                st.success("✅ Image loaded successfully")
                
                # Display the uploaded image
                st.image(st.session_state.original_image, caption="Uploaded Image", use_container_width=True)
            else:
                st.error("❌ Failed to load image")
                return
        else:
            st.warning("⚠️ Please upload an image to begin")
            return
    
    with col2:
        if st.session_state.original_image is not None:
            st.write("**Image Statistics:**")
            st.metric("Width", st.session_state.original_image.shape[1])
            st.metric("Height", st.session_state.original_image.shape[0])
            st.metric("Min Value", f"{st.session_state.original_image.min():.3f}")
            st.metric("Max Value", f"{st.session_state.original_image.max():.3f}")
            st.metric("Mean", f"{st.session_state.original_image.mean():.3f}")
    
    # Pipeline Steps Management
    st.header("🔧 Pipeline Steps")
    
    # Show current pipeline steps first
    if st.session_state.pipeline_steps:
        st.subheader("Current Pipeline")
        
        # Display pipeline steps in a visual way
        operation_options = {
            "preprocessing": "🔧 Preprocessing",
            "artifact_removal": "🎯 Artifact Removal",
            "phase_analysis": "🧪 Phase Analysis",
            "line_analysis": "📏 Line Analysis"
        }
        
        pipeline_display = []
        for i, step in enumerate(st.session_state.pipeline_steps):
            pipeline_display.append(f"{i+1}. {operation_options[step['operation']]}")
        
        # Show pipeline flow
        st.write(" → ".join(pipeline_display))
        
        # Individual step removal
        st.write("**Remove Individual Steps:**")
        cols = st.columns(len(st.session_state.pipeline_steps))
        for i, (col, step) in enumerate(zip(cols, st.session_state.pipeline_steps)):
            with col:
                if st.button(f"❌ Step {i+1}", key=f"remove_{i}", help=f"Remove {operation_options[step['operation']]}"):
                    st.session_state.pipeline_steps.pop(i)
                    st.rerun()
    else:
        st.info("No pipeline steps added yet. Use the controls below to add steps.")
    
    # Configure pipeline steps (only if there are steps)
    if st.session_state.pipeline_steps:
        st.header("⚙️ Configure Pipeline Steps")
        
        # Update parameters for each step
        for i, step in enumerate(st.session_state.pipeline_steps):
            operation = step['operation']
            step_id = f"{i+1}"
            
            if operation == 'preprocessing':
                params = create_preprocessing_params_ui(step_id)
            elif operation == 'artifact_removal':
                params = create_artifact_removal_params_ui(step_id)
            elif operation == 'phase_analysis':
                params = create_phase_analysis_params_ui(step_id)
            elif operation == 'line_analysis':
                params = create_line_analysis_params_ui(step_id)
            
            st.session_state.pipeline_steps[i]['params'] = params
    
    # Add step section - moved after configuration and before execute 
    st.header("➕ Add New Step")
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        operation_options = {
            "preprocessing": "🔧 Preprocessing",
            "artifact_removal": "🎯 Artifact Removal",
            "phase_analysis": "🧪 Phase Analysis",
            "line_analysis": "📏 Line Analysis"
        }
        
        new_operation = st.selectbox(
            "Select operation to add",
            options=list(operation_options.keys()),
            format_func=lambda x: operation_options[x]
        )
    
    with col2:
        if st.button("➕ Add Step", use_container_width=True):
            step_id = len(st.session_state.pipeline_steps) + 1
            st.session_state.pipeline_steps.append({
                'id': step_id,
                'operation': new_operation,
                'params': {}
            })
            st.rerun()
    
    with col3:
        if st.button("🗑️ Clear All", use_container_width=True):
            st.session_state.pipeline_steps = []
            st.session_state.pipeline_results = []
            st.rerun()
    
    # Configure pipeline steps (only if there are steps)
    if st.session_state.pipeline_steps:
        st.header("🚀 Execute Pipeline")
        
        col1, col2, col3 = st.columns([1, 1, 2])
        
        with col1:
            if st.button("▶️ Run Pipeline", type="primary", use_container_width=True):
                execute_pipeline()
        
        with col2:
            if st.button("💾 Save Pipeline", use_container_width=True):
                save_pipeline()
        
        with col3:
            uploaded_pipeline = st.file_uploader("📂 Load Pipeline", type=['json'])
            if uploaded_pipeline is not None:
                load_pipeline(uploaded_pipeline)
        
        # Display results
        display_pipeline_results()

def execute_pipeline():
    """Execute the complete pipeline."""
    if not st.session_state.pipeline_steps:
        st.error("No pipeline steps to execute")
        return
    
    st.session_state.pipeline_results = []
    current_image = st.session_state.original_image.copy()
    material_mask = st.session_state.material_mask
    
    def prepare_for_json(obj):
        """Recursively prepare object for JSON serialization."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        elif isinstance(obj, dict):
            return {k: prepare_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [prepare_for_json(item) for item in obj]
        elif isinstance(obj, (str, int, float, bool)) or obj is None:
            return obj
        else:
            return str(obj)
    
    with st.spinner("Executing pipeline..."):
        for i, step in enumerate(st.session_state.pipeline_steps):
            operation = step['operation']
            params = step['params']
            
            try:
                st.write(f"Executing Step {i+1}: {operation.replace('_', ' ').title()}...")
                
                if operation == 'preprocessing':
                    result = execute_preprocessing(current_image, params)
                    current_image = result
                    st.session_state.pipeline_results.append(result)
                    
                    # Save intermediate result
                    if st.session_state.output_dir:
                        filename = f"step_{i+1}_preprocessing.png"
                        save_image(result, filename, st.session_state.output_dir)
                
                elif operation == 'artifact_removal':
                    result, analysis_results = execute_artifact_removal(current_image, params)
                    current_image = result
                    st.session_state.pipeline_results.append((result, analysis_results))
                    
                    # Save intermediate result
                    if st.session_state.output_dir:
                        filename = f"step_{i+1}_artifact_removal.png"
                        save_image(result, filename, st.session_state.output_dir)
                
                elif operation == 'phase_analysis':
                    # Create material mask if not available
                    if material_mask is None:
                        material_mask = create_auto_material_mask(current_image)
                        st.session_state.material_mask = material_mask
                    
                    result = execute_phase_analysis(current_image, params)
                    st.session_state.pipeline_results.append(result)
                    
                    # Update material mask from phase analysis
                    if result.get('success', False) and 'material_mask' in result:
                        material_mask = result['material_mask']
                        st.session_state.material_mask = material_mask
                    
                    # Save results
                    if st.session_state.output_dir and result.get('success', False):
                        # Save analysis results as JSON
                        results_file = f"step_{i+1}_phase_analysis_results.json"
                        results_path = os.path.join(st.session_state.output_dir, results_file)
                        
                        # Prepare results for JSON using the recursive function
                        results_to_save = prepare_for_json(result)
                        
                        with open(results_path, 'w') as f:
                            json.dump(results_to_save, f, indent=2)
                
                elif operation == 'line_analysis':
                    # Create material mask if not available
                    if material_mask is None:
                        material_mask = create_auto_material_mask(current_image)
                        st.session_state.material_mask = material_mask
                    
                    result = execute_line_analysis(current_image, material_mask, params)
                    st.session_state.pipeline_results.append(result)
                    
                    # Save results
                    if st.session_state.output_dir and result.get('success', False):
                        # Save analysis results as JSON
                        results_file = f"step_{i+1}_line_analysis_results.json"
                        results_path = os.path.join(st.session_state.output_dir, results_file)
                        
                        # Prepare results for JSON using the recursive function
                        results_to_save = prepare_for_json(result)
                        
                        with open(results_path, 'w') as f:
                            json.dump(results_to_save, f, indent=2)
                
                st.success(f"✅ Step {i+1} completed")
                
            except Exception as e:
                st.error(f"❌ Error in Step {i+1}: {str(e)}")
                st.exception(e)
                break
    
    st.success("🎉 Pipeline execution completed!")

def save_pipeline():
    """Save current pipeline configuration."""
    if not st.session_state.pipeline_steps:
        st.warning("No pipeline to save")
        return
    
    pipeline_data = {
        'pipeline_steps': st.session_state.pipeline_steps,
        'created_at': datetime.now().isoformat(),
        'version': '1.0'
    }
    
    # Convert to JSON string
    json_str = json.dumps(pipeline_data, indent=2)
    
    # Offer download
    st.download_button(
        label="💾 Download Pipeline",
        data=json_str,
        file_name=f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
        mime="application/json"
    )

def load_pipeline(uploaded_file):
    """Load pipeline configuration from file."""
    try:
        pipeline_data = json.load(uploaded_file)
        
        if 'pipeline_steps' in pipeline_data:
            st.session_state.pipeline_steps = pipeline_data['pipeline_steps']
            st.session_state.pipeline_results = []
            st.success("✅ Pipeline loaded successfully!")
            st.rerun()
        else:
            st.error("❌ Invalid pipeline file format")
            
    except Exception as e:
        st.error(f"❌ Error loading pipeline: {str(e)}")

if __name__ == "__main__":
    main()