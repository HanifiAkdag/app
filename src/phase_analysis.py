import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from skimage import filters, morphology, img_as_float
from skimage.morphology import disk, remove_small_objects, remove_small_holes, binary_erosion
from scipy.signal import find_peaks
from scipy.ndimage import binary_fill_holes
from sklearn.cluster import KMeans
from typing import Dict, List, Tuple, Optional, Any, Union
import os

from .utils import load_and_prepare_image
from .preprocessing import correct_illumination, denoise_image

class PhaseAnalyzer:
    """
    A utility class for performing phase analysis on microscopy images.
    """
    
    def __init__(self):
        self.results = {}
        self.intermediate_images = {}
    
    def _get_threshold_value(self, 
                           image: np.ndarray, 
                           method_or_value: Union[str, float]) -> float:
        """
        Helper to get threshold value from method name or direct value.
        
        Args:
            image: Input image region
            method_or_value: Threshold method name or direct float value
            
        Returns:
            Threshold value
        """
        if image.size == 0:
            return 0.5
        
        if isinstance(method_or_value, str):
            method_name = method_or_value.lower()
            try:
                if method_name == 'multiotsu':
                    return filters.threshold_otsu(image)
                
                thresh_func = getattr(filters, f'threshold_{method_name}')
                return thresh_func(image)
            except (AttributeError, ValueError) as e:
                # Fallback to Otsu
                min_val, max_val = np.min(image), np.max(image)
                if np.isclose(min_val, max_val):
                    return (min_val + max_val) / 2.0
                return filters.threshold_otsu(image)
        else:
            return float(method_or_value)
    
    def remove_artifacts(self, 
                        image: np.ndarray,
                        material_mask: np.ndarray,
                        background_mask: np.ndarray,
                        dark_spot_fill_area: int = 50,
                        dark_spot_threshold_method: str = 'otsu',
                        bright_spot_method: str = 'opening',
                        bright_spot_opening_size: int = 15,
                        bright_spot_clip_offset: float = 0.05) -> np.ndarray:
        """
        Removes dark spots on material and bright spots in background.
        
        Args:
            image: Input image after median filtering
            material_mask: Binary mask of material region
            background_mask: Binary mask of background region
            dark_spot_fill_area: Maximum area for dark spot filling
            dark_spot_threshold_method: Method for dark spot detection
            bright_spot_method: Method for bright spot removal ('opening' or 'clipping')
            bright_spot_opening_size: Size for morphological opening
            bright_spot_clip_offset: Offset for clipping method
            
        Returns:
            Cleaned image
        """
        image_cleaned = image.copy()
        
        # Fill dark spots on material using remove_small_holes
        if dark_spot_fill_area > 0 and material_mask.any():
            try:
                # Get threshold for material region
                material_pixels = image_cleaned[material_mask]
                if material_pixels.size > 0:
                    thresh_val = self._get_threshold_value(material_pixels, dark_spot_threshold_method)
                    
                    # Create binary mask and fill holes
                    temp_binary = image_cleaned > thresh_val
                    filled_binary = remove_small_holes(temp_binary, area_threshold=dark_spot_fill_area, connectivity=1)
                    fill_mask = filled_binary & (~temp_binary)
                    
                    num_pixels_filled = np.sum(fill_mask)
                    if num_pixels_filled > 0:
                        image_cleaned[fill_mask] = thresh_val
                        
            except Exception as e:
                pass  # Skip if error occurs
        
        # Remove bright spots in background
        if bright_spot_method == 'opening' and bright_spot_opening_size > 0:
            opened_image = morphology.opening(image_cleaned, footprint=disk(bright_spot_opening_size))
            image_cleaned[background_mask] = opened_image[background_mask]
            
        elif bright_spot_method == 'clipping' and bright_spot_clip_offset > 0 and background_mask.any():
            bg_pixels = image_cleaned[background_mask]
            if bg_pixels.size > 0:
                bg_median = np.median(bg_pixels)
                clip_threshold = bg_median + bright_spot_clip_offset
                bright_spots = (image_cleaned > clip_threshold) & background_mask
                image_cleaned[bright_spots] = bg_median
        
        self.intermediate_images['artifact_removed'] = image_cleaned
        return image_cleaned
    
    def create_material_mask(self, 
                           image: np.ndarray,
                           strategy: str = "fill_holes",
                           background_threshold: Union[str, float] = 0.08,
                           cleanup_area: int = 500) -> Tuple[np.ndarray, np.ndarray]:
        """
        Creates material and background masks using different strategies.
        
        Args:
            image: Input image (typically median filtered)
            strategy: Masking strategy ("fill_holes" or "bright_phases")
            background_threshold: Threshold for background identification
            cleanup_area: Area threshold for mask cleaning
            
        Returns:
            Tuple of (material_mask, background_mask)
        """
        if strategy == "fill_holes":
            # Identify background pixels (very dark)
            thresh_bg = self._get_threshold_value(image, background_threshold)
            background_only_mask = image < thresh_bg
            
            # Invert to get everything else (material + internal holes)
            initial_mask = ~background_only_mask
            
            # Fill internal holes
            filled_mask = binary_fill_holes(initial_mask)
            
            # Optional cleanup of external objects
            if cleanup_area > 0 and filled_mask.any():
                final_material_mask = remove_small_objects(filled_mask, min_size=cleanup_area, connectivity=1)
            else:
                final_material_mask = filled_mask
                
        elif strategy == "bright_phases":
            # Original method: threshold for brighter phases
            thresh_bright = self._get_threshold_value(image, background_threshold)
            final_material_mask = image > thresh_bright
            
            if cleanup_area > 0:
                final_material_mask = remove_small_objects(final_material_mask, min_size=cleanup_area, connectivity=1)
                final_material_mask = remove_small_holes(final_material_mask, area_threshold=cleanup_area, connectivity=1)
        
        else:
            raise ValueError(f"Unknown masking strategy: {strategy}")
        
        background_mask = ~final_material_mask
        
        self.intermediate_images['material_mask'] = final_material_mask
        self.intermediate_images['background_mask'] = background_mask
        
        return final_material_mask, background_mask
    
    def detect_number_of_phases(self, 
                              image: np.ndarray,
                              material_mask: np.ndarray,
                              histogram_bins: int = 256,
                              min_distance_bins: int = 5,
                              min_prominence_ratio: float = 0.05,
                              default_phases: int = 2) -> Tuple[int, Optional[np.ndarray], Optional[Dict]]:
        """
        Automatically detects the number of phases based on histogram peaks.
        
        Args:
            image: Cleaned image for analysis
            material_mask: Binary mask of material region
            histogram_bins: Number of histogram bins
            min_distance_bins: Minimum distance between peaks
            min_prominence_ratio: Minimum prominence as ratio of max histogram value
            default_phases: Default number of phases if detection fails
            
        Returns:
            Tuple of (num_phases, peak_intensities, histogram_data)
        """
        if not material_mask.any():
            return default_phases, None, None
        
        material_pixels = image[material_mask]
        if material_pixels.size == 0:
            return default_phases, None, None
        
        # Calculate histogram
        hist, bin_edges = np.histogram(material_pixels, bins=histogram_bins, range=(0, 1))
        bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
        min_prominence = min_prominence_ratio * np.max(hist) if np.max(hist) > 0 else 1
        
        # Find peaks
        peaks, properties = find_peaks(hist, distance=min_distance_bins, prominence=min_prominence)
        num_detected_phases = len(peaks)
        
        peak_intensities = None
        if num_detected_phases > 0:
            peak_intensities = bin_centers[peaks]
        
        final_num_phases = max(2, num_detected_phases) if num_detected_phases > 0 else default_phases
        
        hist_data = {
            'hist': hist,
            'bin_centers': bin_centers,
            'peaks': peaks,
            'properties': properties,
            'peak_intensities': peak_intensities
        }
        
        return final_num_phases, peak_intensities, hist_data
    
    def perform_phase_segmentation(self, 
                                 image: np.ndarray,
                                 material_mask: np.ndarray,
                                 num_phases: int,
                                 method: str = 'auto',
                                 manual_thresholds: Optional[List[float]] = None,
                                 kmeans_random_state: Optional[int] = 42,
                                 kmeans_n_init: Union[int, str] = 'auto') -> Tuple[List[np.ndarray], Union[List[float], np.ndarray]]:
        """
        Segments the image into phases within the material mask.
        
        Args:
            image: Cleaned image for segmentation
            material_mask: Binary mask of material region
            num_phases: Number of phases to segment
            method: Segmentation method ('auto', 'otsu', 'multiotsu', 'percentile', 'kmeans', 'manual')
            manual_thresholds: List of manual thresholds for manual method
            kmeans_random_state: Random state for KMeans
            kmeans_n_init: Number of initializations for KMeans
            
        Returns:
            Tuple of (phase_masks_list, thresholds_or_centers)
        """
        phase_masks = []
        thresholds_or_centers = []
        
        if not material_mask.any() or num_phases <= 0:
            return phase_masks, thresholds_or_centers
        
        material_pixels = image[material_mask]
        if material_pixels.size == 0:
            return phase_masks, thresholds_or_centers
        
        actual_method = method
        
        # Resolve 'auto' method
        if actual_method == 'auto':
            if num_phases == 1:
                actual_method = 'none'
            elif num_phases == 2:
                actual_method = 'otsu'
            else:
                actual_method = 'multiotsu'
        
        # Force percentile for many phases
        if num_phases > 5:
            actual_method = 'percentile'
        
        # KMeans method
        if actual_method == 'kmeans':
            try:
                pixel_data = material_pixels.reshape(-1, 1)
                kmeans = KMeans(n_clusters=num_phases, random_state=kmeans_random_state, n_init=kmeans_n_init)
                kmeans.fit(pixel_data)
                labels_flat = kmeans.labels_
                cluster_centers = kmeans.cluster_centers_.flatten()
                sorted_indices = np.argsort(cluster_centers)
                sorted_centers = cluster_centers[sorted_indices]
                thresholds_or_centers = sorted_centers
                
                # Create phase masks
                label_map = np.full(image.shape, -1, dtype=int)
                label_map[material_mask] = labels_flat
                
                for i in range(num_phases):
                    original_label = sorted_indices[i]
                    phase_mask_i = (label_map == original_label)
                    phase_masks.append(phase_mask_i)
                    
            except Exception as e:
                return [], []
        
        # Thresholding methods
        else:
            thresholds = []
            
            if num_phases == 1:
                phase_masks.append(material_mask)
                return phase_masks, thresholds
            
            elif actual_method == 'manual' and manual_thresholds:
                if len(manual_thresholds) == num_phases - 1:
                    thresholds = sorted([float(t) for t in manual_thresholds])
                else:
                    return [], []
                    
            elif actual_method == 'otsu':
                thresholds = [self._get_threshold_value(material_pixels, 'otsu')]
                
            elif actual_method == 'multiotsu':
                if num_phases <= 2:
                    thresholds = [self._get_threshold_value(material_pixels, 'otsu')]
                else:
                    try:
                        if len(np.unique(material_pixels)) > num_phases:
                            thresholds = filters.threshold_multiotsu(material_pixels, classes=num_phases)
                        else:
                            # Fallback to percentile
                            percentile_values = np.linspace(0, 100, num_phases + 1)[1:-1]
                            thresholds = np.percentile(material_pixels, percentile_values)
                    except Exception:
                        # Fallback to percentile
                        percentile_values = np.linspace(0, 100, num_phases + 1)[1:-1]
                        thresholds = np.percentile(material_pixels, percentile_values)
                        
            elif actual_method == 'percentile':
                percentile_values = np.linspace(0, 100, num_phases + 1)[1:-1]
                thresholds = np.percentile(material_pixels, percentile_values)
            
            else:
                return [], []
            
            # Create phase masks from thresholds
            thresholds = sorted(list(np.unique(thresholds)))
            thresholds_or_centers = thresholds
            
            last_threshold = 0.0
            for thresh in thresholds:
                phase_mask = (image >= last_threshold) & (image < thresh) & material_mask
                phase_masks.append(phase_mask)
                last_threshold = thresh
            
            # Final phase mask
            final_phase_mask = (image >= last_threshold) & material_mask
            phase_masks.append(final_phase_mask)
        
        self.intermediate_images['phase_masks'] = phase_masks
        return phase_masks, thresholds_or_centers
    
    def calculate_phase_statistics(self, 
                                 phase_masks: List[np.ndarray],
                                 material_mask: np.ndarray) -> Dict[str, Dict[str, float]]:
        """
        Calculates area fractions for each phase.
        
        Args:
            phase_masks: List of binary phase masks
            material_mask: Binary mask of material region
            
        Returns:
            Dictionary of phase statistics
        """
        stats = {}
        total_material_pixels = np.sum(material_mask)
        
        if total_material_pixels == 0 or not phase_masks:
            return stats
        
        for i, p_mask in enumerate(phase_masks):
            phase_pixels = np.sum(p_mask)
            fraction = phase_pixels / total_material_pixels if total_material_pixels > 0 else 0
            stats[f'phase_{i+1}'] = {
                'pixels': int(phase_pixels),
                'fraction': float(fraction)
            }
        
        return stats
    
    def create_segmentation_visualization(self, 
                                        original_image: np.ndarray,
                                        cleaned_image: np.ndarray,
                                        phase_masks: List[np.ndarray],
                                        background_mask: np.ndarray,
                                        color_mode: str = 'palette',
                                        palette_name: str = 'viridis',
                                        phase_colors: List[str] = None) -> plt.Figure:
        """
        Creates segmentation visualization plot.
        
        Args:
            original_image: Original input image
            cleaned_image: Cleaned image for intensity calculations
            phase_masks: List of phase masks
            background_mask: Background mask
            color_mode: Coloring mode ('palette' or 'intensity')
            palette_name: Name of matplotlib colormap
            phase_colors: List of color names for fallback
            
        Returns:
            Matplotlib figure object
        """
        if phase_colors is None:
            phase_colors = ['red', 'lime', 'blue', 'yellow', 'cyan', 'pink']
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 6), constrained_layout=True)
        fig.suptitle("Phase Analysis Segmentation Result", fontsize=16)
        
        # Plot 1: Original Image
        axes[0].imshow(original_image, cmap='gray')
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        # Plot 2: Segmented Image
        axes[1].set_title(f'Segmented Phases ({color_mode} colors)')
        axes[1].axis('off')
        
        segmented_image_vis = np.zeros((*original_image.shape, 3), dtype=float)
        segmented_image_vis[background_mask] = [0, 0, 0]  # Black background
        legend_handles = []
        
        if color_mode == 'intensity':
            # Intensity-based coloring
            for i, p_mask in enumerate(phase_masks):
                if p_mask.any():
                    mean_intensity = np.mean(cleaned_image[p_mask])
                    rgb_color = [mean_intensity] * 3
                    segmented_image_vis[p_mask] = rgb_color
                    label = f'Phase {i+1} (Avg: {mean_intensity:.3f})'
                    patch = plt.Rectangle((0, 0), 1, 1, fc=rgb_color)
                    legend_handles.append((patch, label))
                    
        elif color_mode == 'palette':
            # Palette-based coloring
            cmap = None
            try:
                cmap = plt.get_cmap(palette_name)
            except ValueError:
                cmap = None
            
            for i, p_mask in enumerate(phase_masks):
                rgb_color = None
                label = f'Phase {i+1}'
                
                if cmap:
                    color_val = min(i / max(1, len(phase_masks)-1), 1.0) if len(phase_masks) > 1 else 0.5
                    rgb_color = cmap(color_val)[:3]
                else:
                    color_idx = i % len(phase_colors)
                    color_name = phase_colors[color_idx]
                    try:
                        rgb_color = mcolors.to_rgb(color_name)
                    except ValueError:
                        default_cmap = plt.get_cmap('viridis')
                        color_val = min(i / max(1, len(phase_masks)-1), 1.0) if len(phase_masks) > 1 else 0.5
                        rgb_color = default_cmap(color_val)[:3]
                
                if rgb_color is not None:
                    segmented_image_vis[p_mask] = rgb_color
                    patch = plt.Rectangle((0, 0), 1, 1, fc=rgb_color)
                    legend_handles.append((patch, label))
        
        # Display segmented image
        axes[1].imshow(segmented_image_vis)
        
        # Add legend
        if legend_handles:
            patches, labels = zip(*legend_handles)
            fig.legend(patches, labels, loc='center right', bbox_to_anchor=(1.05, 0.5), title="Phases")
        
        return fig
    
    def create_processing_steps_visualization(self, 
                                            original_image: np.ndarray,
                                            median_image: np.ndarray,
                                            cleaned_image: np.ndarray,
                                            material_mask: np.ndarray,
                                            histogram_data: Optional[Dict] = None,
                                            thresholds_or_centers: Optional[Union[List, np.ndarray]] = None,
                                            segmentation_method: str = 'unknown') -> plt.Figure:
        """
        Creates processing steps visualization.
        
        Args:
            original_image: Original input image
            median_image: Median filtered image
            cleaned_image: Artifact-removed image
            material_mask: Material mask
            histogram_data: Histogram analysis data
            thresholds_or_centers: Segmentation thresholds or cluster centers
            segmentation_method: Method used for segmentation
            
        Returns:
            Matplotlib figure object
        """
        fig, axes = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True)
        ax = axes.ravel()
        fig.suptitle("Phase Analysis - Processing Steps", fontsize=16)
        
        plot_idx = 0
        
        # 1. Original Image
        ax[plot_idx].imshow(original_image, cmap='gray')
        ax[plot_idx].set_title('Original Image')
        ax[plot_idx].axis('off')
        plot_idx += 1
        
        # 2. Median Filtered Image
        ax[plot_idx].imshow(median_image, cmap='gray')
        ax[plot_idx].set_title('Median Filtered')
        ax[plot_idx].axis('off')
        plot_idx += 1
        
        # 3. Material Mask
        ax[plot_idx].imshow(material_mask, cmap='gray')
        ax[plot_idx].set_title('Material Mask')
        ax[plot_idx].axis('off')
        plot_idx += 1
        
        # 4. Cleaned Image
        ax[plot_idx].imshow(cleaned_image, cmap='gray')
        ax[plot_idx].set_title('Cleaned Image (Artifacts Removed)')
        ax[plot_idx].axis('off')
        plot_idx += 1
        
        # 5. Histogram
        ax[plot_idx].set_title(f'Cleaned Material Histogram & Seg. ({segmentation_method})')
        ax[plot_idx].set_xlabel("Normalized Pixel Intensity")
        ax[plot_idx].set_ylabel("Frequency")
        ax[plot_idx].grid(True, linestyle=':')
        
        # Plot histogram if available
        if material_mask.any():
            cleaned_material_pixels = cleaned_image[material_mask]
            if cleaned_material_pixels.size > 0:
                hist_disp, bin_edges_disp = np.histogram(cleaned_material_pixels, bins=256, range=(0,1))
                bin_centers_disp = 0.5 * (bin_edges_disp[1:] + bin_edges_disp[:-1])
                ax[plot_idx].plot(bin_centers_disp, hist_disp, label='Cleaned Hist.', color='orange')
                
                # Add detected peaks if available
                if histogram_data and histogram_data.get('peaks') is not None:
                    peaks = histogram_data['peaks']
                    if len(peaks) > 0:
                        peak_x = histogram_data['bin_centers'][peaks]
                        peak_y = histogram_data['hist'][peaks]
                        ax[plot_idx].plot(peak_x, peak_y, "x", markersize=10, color='red', label='Detected Peaks')
                
                # Add thresholds or centers
                if thresholds_or_centers is not None and len(thresholds_or_centers) > 0:
                    if segmentation_method == 'kmeans':
                        for i, center in enumerate(thresholds_or_centers):
                            ax[plot_idx].axvline(center, color='purple', linestyle=':', 
                                               label=f'KMeans Center {i+1}: {center:.3f}')
                    else:
                        for i, thresh in enumerate(thresholds_or_centers):
                            ax[plot_idx].axvline(thresh, color='lime', linestyle='--', 
                                               label=f'Threshold {i+1}: {thresh:.3f}')
                
                ax[plot_idx].legend(fontsize='small')
        
        plot_idx += 1
        
        # Hide remaining subplot
        for i in range(plot_idx, len(ax)):
            ax[i].set_visible(False)
        
        return fig
    
    def run_full_analysis(self, 
                         image: np.ndarray,
                         preprocessing_params: Dict[str, Any] = None,
                         artifact_removal_params: Dict[str, Any] = None,
                         masking_params: Dict[str, Any] = None,
                         phase_detection_params: Dict[str, Any] = None,
                         segmentation_params: Dict[str, Any] = None,
                         visualization_params: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Runs the complete phase analysis pipeline.
        
        Args:
            image: Input grayscale image (already loaded and normalized)
            preprocessing_params: Parameters for preprocessing
            artifact_removal_params: Parameters for artifact removal
            masking_params: Parameters for mask creation
            phase_detection_params: Parameters for phase detection
            segmentation_params: Parameters for segmentation
            visualization_params: Parameters for visualization
            
        Returns:
            Dictionary containing all analysis results
        """
        # Set default parameters
        preprocessing_params = preprocessing_params or {}
        artifact_removal_params = artifact_removal_params or {}
        masking_params = masking_params or {}
        phase_detection_params = phase_detection_params or {}
        segmentation_params = segmentation_params or {}
        visualization_params = visualization_params or {}
        
        results = {
            'success': True,
            'error_message': None,
            'intermediate_images': {},
            'analysis_results': {},
            'phase_statistics': {},
            'visualizations': {}
        }
        
        try:
            # 1. Preprocessing
            processed_image = image.copy()
            
            if preprocessing_params.get('apply_illumination_correction', False):
                processed_image = correct_illumination(
                    processed_image,
                    preprocessing_params.get('illumination_method', 'blur_divide'),
                    preprocessing_params.get('illumination_kernel_size', 65)
                )
            
            if preprocessing_params.get('apply_denoising', False):
                processed_image = denoise_image(
                    processed_image,
                    preprocessing_params.get('denoise_method', 'gaussian'),
                    **preprocessing_params.get('denoise_params', {})
                )
            
            # 2. Create material mask
            material_mask, background_mask = self.create_material_mask(
                processed_image,
                masking_params.get('strategy', 'fill_holes'),
                masking_params.get('background_threshold', 0.08),
                masking_params.get('cleanup_area', 500)
            )
            
            # 4. Remove artifacts
            cleaned_image = self.remove_artifacts(
                processed_image, material_mask, background_mask,
                **artifact_removal_params
            )
            
            # 5. Phase detection (if enabled)
            num_phases = segmentation_params.get('num_phases', 3)
            histogram_data = None
            
            if phase_detection_params.get('auto_detect_phases', False):
                detected_phases, peak_intensities, histogram_data = self.detect_number_of_phases(
                    cleaned_image, material_mask, **phase_detection_params
                )
                num_phases = detected_phases
            
            # 6. Phase segmentation
            phase_masks, thresholds_or_centers = self.perform_phase_segmentation(
                cleaned_image, material_mask, num_phases,
                segmentation_params.get('method', 'auto'),
                segmentation_params.get('manual_thresholds'),
                segmentation_params.get('kmeans_random_state', 42),
                segmentation_params.get('kmeans_n_init', 'auto')
            )
            
            # 7. Calculate statistics
            phase_stats = self.calculate_phase_statistics(phase_masks, material_mask)
            
            # Store results
            results.update({
                'intermediate_images': self.intermediate_images,
                'analysis_results': {
                    'num_phases_detected': num_phases,
                    'num_phase_masks_created': len(phase_masks),
                    'total_material_pixels': int(np.sum(material_mask)),
                    'total_background_pixels': int(np.sum(background_mask)),
                    'segmentation_method': segmentation_params.get('method', 'auto'),
                    'thresholds_or_centers': thresholds_or_centers.tolist() if isinstance(thresholds_or_centers, np.ndarray) else thresholds_or_centers,
                    'histogram_data': histogram_data
                },
                'phase_statistics': phase_stats,
                'phase_masks': phase_masks,
                'material_mask': material_mask,
                'background_mask': background_mask,
                'cleaned_image': cleaned_image,
                'processed_image': processed_image,
                'original_image': image
            })
            
        except Exception as e:
            results['success'] = False
            results['error_message'] = str(e)
            import traceback
            traceback.print_exc()
        
        return results

# Convenience functions for easy integration
def analyze_phases(image: np.ndarray, **kwargs) -> Dict[str, Any]:
    """
    Convenience function for running phase analysis.
    
    Args:
        image: Input grayscale image [0,1]
        **kwargs: Analysis parameters organized by category
        
    Returns:
        Analysis results dictionary
    """
    analyzer = PhaseAnalyzer()
    return analyzer.run_full_analysis(image, **kwargs)

def get_default_phase_analysis_parameters() -> Dict[str, Dict[str, Any]]:
    """
    Returns default parameters for phase analysis.
    
    Returns:
        Dictionary of default parameter sets organized by category
    """
    return {
        'preprocessing': {
            'apply_illumination_correction': False,
            'illumination_method': 'blur_divide',
            'illumination_kernel_size': 65,
            'apply_denoising': False,
            'denoise_method': 'gaussian',
            'denoise_params': {'gaussian_sigma': 1.0},
            'initial_median_size': 1
        },
        'artifact_removal': {
            'dark_spot_fill_area': 50,
            'dark_spot_threshold_method': 'otsu',
            'bright_spot_method': 'opening',
            'bright_spot_opening_size': 15,
            'bright_spot_clip_offset': 0.05
        },
        'masking': {
            'strategy': 'fill_holes',
            'background_threshold': 0.08,
            'cleanup_area': 500
        },
        'phase_detection': {
            'auto_detect_phases': True,
            'histogram_bins': 256,
            'min_distance_bins': 5,
            'min_prominence_ratio': 0.05,
            'default_phases': 2
        },
        'segmentation': {
            'num_phases': 3,
            'method': 'kmeans',  # 'auto', 'otsu', 'multiotsu', 'percentile', 'kmeans', 'manual'
            'manual_thresholds': None,
            'kmeans_random_state': 42,
            'kmeans_n_init': 'auto'
        },
        'visualization': {
            'color_mode': 'palette',  # 'palette' or 'intensity'
            'palette_name': 'viridis',
            'phase_colors': ['red', 'lime', 'blue', 'yellow', 'cyan', 'pink']
        }
    }

def create_visualization_plots(analyzer: PhaseAnalyzer, 
                             results: Dict[str, Any]) -> Dict[str, plt.Figure]:
    """
    Creates matplotlib figures for visualization.
    Separate function to keep the main analysis clean.
    
    Args:
        analyzer: PhaseAnalyzer instance
        results: Analysis results dictionary
        
    Returns:
        Dictionary of matplotlib figures
    """
    figures = {}
    
    if not results['success']:
        return figures
    
    # Get required data
    original_image = results.get('original_image')
    cleaned_image = results.get('cleaned_image')
    median_image = results.get('median_image')
    phase_masks = results.get('phase_masks', [])
    material_mask = results.get('material_mask')
    background_mask = results.get('background_mask')
    
    if original_image is None or not phase_masks:
        return figures
    
    # Create segmentation visualization
    figures['segmentation'] = analyzer.create_segmentation_visualization(
        original_image, cleaned_image, phase_masks, background_mask,
        color_mode='palette', palette_name='viridis'
    )
    
    # Create processing steps visualization
    if median_image is not None and material_mask is not None:
        histogram_data = results['analysis_results'].get('histogram_data')
        thresholds_or_centers = results['analysis_results'].get('thresholds_or_centers')
        segmentation_method = results['analysis_results'].get('segmentation_method', 'unknown')
        
        figures['processing_steps'] = analyzer.create_processing_steps_visualization(
            original_image, median_image, cleaned_image, material_mask,
            histogram_data, thresholds_or_centers, segmentation_method
        )
    
    return figures