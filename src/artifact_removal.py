import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Optional, Dict, Any, List, Union
from PIL import Image

# LaMa inpainting is disabled to avoid PyTorch warnings
LAMA_AVAILABLE = False
SimpleLama = None

from .utils import make_odd

class ArtifactRemover:
    """
    A utility class for performing artifact removal on microscopy images.
    Pure Python implementation without any GUI dependencies.
    Follows the same OOP pattern as PhaseAnalyzer and LineAnalyzer.
    """
    
    def __init__(self):
        """Initialize the artifact remover."""
        self.intermediate_images = {}
    
    def create_artifact_mask(self, 
                           image: np.ndarray, 
                           spot_type: str = "bright",
                           detection_method: str = "percentile",
                           threshold_percentile: float = 95.0,
                           absolute_threshold: int = 200) -> np.ndarray:
        """
        Create initial mask for artifact detection.
        
        Args:
            image: Input grayscale image (uint8)
            spot_type: "bright" or "dark" spots to detect
            detection_method: "otsu", "percentile", or "absolute"
            threshold_percentile: Percentile threshold (0-100)
            absolute_threshold: Absolute intensity threshold (0-255)
        
        Returns:
            Binary mask of detected artifacts
        """
        if detection_method == "otsu":
            # Automatic thresholding using Otsu's method
            if spot_type == "bright":
                _, mask = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            else:
                _, mask = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            return mask.astype(bool)
        
        elif detection_method == "percentile":
            # Statistical thresholding based on intensity distribution
            threshold = np.percentile(image, threshold_percentile)
            if spot_type == "bright":
                return image > threshold
            else:
                return image < threshold
        
        elif detection_method == "absolute":
            # Fixed intensity threshold
            if spot_type == "bright":
                return image > absolute_threshold
            else:
                return image < absolute_threshold
        
        else:
            raise ValueError(f"Unknown detection method: {detection_method}")
    
    def clean_mask(self, 
                   mask: np.ndarray,
                   apply_opening: bool = True,
                   opening_size: int = 3,
                   opening_shape: str = "ellipse",
                   min_area: int = 10,
                   max_area: int = 5000) -> np.ndarray:
        """
        Clean up the artifact mask by removing noise and filtering components.
        
        Args:
            mask: Input binary mask
            apply_opening: Whether to apply morphological opening
            opening_size: Size of opening kernel
            opening_shape: Shape of kernel ("ellipse", "rectangle", "cross")
            min_area: Minimum component area to keep
            max_area: Maximum component area to keep
        
        Returns:
            Cleaned binary mask
        """
        if not np.any(mask):
            return mask
        
        cleaned_mask = mask.copy()
        
        # Apply morphological opening to remove small noise
        if apply_opening and opening_size > 0:
            opening_size = make_odd(opening_size)
            
            if opening_shape == "ellipse":
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (opening_size, opening_size))
            elif opening_shape == "rectangle":
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (opening_size, opening_size))
            elif opening_shape == "cross":
                kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (opening_size, opening_size))
            else:
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (opening_size, opening_size))
            
            cleaned_mask = cv2.morphologyEx(cleaned_mask.astype(np.uint8), cv2.MORPH_OPEN, kernel).astype(bool)
        
        # Filter components by size
        if min_area > 0 or max_area < float('inf'):
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(cleaned_mask.astype(np.uint8))
            
            filtered_mask = np.zeros_like(cleaned_mask, dtype=bool)
            
            for i in range(1, num_labels):  # Skip background (label 0)
                area = stats[i, cv2.CC_STAT_AREA]
                if min_area <= area <= max_area:
                    component_mask = (labels == i)
                    filtered_mask[component_mask] = True
            
            cleaned_mask = filtered_mask
        
        self.intermediate_images['cleaned_mask'] = cleaned_mask
        return cleaned_mask
    
    def expand_mask(self, 
                    mask: np.ndarray,
                    dilation_size: int = 2,
                    dilation_shape: str = "ellipse") -> np.ndarray:
        """
        Expand the mask slightly for better inpainting coverage.
        
        Args:
            mask: Input binary mask
            dilation_size: Size of dilation kernel
            dilation_shape: Shape of kernel ("ellipse", "rectangle", "cross")
        
        Returns:
            Dilated binary mask
        """
        if not np.any(mask) or dilation_size <= 0:
            return mask
        
        dilation_size = make_odd(dilation_size)
        
        if dilation_shape == "ellipse":
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilation_size, dilation_size))
        elif dilation_shape == "rectangle":
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (dilation_size, dilation_size))
        elif dilation_shape == "cross":
            kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (dilation_size, dilation_size))
        else:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilation_size, dilation_size))
        
        expanded_mask = cv2.dilate(mask.astype(np.uint8), kernel).astype(bool)
        self.intermediate_images['expanded_mask'] = expanded_mask
        return expanded_mask
    
    def inpaint_artifacts(self, 
                         image: np.ndarray,
                         mask: np.ndarray,
                         method: str = "telea",
                         radius: int = 3) -> Tuple[np.ndarray, bool]:
        """
        Inpaint the masked regions using the specified method.
        
        Args:
            image: Input grayscale image (uint8)
            mask: Binary mask of regions to inpaint
            method: Inpainting method ("telea", "ns") - LaMa removed to avoid PyTorch warnings
            radius: Inpainting radius (for telea and ns methods)
        
        Returns:
            Tuple of (inpainted_image, success_flag)
        """
        if not np.any(mask):
            return image, False
        
        try:
            if method == "lama":
                # LaMa is disabled - fallback to telea
                method = "telea"
            
            if method == "telea":
                # Fast marching inpainting
                result = cv2.inpaint(image, mask.astype(np.uint8), radius, cv2.INPAINT_TELEA)
                self.intermediate_images['inpainted_image'] = result
                return result, True
            
            elif method == "ns":
                # Navier-Stokes inpainting
                result = cv2.inpaint(image, mask.astype(np.uint8), radius, cv2.INPAINT_NS)
                self.intermediate_images['inpainted_image'] = result
                return result, True
            
            else:
                # Default to telea
                result = cv2.inpaint(image, mask.astype(np.uint8), radius, cv2.INPAINT_TELEA)
                self.intermediate_images['inpainted_image'] = result
                return result, True
        
        except Exception as e:
            # If any method fails, return original image
            return image, False
    
    def calculate_artifact_statistics(self, 
                                    initial_mask: np.ndarray,
                                    final_mask: np.ndarray,
                                    image_shape: Tuple[int, int]) -> Dict[str, Any]:
        """
        Calculate statistics about detected and processed artifacts.
        
        Args:
            initial_mask: Initial detected artifact mask
            final_mask: Final processed artifact mask
            image_shape: Shape of the image
        
        Returns:
            Dictionary of artifact statistics
        """
        total_pixels = image_shape[0] * image_shape[1]
        initial_artifacts = np.sum(initial_mask)
        final_artifacts = np.sum(final_mask)
        
        # Calculate connected components
        num_labels_initial, _, stats_initial, _ = cv2.connectedComponentsWithStats(initial_mask.astype(np.uint8))
        num_labels_final, _, stats_final, _ = cv2.connectedComponentsWithStats(final_mask.astype(np.uint8))
        
        # Component sizes (excluding background)
        initial_sizes = stats_initial[1:, cv2.CC_STAT_AREA] if num_labels_initial > 1 else []
        final_sizes = stats_final[1:, cv2.CC_STAT_AREA] if num_labels_final > 1 else []
        
        stats = {
            'total_pixels': int(total_pixels),
            'initial_artifacts': {
                'pixels': int(initial_artifacts),
                'coverage_percent': float(initial_artifacts / total_pixels * 100),
                'num_components': int(num_labels_initial - 1),  # Exclude background
                'mean_component_size': float(np.mean(initial_sizes)) if len(initial_sizes) > 0 else 0.0,
                'max_component_size': int(np.max(initial_sizes)) if len(initial_sizes) > 0 else 0,
                'min_component_size': int(np.min(initial_sizes)) if len(initial_sizes) > 0 else 0
            },
            'final_artifacts': {
                'pixels': int(final_artifacts),
                'coverage_percent': float(final_artifacts / total_pixels * 100),
                'num_components': int(num_labels_final - 1),  # Exclude background
                'mean_component_size': float(np.mean(final_sizes)) if len(final_sizes) > 0 else 0.0,
                'max_component_size': int(np.max(final_sizes)) if len(final_sizes) > 0 else 0,
                'min_component_size': int(np.min(final_sizes)) if len(final_sizes) > 0 else 0
            },
            'filtered_out': {
                'pixels': int(initial_artifacts - final_artifacts),
                'components': int((num_labels_initial - 1) - (num_labels_final - 1))
            }
        }
        
        return stats
    
    def create_visualization_plots(self, 
                                 original_image: np.ndarray,
                                 processed_image: np.ndarray,
                                 initial_mask: np.ndarray,
                                 final_mask: np.ndarray,
                                 artifacts_found: bool) -> plt.Figure:
        """
        Create visualization plots for artifact removal results.
        
        Args:
            original_image: Original input image
            processed_image: Processed output image
            initial_mask: Initial detection mask
            final_mask: Final processing mask
            artifacts_found: Whether artifacts were found and processed
        
        Returns:
            Matplotlib figure with visualization
        """
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        # 1. Original Image
        axes[0].imshow(original_image, cmap='gray')
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        # 2. Initial Detection Mask
        axes[1].imshow(initial_mask, cmap='red_mask', alpha=0.7)
        axes[1].imshow(original_image, cmap='gray', alpha=0.3)
        axes[1].set_title('Initial Detection')
        axes[1].axis('off')
        
        # 3. Final Processing Mask
        axes[2].imshow(final_mask, cmap='red_mask', alpha=0.7)
        axes[2].imshow(original_image, cmap='gray', alpha=0.3)
        axes[2].set_title('Final Mask')
        axes[2].axis('off')
        
        # 4. Processed Image
        axes[3].imshow(processed_image, cmap='gray')
        axes[3].set_title('Processed Image')
        axes[3].axis('off')
        
        # 5. Difference Image
        if artifacts_found:
            diff_image = np.abs(processed_image.astype(float) - original_image.astype(float))
            axes[4].imshow(diff_image, cmap='hot')
            axes[4].set_title('Difference (Enhanced)')
        else:
            axes[4].text(0.5, 0.5, 'No artifacts\nprocessed', 
                        ha='center', va='center', transform=axes[4].transAxes,
                        fontsize=14, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
            axes[4].set_title('Difference')
        axes[4].axis('off')
        
        # 6. Overlay Comparison
        # Create overlay: original in blue, processed in red
        overlay = np.zeros((*original_image.shape, 3), dtype=float)
        overlay[:, :, 0] = processed_image / 255.0  # Red channel
        overlay[:, :, 2] = original_image / 255.0   # Blue channel
        overlay[:, :, 1] = np.minimum(processed_image, original_image) / 255.0  # Green where they match
        
        axes[5].imshow(overlay)
        axes[5].set_title('Overlay (Red: Processed, Blue: Original)')
        axes[5].axis('off')
        
        plt.tight_layout()
        fig.suptitle("Artifact Removal Analysis", fontsize=16, y=0.98)
        
        return fig
    
    def run_full_analysis(self, 
                         image: np.ndarray,
                         spot_type: str = "bright",
                         detection_params: Dict[str, Any] = None,
                         filtering_params: Dict[str, Any] = None,
                         inpainting_params: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Run the complete artifact removal pipeline.
        
        Args:
            image: Input grayscale image
            spot_type: "bright" or "dark" artifacts to remove
            detection_params: Parameters for artifact detection
            filtering_params: Parameters for mask filtering
            inpainting_params: Parameters for inpainting
        
        Returns:
            Dictionary containing all analysis results
        """
        # Set default parameters
        detection_params = detection_params or {}
        filtering_params = filtering_params or {}
        inpainting_params = inpainting_params or {}
        
        # Clear intermediate images
        self.intermediate_images = {}
        
        results = {
            'success': True,
            'error_message': None,
            'artifacts_found': False,
            'artifacts_processed': False,
            'intermediate_images': {},
            'analysis_results': {},
            'artifact_statistics': {},
            'visualizations': {}
        }
        
        try:
            # Ensure image is uint8
            if image.dtype != np.uint8:
                if image.max() <= 1.0:
                    image_uint8 = (image * 255).astype(np.uint8)
                else:
                    image_uint8 = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            else:
                image_uint8 = image.copy()
            
            self.intermediate_images['original_image'] = image_uint8
            
            # 1. Create initial artifact mask
            initial_mask = self.create_artifact_mask(
                image_uint8, 
                spot_type,
                detection_params.get('method', 'percentile'),
                detection_params.get('threshold_percentile', 95.0 if spot_type == 'bright' else 5.0),
                detection_params.get('absolute_threshold', 200 if spot_type == 'bright' else 50)
            )
            
            self.intermediate_images['initial_mask'] = initial_mask
            artifacts_detected = np.any(initial_mask)
            
            # 2. Clean up the mask
            cleaned_mask = self.clean_mask(
                initial_mask,
                filtering_params.get('apply_opening', True),
                filtering_params.get('opening_size', 3),
                filtering_params.get('opening_shape', 'ellipse'),
                filtering_params.get('min_area', 10),
                filtering_params.get('max_area', 5000)
            )
            
            # 3. Expand mask for better inpainting
            final_mask = self.expand_mask(
                cleaned_mask,
                filtering_params.get('dilation_size', 2),
                filtering_params.get('dilation_shape', 'ellipse')
            )
            
            self.intermediate_images['final_mask'] = final_mask
            artifacts_to_process = np.any(final_mask)
            
            # 4. Inpaint the artifacts
            processed_image = image_uint8.copy()
            inpainting_success = False
            
            if artifacts_to_process:
                processed_image, inpainting_success = self.inpaint_artifacts(
                    image_uint8,
                    final_mask,
                    inpainting_params.get('method', 'telea'),
                    inpainting_params.get('radius', 3)
                )
            
            # 5. Calculate statistics
            artifact_stats = self.calculate_artifact_statistics(
                initial_mask, final_mask, image_uint8.shape
            )
            
            # 6. Update results
            results.update({
                'artifacts_found': artifacts_detected,
                'artifacts_processed': artifacts_to_process and inpainting_success,
                'intermediate_images': self.intermediate_images,
                'processed_image': processed_image,
                'initial_mask': initial_mask,
                'final_mask': final_mask,
                'analysis_results': {
                    'spot_type': spot_type,
                    'detection_method': detection_params.get('method', 'percentile'),
                    'inpainting_method': inpainting_params.get('method', 'telea'),
                    'artifacts_detected': artifacts_detected,
                    'artifacts_processed': artifacts_to_process and inpainting_success,
                    'num_initial_components': artifact_stats['initial_artifacts']['num_components'],
                    'num_final_components': artifact_stats['final_artifacts']['num_components'],
                    'total_pixels_processed': artifact_stats['final_artifacts']['pixels'],
                    'coverage_percent': artifact_stats['final_artifacts']['coverage_percent']
                },
                'artifact_statistics': artifact_stats
            })
            
        except Exception as e:
            results['success'] = False
            results['error_message'] = str(e)
            import traceback
            traceback.print_exc()
        
        return results

# Convenience functions for easy integration (following the same pattern as phase/line analysis)
def remove_artifacts(image: np.ndarray, **kwargs) -> Dict[str, Any]:
    """
    Convenience function for artifact removal analysis.
    
    Args:
        image: Input image
        **kwargs: Parameters for artifact removal
    
    Returns:
        Dictionary containing analysis results
    """
    analyzer = ArtifactRemover()
    return analyzer.run_full_analysis(image, **kwargs)

def get_default_parameters() -> Dict[str, Dict[str, Any]]:
    """
    Get default parameters for artifact removal analysis.
    
    Returns:
        Dictionary of default parameters organized by category
    """
    return {
        'detection_params': {
            'method': 'percentile',
            'threshold_percentile': 95.0,
            'absolute_threshold': 200
        },
        'filtering_params': {
            'apply_opening': True,
            'opening_size': 3,
            'opening_shape': 'ellipse',
            'min_area': 10,
            'max_area': 5000,
            'dilation_size': 2,
            'dilation_shape': 'ellipse'
        },
        'inpainting_params': {
            'method': 'telea',
            'radius': 3
        }
    }

def create_visualization_plots(analyzer: ArtifactRemover, 
                             results: Dict[str, Any]) -> Dict[str, plt.Figure]:
    """
    Create visualization plots for artifact removal results.
    
    Args:
        analyzer: ArtifactRemover instance with intermediate results
        results: Results dictionary from analysis
    
    Returns:
        Dictionary of matplotlib figures
    """
    figures = {}
    
    if results['success'] and 'processed_image' in results:
        try:
            original_image = analyzer.intermediate_images.get('original_image')
            processed_image = results.get('processed_image')
            initial_mask = results.get('initial_mask')
            final_mask = results.get('final_mask')
            artifacts_found = results.get('artifacts_processed', False)
            
            if all(x is not None for x in [original_image, processed_image, initial_mask, final_mask]):
                figures['artifact_removal_analysis'] = analyzer.create_visualization_plots(
                    original_image, processed_image, initial_mask, final_mask, artifacts_found
                )
        except Exception as e:
            print(f"Warning: Could not create artifact removal visualizations: {e}")
    
    return figures

# Legacy compatibility functions
def remove_bright_spots(image: np.ndarray, **kwargs) -> Tuple[np.ndarray, bool, np.ndarray]:
    """Legacy function for removing bright spots."""
    results = remove_artifacts(image, spot_type="bright", **kwargs)
    return results.get('processed_image', image), results.get('artifacts_processed', False), results.get('final_mask', np.zeros_like(image, dtype=bool))

def remove_dark_spots(image: np.ndarray, **kwargs) -> Tuple[np.ndarray, bool, np.ndarray]:
    """Legacy function for removing dark spots."""
    results = remove_artifacts(image, spot_type="dark", **kwargs)
    return results.get('processed_image', image), results.get('artifacts_processed', False), results.get('final_mask', np.zeros_like(image, dtype=bool))

def remove_spots(image: np.ndarray, **kwargs) -> Tuple[np.ndarray, bool, np.ndarray]:
    """Legacy function name for compatibility."""
    results = remove_artifacts(image, **kwargs)
    return results.get('processed_image', image), results.get('artifacts_processed', False), results.get('final_mask', np.zeros_like(image, dtype=bool))