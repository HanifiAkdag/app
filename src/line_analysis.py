import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, hsv_to_rgb
from skimage import exposure, filters, transform
from skimage.morphology import disk, skeletonize, binary_erosion
from typing import Dict, List, Tuple, Optional, Any

from .preprocessing import correct_illumination, denoise_image

class LineAnalyzer:
    """
    A utility class for performing line/orientation analysis on microscopy images.
    Pure Python implementation without any GUI dependencies.
    """
    
    def __init__(self):
        self.results = {}
        self.intermediate_images = {}
        
    def apply_frangi_filter(self, 
                           image: np.ndarray, 
                           sigmas: range = range(1, 8, 2), 
                           black_ridges: bool = False) -> np.ndarray:
        """
        Applies the Frangi filter to enhance vessel-like structures.
        
        Args:
            image: Input grayscale image
            sigmas: Range of scales for feature detection
            black_ridges: True for dark lines on bright background
            
        Returns:
            Enhanced ridge image
        """
        ridge_image = filters.frangi(image, sigmas=sigmas, black_ridges=black_ridges, mode='reflect')
        self.intermediate_images['frangi_output'] = ridge_image
        return ridge_image
    
    def threshold_image_otsu(self, 
                            image: np.ndarray, 
                            epsilon: float = 1e-10) -> Tuple[np.ndarray, float]:
        """
        Applies Otsu's thresholding to create binary image.
        
        Args:
            image: Input grayscale image
            epsilon: Small value for numerical stability
            
        Returns:
            Tuple of (binary_image, threshold_value)
        """
        try:
            threshold_value = filters.threshold_otsu(image + epsilon)
            binary_image = image > threshold_value
            self.intermediate_images['binary_ridges'] = binary_image
            return binary_image, threshold_value
        except ValueError:
            # Fallback for uniform images
            binary_image = np.zeros_like(image, dtype=bool)
            return binary_image, 0
    
    def apply_skeletonization(self, 
                             binary_image: np.ndarray, 
                             apply_skeleton: bool = True) -> Optional[np.ndarray]:
        """
        Applies skeletonization to reduce thick ridges to single-pixel lines.
        
        Args:
            binary_image: Input binary image
            apply_skeleton: Whether to apply skeletonization
            
        Returns:
            Skeletonized image or None if not applied
        """
        if apply_skeleton and binary_image.any():
            skeleton = skeletonize(binary_image)
            self.intermediate_images['skeleton'] = skeleton
            return skeleton
        return None
    
    def exclude_boundary_region(self, 
                               image: Optional[np.ndarray], 
                               material_mask: np.ndarray,
                               erosion_size: int = 10) -> Tuple[Optional[np.ndarray], np.ndarray]:
        """
        Excludes pixels near the material boundary and creates boundary zone mask.
        
        Args:
            image: Image to modify (can be None to just get boundary mask)
            material_mask: Binary mask of material region
            erosion_size: Size of erosion for boundary exclusion
            
        Returns:
            Tuple of (modified_image, boundary_zone_mask)
        """
        if erosion_size > 0 and material_mask.any():
            eroded_mask = binary_erosion(material_mask, disk(erosion_size))
            boundary_zone_mask = material_mask & (~eroded_mask)
            
            if image is not None:
                modified_image = image.copy()
                modified_image[boundary_zone_mask] = False
                self.intermediate_images['boundary_excluded'] = modified_image
                return modified_image, boundary_zone_mask
            else:
                return None, boundary_zone_mask
        else:
            boundary_zone_mask = np.zeros_like(material_mask, dtype=bool)
            return image, boundary_zone_mask
    
    def detect_hough_lines(self, 
                          binary_image: np.ndarray,
                          threshold: int = 5,
                          min_length: int = 20,
                          max_gap: int = 10) -> List[Tuple]:
        """
        Detects line segments using Probabilistic Hough Transform.
        
        Args:
            binary_image: Input binary image
            threshold: Minimum votes for line detection
            min_length: Minimum line segment length
            max_gap: Maximum gap between line segments
            
        Returns:
            List of detected line segments
        """
        if binary_image is None or not binary_image.any():
            return []
        
        lines = transform.probabilistic_hough_line(
            binary_image,
            threshold=threshold,
            line_length=min_length,
            line_gap=max_gap
        )
        return lines
    
    def calculate_line_orientations(self, lines: List[Tuple]) -> List[float]:
        """
        Calculates orientation angles for detected line segments.
        
        Args:
            lines: List of line segments from Hough transform
            
        Returns:
            List of orientation angles in degrees (0-180)
        """
        angles_deg = []
        for line in lines:
            p0, p1 = line
            angle_rad = np.arctan2(p1[1] - p0[1], p1[0] - p0[0])
            angle_deg = np.rad2deg(angle_rad) % 180
            angles_deg.append(angle_deg)
        return angles_deg
    
    def calculate_sobel_orientation(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculates edge orientation using Sobel filters.
        
        Args:
            image: Input grayscale image
            
        Returns:
            Tuple of (edge_angles_deg, magnitude_normalized, magnitude_raw)
        """
        # Calculate Sobel gradients
        gy = filters.sobel_h(image)  # Horizontal edges
        gx = filters.sobel_v(image)  # Vertical edges
        
        magnitude = np.sqrt(gx**2 + gy**2)
        gradient_angle_rad = np.arctan2(gy, gx)
        
        # Edge orientation is perpendicular to gradient
        edge_angle_rad = gradient_angle_rad + np.pi / 2
        edge_angle_deg = (np.rad2deg(edge_angle_rad) + 180) % 180
        
        # Normalize magnitude for visualization
        magnitude_norm = exposure.rescale_intensity(magnitude, out_range=(0, 1))
        
        self.intermediate_images['sobel_orientation'] = edge_angle_deg
        self.intermediate_images['sobel_magnitude'] = magnitude_norm
        
        return edge_angle_deg, magnitude_norm, magnitude
    
    def analyze_hough_orientations(self, 
                                  angles_deg: List[float], 
                                  bins: int = 60) -> Tuple[Optional[float], np.ndarray, np.ndarray]:
        """
        Analyzes Hough line orientations and finds dominant angle.
        
        Args:
            angles_deg: List of line orientation angles
            bins: Number of histogram bins
            
        Returns:
            Tuple of (dominant_angle, histogram_values, bin_centers)
        """
        if not angles_deg:
            return None, np.array([]), np.array([])
        
        hist, bin_edges = np.histogram(angles_deg, bins=bins, range=(0, 180))
        bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
        
        dominant_angle = None
        if hist.size > 0 and np.sum(hist) > 0:
            peak_bin_index = np.argmax(hist)
            dominant_angle = bin_centers[peak_bin_index]
        
        return dominant_angle, hist, bin_centers
    
    def analyze_sobel_orientations(self, 
                                  angles_deg: np.ndarray,
                                  magnitude: np.ndarray,
                                  material_mask: np.ndarray,
                                  boundary_zone_mask: np.ndarray,
                                  magnitude_threshold: float = 0.01,
                                  bins: int = 90) -> Tuple[Optional[float], np.ndarray, np.ndarray]:
        """
        Analyzes Sobel edge orientations with magnitude weighting.
        
        Args:
            angles_deg: 2D array of edge angles
            magnitude: 2D array of gradient magnitudes
            material_mask: Binary mask of material region
            boundary_zone_mask: Binary mask of boundary region to exclude
            magnitude_threshold: Minimum magnitude threshold
            bins: Number of histogram bins
            
        Returns:
            Tuple of (dominant_angle, histogram_values, bin_centers)
        """
        # Flatten arrays and apply filters
        angles_flat = angles_deg.flatten()
        magnitude_flat = magnitude.flatten()
        mask_flat = material_mask.flatten()
        boundary_flat = boundary_zone_mask.flatten()
        
        # Filter valid pixels
        valid_indices = (mask_flat) & (~boundary_flat) & (magnitude_flat > magnitude_threshold)
        
        if not np.any(valid_indices):
            return None, np.array([]), np.array([])
        
        valid_angles = angles_flat[valid_indices]
        valid_magnitudes = magnitude_flat[valid_indices]
        
        # Calculate weighted histogram
        hist, bin_edges = np.histogram(
            valid_angles, 
            bins=bins, 
            range=(0, 180), 
            weights=valid_magnitudes
        )
        bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
        
        dominant_angle = None
        if hist.size > 0 and np.sum(hist) > 0:
            peak_bin_index = np.argmax(hist)
            dominant_angle = bin_centers[peak_bin_index]
        
        return dominant_angle, hist, bin_centers
    
    def create_histogram_plot(self, 
                             hist_values: np.ndarray, 
                             bin_centers: np.ndarray,
                             dominant_angle: Optional[float] = None,
                             title: str = "Orientation Histogram",
                             color: str = 'blue') -> plt.Figure:
        """
        Creates a histogram plot and returns matplotlib figure.
        
        Args:
            hist_values: Histogram values
            bin_centers: Bin center positions
            dominant_angle: Peak angle to highlight
            title: Plot title
            color: Bar color
            
        Returns:
            Matplotlib figure object
        """
        fig, ax = plt.subplots(figsize=(8, 5))
        
        width = 180 / len(bin_centers) * 0.9 if len(bin_centers) > 0 else 1
        ax.bar(bin_centers, hist_values, width=width, color=color, alpha=0.7)
        ax.set_title(title)
        ax.set_xlabel('Angle (Degrees)')
        ax.set_ylabel('Frequency')
        ax.set_xticks(np.arange(0, 181, 30))
        ax.grid(axis='y', linestyle='--', alpha=0.3)
        
        if dominant_angle is not None:
            ax.axvline(dominant_angle, color='red', linestyle='--', linewidth=2, 
                      label=f'Peak: {dominant_angle:.1f}°')
            ax.legend()
        
        plt.tight_layout()
        return fig
    
    def create_visualization_overlay(self, 
                                   image: np.ndarray,
                                   lines: List[Tuple] = None,
                                   colormap: str = 'hsv') -> plt.Figure:
        """
        Creates line overlay visualization.
        
        Args:
            image: Base grayscale image
            lines: Detected line segments
            colormap: Colormap name
            
        Returns:
            Matplotlib figure object
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.imshow(image, cmap='gray')
        
        if lines:
            cmap = plt.get_cmap(colormap)
            angles = self.calculate_line_orientations(lines)
            norm = Normalize(vmin=0, vmax=180)
            
            for line, angle in zip(lines, angles):
                p0, p1 = line
                color = cmap(norm(angle))
                ax.plot([p0[0], p1[0]], [p0[1], p1[1]], color=color, linewidth=2, alpha=0.7)
        
        ax.set_title(f'Detected Lines ({len(lines) if lines else 0} lines)')
        ax.axis('off')
        plt.tight_layout()
        return fig
    
    def create_sobel_visualization(self, 
                                 sobel_angles: np.ndarray,
                                 sobel_magnitude: np.ndarray,
                                 boundary_mask: np.ndarray = None,
                                 colormap: str = 'hsv') -> plt.Figure:
        """
        Creates Sobel orientation visualization.
        
        Args:
            sobel_angles: 2D array of edge angles
            sobel_magnitude: 2D array of magnitude values
            boundary_mask: Optional boundary mask to exclude
            colormap: Colormap name
            
        Returns:
            Matplotlib figure object
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        
        cmap = plt.get_cmap(colormap)
        norm = Normalize(vmin=0, vmax=180)
        
        hue = norm(sobel_angles)
        saturation = np.ones_like(hue)
        value = sobel_magnitude
        
        sobel_hsv = np.stack((hue, saturation, value), axis=-1)
        sobel_rgb = hsv_to_rgb(sobel_hsv)
        
        # Apply boundary exclusion
        if boundary_mask is not None:
            boundary_mask_rgb = np.stack([boundary_mask]*3, axis=-1)
            sobel_rgb[boundary_mask_rgb] = 0
        
        ax.imshow(sobel_rgb)
        ax.set_title('Sobel Edge Orientation (Color = Angle, Brightness = Magnitude)')
        ax.axis('off')
        plt.tight_layout()
        return fig
    
    def run_full_analysis(self, 
                         image: np.ndarray,
                         material_mask: np.ndarray,
                         preprocessing_params: Dict[str, Any] = None,
                         frangi_params: Dict[str, Any] = None,
                         hough_params: Dict[str, Any] = None,
                         analysis_params: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Runs the complete line orientation analysis pipeline.
        
        Args:
            image: Input grayscale image
            material_mask: Binary mask of material region
            preprocessing_params: Parameters for preprocessing
            frangi_params: Parameters for Frangi filtering
            hough_params: Parameters for Hough transform
            analysis_params: Parameters for analysis
            
        Returns:
            Dictionary containing all analysis results
        """
        # Set default parameters
        preprocessing_params = preprocessing_params or {}
        frangi_params = frangi_params or {}
        hough_params = hough_params or {}
        analysis_params = analysis_params or {}
        
        results = {
            'success': True,
            'error_message': None,
            'intermediate_images': {},
            'analysis_results': {},
            'visualizations': {},
            'histograms': {}
        }
        
        try:
            # 1. Preprocessing (optional)
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
            
            # 2. Frangi filtering
            ridge_image = self.apply_frangi_filter(
                processed_image,
                frangi_params.get('sigmas', range(1, 8, 2)),
                frangi_params.get('black_ridges', False)
            )
            
            # 3. Thresholding
            binary_image, threshold_value = self.threshold_image_otsu(ridge_image)
            
            # 4. Optional skeletonization
            skeleton = None
            if analysis_params.get('apply_skeletonization', True):
                skeleton = self.apply_skeletonization(binary_image)
            
            # 5. Boundary exclusion
            hough_input = skeleton if skeleton is not None else binary_image
            boundary_excluded_image, boundary_mask = self.exclude_boundary_region(
                hough_input.copy() if hough_input is not None else None,
                material_mask,
                analysis_params.get('boundary_erosion_size', 10)
            )
            
            # 6. Hough line detection
            lines = self.detect_hough_lines(
                boundary_excluded_image if boundary_excluded_image is not None else hough_input,
                hough_params.get('threshold', 5),
                hough_params.get('min_length', 20),
                hough_params.get('max_gap', 10)
            )
            
            hough_angles = self.calculate_line_orientations(lines)
            
            # 7. Sobel analysis (optional)
            sobel_angles, sobel_magnitude_norm, sobel_magnitude_raw = None, None, None
            if analysis_params.get('analyze_sobel', True):
                sobel_angles, sobel_magnitude_norm, sobel_magnitude_raw = self.calculate_sobel_orientation(processed_image)
            
            # 8. Orientation analysis
            dominant_hough, hough_hist, hough_bins = self.analyze_hough_orientations(
                hough_angles, analysis_params.get('hough_histogram_bins', 60)
            )
            
            dominant_sobel, sobel_hist, sobel_bins = None, None, None
            if sobel_angles is not None:
                dominant_sobel, sobel_hist, sobel_bins = self.analyze_sobel_orientations(
                    sobel_angles, sobel_magnitude_raw, material_mask, boundary_mask,
                    analysis_params.get('sobel_magnitude_threshold', 0.01),
                    analysis_params.get('sobel_histogram_bins', 90)
                )
            
            # 9. Store visualization data (without creating plots)
            visualizations = {}
            if lines and hough_angles:
                visualizations['hough_lines'] = {
                    'base_image': processed_image,
                    'lines': lines,
                    'angles': hough_angles,
                    'colormap': analysis_params.get('colormap', 'hsv')
                }
            
            if sobel_angles is not None and sobel_magnitude_norm is not None:
                cmap = plt.get_cmap(analysis_params.get('colormap', 'hsv'))
                norm = Normalize(vmin=0, vmax=180)
                
                hue = norm(sobel_angles)
                saturation = np.ones_like(hue)
                value = sobel_magnitude_norm
                
                sobel_hsv = np.stack((hue, saturation, value), axis=-1)
                sobel_rgb = hsv_to_rgb(sobel_hsv)
                
                # Apply boundary exclusion
                if boundary_mask is not None:
                    boundary_mask_rgb = np.stack([boundary_mask]*3, axis=-1)
                    sobel_rgb[boundary_mask_rgb] = 0
                
                visualizations['sobel_orientation'] = sobel_rgb
            
            # 10. Store histogram data
            histograms = {}
            if len(hough_hist) > 0:
                histograms['hough'] = {
                    'hist': hough_hist,
                    'bins': hough_bins,
                    'dominant': dominant_hough
                }
            
            if sobel_hist is not None and len(sobel_hist) > 0:
                histograms['sobel'] = {
                    'hist': sobel_hist,
                    'bins': sobel_bins,
                    'dominant': dominant_sobel
                }
            
            # Compile results
            results.update({
                'intermediate_images': self.intermediate_images,
                'analysis_results': {
                    'num_lines_detected': len(lines),
                    'dominant_hough_angle': dominant_hough,
                    'dominant_sobel_angle': dominant_sobel,
                    'threshold_value': threshold_value,
                    'hough_angles': hough_angles,
                    'boundary_excluded_pixels': np.sum(boundary_mask) if boundary_mask is not None else 0
                },
                'visualizations': visualizations,
                'histograms': histograms
            })
            
        except Exception as e:
            results['success'] = False
            results['error_message'] = str(e)
            import traceback
            traceback.print_exc()
        
        return results

# Convenience functions for easy integration
def analyze_line_orientations(image: np.ndarray, 
                            material_mask: np.ndarray,
                            **kwargs) -> Dict[str, Any]:
    """
    Convenience function for running line orientation analysis.
    
    Args:
        image: Input grayscale image
        material_mask: Binary mask of material region
        **kwargs: Additional parameters organized by category
        
    Returns:
        Analysis results dictionary
    """
    analyzer = LineAnalyzer()
    return analyzer.run_full_analysis(image, material_mask, **kwargs)

def get_default_parameters() -> Dict[str, Dict[str, Any]]:
    """
    Returns default parameters for line analysis.
    
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
            'denoise_params': {'gaussian_sigma': 1.0}
        },
        'frangi': {
            'sigmas': range(1, 8, 2),
            'black_ridges': False
        },
        'hough': {
            'threshold': 5,
            'min_length': 20,
            'max_gap': 10
        },
        'analysis': {
            'apply_skeletonization': True,
            'boundary_erosion_size': 10,
            'analyze_sobel': True,
            'hough_histogram_bins': 60,
            'sobel_histogram_bins': 90,
            'sobel_magnitude_threshold': 0.01,
            'colormap': 'hsv'
        }
    }

def create_visualization_plots(analyzer: LineAnalyzer, 
                             results: Dict[str, Any]) -> Dict[str, plt.Figure]:
    """
    Creates matplotlib figures for visualization.
    Separate function to keep the main analysis clean.
    
    Args:
        analyzer: LineAnalyzer instance
        results: Analysis results dictionary
        
    Returns:
        Dictionary of matplotlib figures
    """
    figures = {}
    
    # Create histogram plots
    if 'hough' in results['histograms']:
        hough_data = results['histograms']['hough']
        figures['hough_histogram'] = analyzer.create_histogram_plot(
            hough_data['hist'], hough_data['bins'], hough_data['dominant'],
            "Hough Line Orientations", 'blue'
        )
    
    if 'sobel' in results['histograms']:
        sobel_data = results['histograms']['sobel']
        figures['sobel_histogram'] = analyzer.create_histogram_plot(
            sobel_data['hist'], sobel_data['bins'], sobel_data['dominant'],
            "Sobel Edge Orientations (Weighted)", 'green'
        )
    
    # Create line overlay visualization
    if 'hough_lines' in results['visualizations']:
        vis_data = results['visualizations']['hough_lines']
        figures['line_overlay'] = analyzer.create_visualization_overlay(
            vis_data['base_image'], vis_data['lines'], vis_data['colormap']
        )
    
    # Create Sobel visualization
    if 'sobel_orientation' in results['visualizations']:
        sobel_angles = analyzer.intermediate_images.get('sobel_orientation')
        sobel_magnitude = analyzer.intermediate_images.get('sobel_magnitude')
        
        if sobel_angles is not None and sobel_magnitude is not None:
            figures['sobel_visualization'] = analyzer.create_sobel_visualization(
                sobel_angles, sobel_magnitude, colormap='hsv'
            )
    
    return figures