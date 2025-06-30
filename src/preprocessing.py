import cv2
import numpy as np
from typing import Union, Optional, Tuple

from .utils import make_odd

def correct_illumination(image: np.ndarray, method: Optional[str], kernel_size: int = 65) -> np.ndarray:
    """
    Corrects uneven illumination in an image using various methods.
    
    Args:
        image (np.ndarray): Input grayscale image as numpy array
        method (Optional[str]): Illumination correction method. Options: "none", "blur_subtract", "blur_divide", or None
        kernel_size (int, optional): Size of Gaussian blur kernel for background estimation. Defaults to 65.
    
    Returns:
        np.ndarray: Illumination-corrected image as uint8 numpy array
        
    Raises:
        ValueError: If kernel_size is <= 0
    """
    img_corrected = image.copy()
    if method is not None and method != "none":
        k_size = make_odd(kernel_size)
        if k_size <= 0: raise ValueError("illumination_kernel_size must be > 0.")
        background = cv2.GaussianBlur(img_corrected, (k_size, k_size), 0)
        
        # Blur subtract: Good for removing gradual lighting variations while preserving contrast
        # Subtracts background illumination and adds mean to maintain brightness levels
        if method == "blur_subtract":
            img_i16, bg_i16 = img_corrected.astype(np.int16), background.astype(np.int16)
            corrected_i16 = np.clip(img_i16 - bg_i16 + int(np.mean(bg_i16)), 0, 255)
            img_corrected = corrected_i16.astype(np.uint8)
        
        # Blur divide: Effective for multiplicative illumination variations (shadows, vignetting)
        # Normalizes pixel values relative to local background illumination
        elif method == "blur_divide":
            bg_float = background.astype(np.float32) + 1e-5
            img_float = img_corrected.astype(np.float32)
            original_mean = np.mean(img_float) if np.mean(img_float) > 1e-5 else 128
            corrected_float = np.clip((img_float / bg_float) * original_mean, 0, 255)
            img_corrected = corrected_float.astype(np.uint8)
    return img_corrected

def denoise_image(image: np.ndarray, method: Optional[str],
                  median_k_size: int = 3,
                  gaussian_sigma: float = 1,
                  bilateral_d: int = 5, bilateral_sigma_color: float = 75, bilateral_sigma_space: float = 75,
                  nlm_h: float = 10, nlm_template_window_size: int = 7, nlm_search_window_size: int = 21) -> np.ndarray:
    """
    Applies denoising to an image using various filtering methods.
    
    Args:
        image (np.ndarray): Input grayscale image as numpy array
        method (Optional[str]): Denoising method. Options: "median", "gaussian", "bilateral", "nlm", or None
        median_k_size (int, optional): Kernel size for median filter. Defaults to 3.
        gaussian_sigma (float, optional): Standard deviation for Gaussian blur. Defaults to 1.
        bilateral_d (int, optional): Diameter for bilateral filter. Defaults to 5.
        bilateral_sigma_color (float, optional): Filter sigma in color space for bilateral filter. Defaults to 75.
        bilateral_sigma_space (float, optional): Filter sigma in coordinate space for bilateral filter. Defaults to 75.
        nlm_h (float, optional): Filter strength for Non-Local Means. Higher value removes more noise but removes image details too. Defaults to 10.
        nlm_template_window_size (int, optional): Template patch size for NLM. Should be odd. Defaults to 7.
        nlm_search_window_size (int, optional): Search window size for NLM. Should be odd. Defaults to 21.
    
    Returns:
        np.ndarray: Denoised image as uint8 numpy array
    """
    img_denoised = image.copy()

    # Median filter: Excellent for removing salt-and-pepper noise while preserving sharp edges
    # Non-linear filter that replaces each pixel with the median of its neighborhood
    if method == "median":
        k_size = make_odd(median_k_size)
        if k_size > 1: img_denoised = cv2.medianBlur(img_denoised, k_size)

    # Gaussian blur: Fast and effective for reducing random noise and smoothing images
    # Linear filter that creates smooth transitions but may blur important edges
    elif method == "gaussian":
        img_denoised = cv2.GaussianBlur(img_denoised, (0,0), gaussian_sigma)

    # Bilateral filter: Preserves edges while reducing noise in smooth regions
    # Considers both spatial distance and intensity difference for edge-aware smoothing
    elif method == "bilateral":
        img_denoised = cv2.bilateralFilter(img_denoised, bilateral_d, bilateral_sigma_color, bilateral_sigma_space)

    # Non-Local Means (NLM): Superior noise reduction by finding similar patches across the image
    # Computationally intensive but excellent for preserving textures and fine details
    elif method == "nlm":
        template_win = make_odd(nlm_template_window_size)
        search_win = make_odd(nlm_search_window_size)
        img_denoised = cv2.fastNlMeansDenoising(img_denoised, None, nlm_h, template_win, search_win)
        
    return img_denoised

def _get_threshold_value(image: np.ndarray, 
                        method_or_value: Union[str, float]) -> float:
    """
    Helper to get threshold value from method name or direct value.
    
    Args:
        image: Input image region
        method_or_value: Threshold method name or direct float value
        
    Returns:
        Threshold value
    """
    from skimage import filters
    
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

def create_material_mask(image: np.ndarray,
                        strategy: str = "fill_holes",
                        background_threshold: Union[str, float] = 0.08,
                        cleanup_area: int = 500) -> Tuple[np.ndarray, np.ndarray]:
    """
    Creates material and background masks using different strategies.
    
    Args:
        image: Input image (typically preprocessed)
        strategy: Masking strategy ("fill_holes" or "bright_phases")
        background_threshold: Threshold for background identification
        cleanup_area: Area threshold for mask cleaning
        
    Returns:
        Tuple of (material_mask, background_mask)
    """
    from skimage.morphology import remove_small_objects, remove_small_holes
    from scipy.ndimage import binary_fill_holes

    if strategy == "fill_holes":
        # Identify background pixels (very dark)
        thresh_bg = _get_threshold_value(image, background_threshold)
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
        thresh_bright = _get_threshold_value(image, background_threshold)
        final_material_mask = image > thresh_bright
        
        if cleanup_area > 0:
            final_material_mask = remove_small_objects(final_material_mask, min_size=cleanup_area, connectivity=1)
            final_material_mask = remove_small_holes(final_material_mask, area_threshold=cleanup_area, connectivity=1)
    
    else:
        raise ValueError(f"Unknown masking strategy: {strategy}")
    
    background_mask = ~final_material_mask
    
    return final_material_mask, background_mask