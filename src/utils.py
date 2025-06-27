import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import filters
from skimage.morphology import remove_small_objects, remove_small_holes

def make_odd(k):
    """
    Ensures a kernel size is odd.
    Many OpenCV functions require odd kernel sizes. This function takes an
    integer or float, rounds it to the nearest integer, and if it's even,
    adds 1 to make it odd.

    Args:
        k (int or float): The desired kernel size.

    Returns:
        int: An odd integer kernel size.
    """
    k = int(round(k))
    return k if k % 2 == 1 else k + 1

def load_and_prepare_image(image_path_or_array):
    """Loads an image, converts to grayscale, ensures np.uint8 format."""
    if isinstance(image_path_or_array, str):
        img = cv2.imread(image_path_or_array, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Image not found at {image_path_or_array}")
    elif isinstance(image_path_or_array, np.ndarray):
        img = image_path_or_array.copy()
        if len(img.shape) == 3: # Color image
            if img.shape[2] == 3: img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            elif img.shape[2] == 4: img = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
            else: raise ValueError("Input NumPy array has 3 dims but not 3 or 4 channels.")

        if img.dtype != np.uint8: # Convert to uint8 if not already
            if np.issubdtype(img.dtype, np.floating):
                if img.min() >= 0.0 and img.max() <= 1.0 and not (img.min()==0 and img.max()==0):
                    img = (img * 255.0).astype(np.uint8)
                else:
                    img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            elif np.issubdtype(img.dtype, np.integer): # e.g., uint16
                img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            else:
                raise TypeError(f"Unsupported NumPy array dtype: {img.dtype}")
    else:
        raise TypeError("Input must be a file path (str) or a NumPy array.")
    return img

def show_image_steps(step_images, step_titles, figsize=(18, 10)):
    """
    Helper function to display multiple images in a grid using Matplotlib.
    It arranges images in a grid with a maximum of 3 columns.

    Args:
        step_images (list of np.ndarray): A list of images to display.
        step_titles (list of str): A list of titles corresponding to each image.
        figsize (tuple): The size of the Matplotlib figure.
    """
    num_steps = len(step_images)
    if num_steps == 0:
        print("No images to display.")
        return

    cols = min(3, num_steps)  # Max 3 columns
    rows = (num_steps + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    if num_steps == 1: # if only one image, axes is not an array
        axes = [axes]
    else:
        axes = axes.flatten()

    for i in range(num_steps):
        img_to_show = step_images[i]
        if img_to_show.dtype != np.uint8:
            img_to_show = cv2.normalize(img_to_show, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        axes[i].imshow(img_to_show, cmap='gray', vmin=0, vmax=255)
        axes[i].set_title(step_titles[i])
        axes[i].axis('off')

    for j in range(num_steps, len(axes)): # Remove unused subplots
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()