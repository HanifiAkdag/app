# Functional Materials Image Analysis

A comprehensive Streamlit-based application for analyzing microscopy images of functional materials. This tool provides advanced image processing capabilities including phase analysis, line detection, and artifact removal specifically designed for materials science applications.

## Features

### 🔬 **Phase Analysis**
- Automatic phase detection using histogram analysis
- Multiple segmentation methods (K-means, Gaussian Mixture, Manual thresholding)
- Artifact removal with various detection and inpainting methods
- Comprehensive statistical analysis and visualization

### 📏 **Line Analysis**
- Frangi filter for line enhancement
- Hough transform for line detection
- Sobel edge analysis with orientation histograms
- Skeletonization and boundary exclusion
- Material mask integration

### 🎯 **Artifact Removal**
- Multiple detection methods (percentile, absolute threshold, Otsu)
- Advanced inpainting techniques (Telea, Navier-Stokes, LaMa)
- Morphological filtering and component analysis
- Before/after comparison visualizations

### 🔧 **Preprocessing**
- Illumination correction (blur subtract/divide)
- Multiple denoising methods (median, Gaussian, bilateral, NLM)
- Image normalization and enhancement
- Processing history tracking

## Installation

1. **Create a virtual environment:**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

## Usage

### Running the Application

```bash
streamlit run main.py
```

The application will open in your default web browser at `http://localhost:8501`.

### Basic Workflow

1. **Upload Image**: Upload microscopy images (PNG, JPG, TIFF formats supported)
2. **Preprocessing**: Apply illumination correction and denoising if needed
3. **Analysis**: Choose from phase analysis, line analysis, or artifact removal
4. **Results**: View comprehensive visualizations and download processed images
5. **History**: Track and compare different processing steps


## Key Dependencies

- **Streamlit**: Web application framework
- **OpenCV**: Computer vision and image processing
- **scikit-image**: Image processing algorithms
- **NumPy**: Numerical computing
- **Matplotlib**: Plotting and visualization
- **PIL/Pillow**: Image I/O operations

### Optional Dependencies

- **simple-lama-inpainting**: Advanced deep learning inpainting (for artifact removal)

## Analysis Modules

### PhaseAnalyzer
Provides comprehensive phase analysis capabilities:
- Automatic phase detection using histogram peak finding
- Multiple segmentation algorithms
- Statistical analysis of phase distributions
- Artifact removal and cleaning

### LineAnalyzer
Specialized for detecting and analyzing linear features:
- Frangi filter for line enhancement
- Hough transform for geometric line detection
- Sobel edge analysis for orientation patterns
- Boundary exclusion and skeletonization

### ArtifactRemover
Focuses on cleaning microscopy artifacts:
- Multiple detection methods for bright/dark spots
- Advanced inpainting using various algorithms
- Morphological filtering and size-based selection
- Statistical reporting of removed artifacts

## Output

The application automatically creates timestamped output directories containing:
- Processed images at each step
- Analysis results in JSON format
- Visualization plots
- Processing history logs

## Tips for Best Results

1. **Image Quality**: Use high-quality, well-illuminated microscopy images
2. **Preprocessing**: Apply illumination correction for uneven lighting
3. **Parameter Tuning**: Adjust analysis parameters based on your specific material
4. **Material Masks**: Use phase analysis results to create material masks for line analysis
5. **Processing Chain**: Follow preprocessing → phase analysis → line analysis workflow