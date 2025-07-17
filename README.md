# Functional Materials Image Analysis

Streamlit-based microscopy image analysis tool for materials science with custom pipeline builder.

## 🚀 Features

- **Custom Mask Drawing**: Interactive canvas for drawing custom analysis regions
- **Phase Analysis**: Automatic detection, K-means/manual segmentation, statistical analysis
- **Line Analysis**: Frangi filter, Hough transform, orientation analysis  
- **Artifact Removal**: Multiple detection methods, advanced inpainting (Telea, Navier-Stokes)
- **Preprocessing**: Illumination correction, denoising, material masking
- **Pipeline Builder**: Interactive workflow design, save/load configurations, batch processing

## 💻 Setup

### Quick Start

1. **Create a virtual environment:**
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

**Note:** For the mask drawing feature, `streamlit-drawable-canvas` is included in requirements.txt and will be installed automatically.

### Verify Installation
```bash
python -c "import streamlit; import cv2; import numpy; print('Installation successful!')"
```

## 🎯 Usage

### Running the Application

```bash
streamlit run main.py
```

The application will open at `http://localhost:8501`.

### Basic Workflow

1. **Upload Image**: Drag and drop microscopy images (PNG, JPG, TIFF)
2. **Build Pipeline**: Add processing steps (preprocessing, phase analysis, line analysis, artifact removal)
3. **Configure Parameters**: Adjust settings for each step
4. **Execute Pipeline**: Run the complete workflow
5. **Review Results**: Examine outputs, statistics, and visualizations
6. **Export**: Download processed images and analysis reports

### Analysis Modules Deep Dive

#### PhaseAnalyzer (`src/phase_analysis.py`)
**Purpose**: Comprehensive phase segmentation and analysis for multi-phase materials

**Key Methods**:
- `detect_number_of_phases()`: Automatic phase detection using histogram peak analysis
- `perform_phase_segmentation()`: K-means, Gaussian mixture, or manual thresholding
- `calculate_phase_statistics()`: Area fractions, distributions, and morphological metrics
- `run_full_analysis()`: Complete pipeline with preprocessing and visualization

**Algorithms**:
- Histogram-based peak detection with prominence filtering
- K-means clustering with optimized initialization
- Gaussian mixture models for overlapping phase separation
- Morphological post-processing for mask refinement

#### LineAnalyzer (`src/line_analysis.py`)
**Purpose**: Detection and analysis of linear features and orientations

**Key Methods**:
- `apply_frangi_filter()`: Multi-scale ridge enhancement for line detection
- `detect_hough_lines()`: Geometric line detection with parameter optimization
- `calculate_sobel_orientation()`: Edge-based orientation analysis
- `analyze_orientations()`: Statistical analysis of directional features

**Algorithms**:
- Frangi vesselness filter for scale-adaptive line enhancement
- Probabilistic Hough transform for robust line detection
- Sobel gradient analysis with magnitude weighting
- Circular statistics for orientation distribution analysis

#### ArtifactRemover (`src/artifact_removal.py`)
**Purpose**: Detection and removal of imaging artifacts and noise

**Key Methods**:
- `create_artifact_mask()`: Multi-method artifact detection
- `clean_mask()`: Morphological filtering and size-based selection
- `inpaint_artifacts()`: Multiple inpainting algorithms
- `run_full_analysis()`: Complete artifact removal pipeline

**Algorithms**:
- Percentile-based and Otsu threshold detection
- Morphological opening and component analysis
- Telea and Navier-Stokes inpainting

#### Preprocessing (`src/preprocessing.py`)
**Purpose**: Image enhancement and preparation for analysis

**Key Functions**:
- `correct_illumination()`: Blur-based illumination correction methods
- `denoise_image()`: Multiple denoising algorithms
- `create_material_mask()`: Automatic foreground/background separation

**Algorithms**:
- Gaussian blur background estimation and correction
- Median, bilateral, and Non-Local Means denoising
- Otsu thresholding and morphological mask refinement

### Parameter Optimization

#### Phase Analysis
- **Automatic Detection**: Enable auto-detection for unknown phase numbers
- **K-means vs Manual**: Use K-means for clear phase separation, manual thresholds for fine control
- **Material Masking**: Choose "fill_holes" for porous materials, "bright_phases" for high-contrast samples

#### Line Analysis
- **Frangi Parameters**: Adjust sigma range based on expected line thickness
- **Hough Sensitivity**: Lower thresholds for weak lines, higher for noise reduction
- **Boundary Exclusion**: Use larger erosion sizes for high-noise edge regions

#### Artifact Removal
- **Detection Method**: Percentile-based for consistent results, Otsu for automatic adaptation
- **Inpainting Choice**: Telea for speed, Navier-Stokes for quality
- **Size Filtering**: Set appropriate min/max areas based on typical artifact sizes