# Traffic Sign Detection

A modular computer vision project for preparing traffic-sign images, detecting
candidate sign regions, and extracting features for later classification.

## Project Workflow

1. **Add the dataset**
   - Place original images in `data/raw/images/`.
   - Place matching VOC XML annotations in `data/raw/annotations/`.

2. **Preprocess images**
   - Run the preprocessing entry point to create Gaussian and median filtered
     versions of the raw images:
     ```bash
     python main.py
     ```
   - Outputs are saved under:
     - `data/processed/gaussian/`
     - `data/processed/median/`

3. **Detect candidate traffic-sign regions**
   - Run the K-Means batch detector:
     ```bash
     python detection/run_kmeans_batch.py
     ```
   - For a quick smoke test, limit the number of images:
     ```bash
     python detection/run_kmeans_batch.py --max-images 10
     ```
   - Detection outputs are written to `data/detection/kmeans/`, including
     annotated images, ROI crops, and per-image JSON metrics.

4. **Experiment and tune**
   - Use `detection/kmeans_segmentation_prototype.ipynb` for segmentation
     experiments.
   - See `detection/README.md` for detector options such as `--color-space`,
     `--k`, `--morph-kernel`, and `--select-by-detection`.

5. **Extract SIFT features from ROIs**
   - After detection creates ROI crops, extract SIFT descriptors and fixed-size
     feature vectors:
     ```bash
     python features/sift_extractor.py
     ```
   - Outputs are saved under `data/features/`:
     - `vectors/` for per-ROI fixed feature vectors
     - `descriptors/` for raw SIFT descriptors
     - `keypoints/` for keypoint metadata
     - `all_feature_vectors.npy` and `all_feature_vectors.csv` for the combined matrix


## Project Architecture

```text
project/
├── data/
│   ├── raw/                  # Original images and annotations
│   ├── processed/            # Gaussian and median filtered images
│   ├── detection/            # Detection outputs and ROI crops
│   └── features/             # SIFT vectors, descriptors, and summaries
├── detection/                # K-Means segmentation and batch detection tools
│   ├── run_kmeans_batch.py
│   └── README.md
├── features/                 # Feature extraction from detected ROIs
│   └── sift_extractor.py
├── preprocessing/            # Filtering, resizing, and normalization modules
│   └── preprocessing.py
├── utils/                    # Image loading and visualization helpers
├── main.py                   # Preprocessing entry point
├── requirements.txt          # Project dependencies
└── README.md                 # Project documentation
```

## Setup Instructions

1. **Clone the repository** or navigate to the workspace, then create a virtual environment:
   ```bash
   python -m venv venv
   ```

2. **Activate the virtual environment**:
   - **Windows:**
     ```bash
     venv\Scripts\activate
     ```
   - **macOS/Linux:**
     ```bash
     source venv/bin/activate
     ```

3. **Install the required packages**:
   ```bash
   pip install -r requirements.txt
   ```

## Quick Start

```bash
python main.py
python detection/run_kmeans_batch.py --max-images 10
python features/sift_extractor.py
```

Use the full detector command when the smoke test looks correct:

```bash
python detection/run_kmeans_batch.py
```
