# Traffic Sign Detection

A clean and modular Computer Vision project for detecting and classifying traffic signs.

## Project Architecture
```
project/
├── data/
│   ├── raw/                 # Original images from Kaggle
│   └── processed/           # Images after preprocessing
├── preprocessing/           # Modules for data preparation and filtering
│   ├── gaussian_filter.py
│   ├── median_filter.py
│   ├── resize.py
│   ├── normalize.py
│   └── metrics.py           # MSE, PSNR calculations
├── utils/                   # Helpers for loading and visualization
│   ├── image_loader.py
│   └── visualization.py
├── notebooks/               # Jupyter notebooks for experimentation
│   └── preprocessing_demo.ipynb
├── main.py                  # Main project entry point
├── requirements.txt         # Project dependencies
└── README.md                # Project documentation
```

## Setup Instructions

1. **Clone the repository** (or navigate to the workspace) and create a virtual environment:
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

## Getting Started
- Run the entry pipeline to verify setup:
  ```bash
  python main.py
  ```
- Explore the interactive Jupyter notebook in `notebooks/` to visualize preprocessing steps.
- Add your Kaggle dataset images to `data/raw/` to begin filtering and feature extraction.
