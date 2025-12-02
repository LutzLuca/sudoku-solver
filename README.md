# Sudoku Solver

A computer vision-based Sudoku solver that uses deep learning to recognize digits and automatically solve Sudoku puzzles from images.

## Overview

This project combines computer vision techniques with a convolutional neural network (CNN) to:
1. Extract a Sudoku grid from an image
2. Identify individual digits using a trained neural network
3. Solve the puzzle using backtracking
4. Display the solution through a web interface

## Project Structure

```
.
├── digit_recognition/          # Digit recognition model training
│   ├── digit_image_generator.py   # Generates training images from fonts
│   ├── prepare_data.py            # Data preparation and loading
│   └── train.py                   # Model training script
├── sudoku_solver/              # Core Sudoku solving logic
│   ├── extractor.py               # Image processing and digit extraction
│   ├── solver.py                  # Backtracking algorithm
│   └── main.py                    # Main processing pipeline
├── static/                     # Frontend assets
│   ├── index.js                   # Client-side JavaScript
│   └── styles.css                 # Styling
├── templates/                  # HTML templates
│   └── index.html                 # Web interface
├── server.py                   # Flask web server
└── requirements.txt            # Python dependencies
```

## Components

### 1. Digit Recognition Model

The digit recognition system uses a CNN trained on:
- **MNIST dataset** (handwritten digits 1-9)
- **Custom font-generated digits** - Simple, clean digit renders from various fonts to provide additional training data

**Note**: The font-generated images are basic renders without augmentation techniques (like blurring, rotation, or distortion). Adding such augmentation could potentially improve accuracy across different handwriting styles.

**Model Architecture:**
- 2 Convolutional layers (32 and 64 filters)
- MaxPooling layers
- Dropout for regularization
- Dense output layer with softmax activation (9 classes)

### 2. Image Processing

The `extractor.py` module handles:
- Preprocessing (Gaussian blur, adaptive thresholding, dilation)
- Sudoku grid detection using contour detection
- Perspective transformation to normalize the grid
- Individual cell extraction with noise filtering

### 3. Sudoku Solver

Uses a standard backtracking algorithm to solve the puzzle.

### 4. Web Interface

Flask-based web application that accepts image uploads, processes them, and displays the solved grid

## Setup

### Prerequisites

- Python 3.8+
- pip

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd <project-directory>
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

### Training the Model (Optional)

If you want to train your own model:

1. Generate training data from fonts:
```bash
python -m digit_recognition.digit_image_generator fonts/ data/digits/
```

2. Train the model:
```bash
python -m digit_recognition.train --data_dir data/digits --save_path models/model.keras --epochs 10
```

Parameters:
- `--data_dir`: Directory containing training images
- `--save_path`: Where to save the trained model
- `--batch_size`: Training batch size (default: 128)
- `--epochs`: Number of training epochs (default: 10)
- `--mnist`: Percentage of MNIST data to use (default: 1.0)

## Usage

### Running the Web Server

Start the Flask server with your trained model:

```bash
python server.py models/model.keras
```

The server will start at `http://localhost:8000`

### Using the Web Interface

1. Open your browser and navigate to `http://localhost:8000`
2. Click "Upload Sudoku Image"
3. Select an image of a Sudoku puzzle
4. Wait for processing (the image will be automatically resized if needed)
5. View the solved puzzle displayed in a grid

### Image Requirements

For best results, ensure:
- Clear, well-lit images with good contrast
- Visible grid lines
- Readable digits (printed or clearly handwritten)
- The Sudoku grid is the main feature in the image
- Avoid excessive shadows or glare

### Command-Line Usage

You can also process images directly:

```bash
python -m sudoku_solver.main <image_path> --model models/model.keras
```

## How It Works

1. **Image Upload**: User uploads a Sudoku image through the web interface
2. **Preprocessing**: Image is converted to grayscale and preprocessed with adaptive thresholding
3. **Grid Detection**: Contour detection finds the largest quadrilateral (the Sudoku grid)
4. **Perspective Correction**: Applies perspective transformation to normalize the grid
5. **Cell Extraction**: Divides the grid into 81 cells and extracts individual digits
6. **Digit Recognition**: CNN predicts the digit in each non-empty cell
7. **Solving**: Backtracking algorithm solves the puzzle
8. **Display**: Solution is returned and displayed in the web interface

## Limitations

- Requires clear images with visible grid lines
- Works best with printed or clearly handwritten digits
- May struggle with heavily stylized fonts or poor lighting

## Technologies Used

- **TensorFlow/Keras**: Deep learning framework
- **OpenCV**: Computer vision and image processing
- **Flask**: Web framework
- **NumPy**: Numerical computations
- **Pillow**: Image manipulation
- **scikit-learn**: Data preprocessing