# Face Similarity & Recognition Project

A comprehensive tool for **Face Similarity Comparison** and **Real-time Face Recognition**, built with Python, OpenCV, and Flask.

## Features

### 1. Face Similarity Meter (Web App)
- **Web Interface**: Clean, modern UI for uploading images.
- **Euclidean Distance**: Calculates the mathematical distance between two faces.
- **Similarity Percentage**: Converts distance into a readable 0-100% similarity score.
- **Privacy Focused**: Processes images locally.

### 2. Face Recognition System (CLI Tool)
- **Face Capture**: Caputure training data from your webcam.
- **Model Training**: Train an LBPH (Local Binary Patterns Histograms) recognizer.
- **Real-time Recognition**: Detect and identify faces via webcam feed.

## Installation

1.  **Clone the repository** (skip if you have the files):
    ```bash
    git clone https://github.com/Aryan1337x/Face-Similarity.git
    cd <repository-folder>
    ```

2.  **Install Dependencies**:
    Make sure you have Python installed. Then run:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

### Part 1: Face Similarity Web App

1.  **Run the Flask App**:
    ```bash
    python app.py
    ```
2.  **Open in Browser**:
    Go to [http://127.0.0.1:5000](http://127.0.0.1:5000)
3.  **Upload & Compare**: Upload two images to see their similarity score.

### Part 2: Face Recognition Tool

This tool operates in three modes: `capture`, `train`, and `recognize`. Run it using `main_recognition.py`.

#### Step 1: Capture Faces
Register a new user by capturing their face data.
```bash
python main_recognition.py capture
```
- Enter a numerical **User ID** (e.g., `1`) and **Name** (e.g., `Alice`).
- Look at the camera while it captures samples.

#### Step 2: Train Model
Train the recognizer on the captured data.
```bash
python main_recognition.py train
```

#### Step 3: Recognize Faces
Start real-time recognition.
```bash
python main_recognition.py recognize
```
- Press **ESC** to stop.

## Project Structure

- `app.py`: Main Flask application for the Similarity Meter.
- `main_recognition.py`: CLI entry point for the Face Recognition system.
- `dataset_manager.py`: Handles face data capture.
- `similarity.py`: Logic for calculating face similarity.
- `utils.py`: Helper functions and core logic for image processing, detection, training, and recognition.
- `static/` & `templates/`: Frontend assets for the Flask app.

## Deployment
- **Render**:
    1.  Create a new Web Service on Render.
    2.  Connect your repository.
    3.  Set the **Start Command** to `gunicorn app:app`.
    4.  **Optimization**: In the "Environment" tab, set the following variables to `1` to prevent memory issues on the basic plan: `OMP_NUM_THREADS`, `MKL_NUM_THREADS`, `OPENBLAS_NUM_THREADS`.

## Credits

Made by Aryan  
Instagram: [@aryan.skid](https://www.instagram.com/aryan.skid/)
