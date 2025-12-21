import numpy as np
from PIL import Image
import os

def load_image_as_matrix(image_path, size=(100, 100)):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found at {image_path}")

    img = Image.open(image_path)
    img_gray = img.convert('L')
    img_resized = img_gray.resize(size)
    matrix = np.array(img_resized)
    
    return matrix

def flatten_matrix(matrix):
    return matrix.flatten()
