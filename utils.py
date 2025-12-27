import numpy as np
from PIL import Image
import os

try:
    import cv2
    OPENCV_AVAILABLE = True
except ImportError:
    OPENCV_AVAILABLE = False

def load_image_as_matrix(image_path, size=(100, 100)):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found at {image_path}")

    if OPENCV_AVAILABLE:
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError("Could not load image")
            
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        face_cascade = cv2.CascadeClassifier(cascade_path)
        
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        
        if len(faces) == 0:
            raise ValueError("No face detected in the image. Please upload a clearer face image.")
            
        largest_face = max(faces, key=lambda rect: rect[2] * rect[3])
        x, y, w, h = largest_face
        
        face_img = gray[y:y+h, x:x+w]
        
        img_resized = cv2.resize(face_img, size, interpolation=cv2.INTER_AREA)
        matrix = np.array(img_resized)
        return matrix
    
    else:
        img = Image.open(image_path)
        img_gray = img.convert('L')
        img_resized = img_gray.resize(size)
        matrix = np.array(img_resized)
        return matrix

def flatten_matrix(matrix):
    return matrix.flatten()

# --- Classes merged from core ---

class FaceDetector:
    def __init__(self, cascade_path=None):
        if not OPENCV_AVAILABLE:
            self.detector = None
            return
            
        if cascade_path is None:
            # Handle potential missing data attribute in some cv2 versions or installs
            try:
                cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            except AttributeError:
                cascade_path = 'haarcascade_frontalface_default.xml' # Fallback
                
        self.detector = cv2.CascadeClassifier(cascade_path)

    def detect_faces(self, image):
        if not OPENCV_AVAILABLE or self.detector is None:
            return []
            
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.detector.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        return faces

class ImagePreprocessor:
    def __init__(self, target_size=(100, 100)):
        self.target_size = target_size

    def process(self, face_image):
        if not OPENCV_AVAILABLE:
            return face_image

        if len(face_image.shape) > 2:
            face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
        
        face_image = cv2.resize(face_image, self.target_size)
        face_image = cv2.equalizeHist(face_image)
        face_image = cv2.normalize(face_image, None, 0, 255, cv2.NORM_MINMAX)
        
        return face_image

class FaceRecognizer:
    def __init__(self, model_path='model.yml', threshold=100):
        self.threshold = threshold
        if OPENCV_AVAILABLE:
            self.recognizer = cv2.face.LBPHFaceRecognizer_create()
            if os.path.exists(model_path):
                self.recognizer.read(model_path)
        else:
            self.recognizer = None

    def recognize(self, face_image):
        if not OPENCV_AVAILABLE or self.recognizer is None:
            return "Unknown", 0
            
        label, confidence = self.recognizer.predict(face_image)
        if confidence > self.threshold:
            return "Unknown", confidence
        return label, confidence

class ModelTrainer:
    def __init__(self, dataset_path='dataset', model_path='model.yml'):
        self.dataset_path = dataset_path
        self.model_path = model_path
        if OPENCV_AVAILABLE:
            self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        else:
            self.recognizer = None

    def get_images_and_labels(self):
        if not os.path.exists(self.dataset_path):
            return [], []
            
        image_paths = [os.path.join(self.dataset_path, f) for f in os.listdir(self.dataset_path)]
        face_samples = []
        ids = []

        for image_path in image_paths:
            try:
                PIL_img = Image.open(image_path).convert('L')
                img_numpy = np.array(PIL_img, 'uint8')
                # Expected format: User.id.sample.jpg
                filename = os.path.split(image_path)[-1]
                id = int(filename.split(".")[1])
                face_samples.append(img_numpy)
                ids.append(id)
            except Exception:
                continue
        
        return face_samples, ids

    def train(self):
        if not OPENCV_AVAILABLE or self.recognizer is None:
            return
            
        faces, ids = self.get_images_and_labels()
        if len(faces) == 0:
            print("No training data found.")
            return
            
        self.recognizer.train(faces, np.array(ids))
        self.recognizer.write(self.model_path)
        print(f"Model trained with {len(faces)} samples.")
