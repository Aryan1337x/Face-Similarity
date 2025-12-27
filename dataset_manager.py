import cv2
import os
from utils import FaceDetector, ImagePreprocessor

class DatasetManager:
    def __init__(self, dataset_path='dataset'):
        self.dataset_path = dataset_path
        if not os.path.exists(self.dataset_path):
            os.makedirs(self.dataset_path)
        self.detector = FaceDetector()
        self.preprocessor = ImagePreprocessor()

    def capture_faces(self, user_id, num_samples=30):
        cam = cv2.VideoCapture(0)
        count = 0
        
        while True:
            ret, img = cam.read()
            if not ret:
                break
                
            faces = self.detector.detect_faces(img)
            
            for (x, y, w, h) in faces:
                face = img[y:y+h, x:x+w]
                processed_face = self.preprocessor.process(face)
                
                count += 1
                cv2.imwrite(f"{self.dataset_path}/User.{user_id}.{count}.jpg", processed_face)
                cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
                cv2.imshow('Capturing Faces', img)

            k = cv2.waitKey(100) & 0xff
            if k == 27 or count >= num_samples:
                break

        cam.release()
        cv2.destroyAllWindows()