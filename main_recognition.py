import cv2
import os
import sys
from dataset_manager import DatasetManager
from utils import ModelTrainer, FaceRecognizer, FaceDetector, ImagePreprocessor

def load_names():
    names = {}
    if os.path.exists("names.txt"):
        with open("names.txt", "r") as f:
            for line in f:
                parts = line.strip().split(",")
                if len(parts) == 2:
                    names[int(parts[0])] = parts[1]
    return names

def save_name(id, name):
    names = load_names()
    names[id] = name
    with open("names.txt", "w") as f:
        for k, v in names.items():
            f.write(f"{k},{v}\n")

def main():
    if len(sys.argv) < 2:
        print("Usage: python main_recognition.py [capture|train|recognize]")
        return

    mode = sys.argv[1]

    if mode == "capture":
        id = int(input("Enter User ID: "))
        name = input("Enter User Name: ")
        save_name(id, name)
        dm = DatasetManager()
        dm.capture_faces(id)

    elif mode == "train":
        mt = ModelTrainer()
        mt.train()

    elif mode == "recognize":
        names = load_names()
        fr = FaceRecognizer()
        fd = FaceDetector()
        pp = ImagePreprocessor()
        
        cam = cv2.VideoCapture(0)
        
        while True:
            ret, img = cam.read()
            if not ret:
                break
            
            faces = fd.detect_faces(img)
            
            for (x, y, w, h) in faces:
                face = img[y:y+h, x:x+w]
                processed_face = pp.process(face)
                label, confidence = fr.recognize(processed_face)
                
                name = names.get(label, f"ID:{label}") if label != "Unknown" else "Unknown"
                text = f"{name} ({round(confidence, 2)})"
                
                color = (0, 255, 0) if label != "Unknown" else (0, 0, 255)
                cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)
                cv2.putText(img, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            
            cv2.imshow('Face Recognition', img)
            
            if cv2.waitKey(10) & 0xFF == 27:
                break
        
        cam.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
