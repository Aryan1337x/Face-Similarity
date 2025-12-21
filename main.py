import sys
import os
from utils import load_image_as_matrix, flatten_matrix
from similarity import calculate_euclidean_distance, explain_calculation_step

def main():
    if len(sys.argv) == 3:
        img_path1 = sys.argv[1]
        img_path2 = sys.argv[2]
    else:
        from PIL import Image, ImageDraw
        
        img1 = Image.new('L', (100, 100), color=255)
        d1 = ImageDraw.Draw(img1)
        d1.rectangle([20, 20, 80, 80], fill=0)
        img1.save("demo_face_a.png")
        img_path1 = "demo_face_a.png"
        
        img2 = Image.new('L', (100, 100), color=255)
        d2 = ImageDraw.Draw(img2)
        d2.rectangle([25, 25, 85, 85], fill=50)
        img2.save("demo_face_b.png")
        img_path2 = "demo_face_b.png"

    try:
        matrix1 = load_image_as_matrix(img_path1)
        vec1 = flatten_matrix(matrix1)
        
        matrix2 = load_image_as_matrix(img_path2)
        vec2 = flatten_matrix(matrix2)
        
        dist = calculate_euclidean_distance(vec1, vec2)
        print(dist)
        
    except Exception as e:
        print(e)

if __name__ == "__main__":
    main()
