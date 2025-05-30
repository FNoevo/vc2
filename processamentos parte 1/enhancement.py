import cv2
import numpy as np
import os

input_folder = os.path.join(os.path.dirname(__file__), "..", "imagens")
output_folder = os.path.join(os.path.dirname(__file__), "..", "resultados", "enhancement")
os.makedirs(output_folder, exist_ok=True)

kernel = np.array([[0, -1, 0],
                   [-1, 5, -1],
                   [0, -1, 0]])

for filename in os.listdir(input_folder):
    if filename.lower().endswith((".jpg", ".png", ".jpeg")):
        path = os.path.join(input_folder, filename)
        img = cv2.imread(path)

        sharpened = cv2.filter2D(img, -1, kernel)

        name = filename.rsplit('.', 1)[0]
        cv2.imwrite(f"{output_folder}/{name}_enhanced.jpg", sharpened)
