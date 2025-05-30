import cv2
import os

input_folder = os.path.join(os.path.dirname(__file__), "..", "imagens")
output_folder = os.path.join(os.path.dirname(__file__), "..", "resultados", "edges")
os.makedirs(output_folder, exist_ok=True)

for filename in os.listdir(input_folder):
    if filename.lower().endswith((".jpg", ".png", ".jpeg")):
        path = os.path.join(input_folder, filename)
        img = cv2.imread(path, 0)

        edges = cv2.Canny(img, 100, 200)

        name = filename.rsplit('.', 1)[0]
        cv2.imwrite(f"{output_folder}/{name}_edges.jpg", edges)
