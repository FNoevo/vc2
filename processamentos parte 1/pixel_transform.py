import cv2, os
import numpy as np

input_folder  = os.path.join(os.path.dirname(__file__), "..", "imagens")
output_folder = os.path.join(os.path.dirname(__file__), "..", "resultados", "pixel_transform")
os.makedirs(output_folder, exist_ok=True)

for fn in os.listdir(input_folder):
    if not fn.lower().endswith((".jpg",".png",".jpeg")): continue
    img = cv2.imread(os.path.join(input_folder, fn))
    if img is None:
        continue

    # Negativo
    inverted = 255 - img

    # Binarização simples (threshold)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)
    binary_bgr = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)

    # Comparação lado a lado
    compare = np.hstack([img, inverted, binary_bgr])

    name = os.path.splitext(fn)[0]
    cv2.imwrite(f"{output_folder}/{name}_compare.jpg", compare)
