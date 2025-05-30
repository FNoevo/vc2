import cv2
import numpy as np
import os

# Pastas
input_folder = os.path.join(os.path.dirname(__file__), "..", "imagens")
output_folder = os.path.join(os.path.dirname(__file__), "..", "resultados", "geometric_logic")
os.makedirs(output_folder, exist_ok=True)

# Processar cada imagem
for filename in os.listdir(input_folder):
    if filename.lower().endswith((".jpg", ".jpeg", ".png")):
        input_path = os.path.join(input_folder, filename)
        img = cv2.imread(input_path)

        if img is None:
            print(f"⚠️ Erro ao carregar {filename}, a ignorar...")
            continue

        # Redimensionar
        resized = cv2.resize(img, (300, 300))

        # Rotacionar
        rotated = cv2.rotate(resized, cv2.ROTATE_90_CLOCKWISE)

        # Operação lógica: NOT
        not_img = cv2.bitwise_not(rotated)

        # Nome do ficheiro sem extensão
        name = os.path.splitext(filename)[0]

        # Caminho de saída
        output_path = os.path.join(output_folder, f"{name}_processed.jpg")

        # Guardar imagem resultante
        cv2.imwrite(output_path, not_img)
        print(f"✅ Processado: {filename} → {output_path}")
