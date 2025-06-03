import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

input_folder = os.path.join(os.path.dirname(__file__), "..", "imagens")
output_folder = os.path.join(os.path.dirname(__file__), "..", "resultados", "histogram_analysis")
os.makedirs(output_folder, exist_ok=True)

for filename in os.listdir(input_folder):
    if filename.lower().endswith((".jpg", ".jpeg", ".png")):
        input_path = os.path.join(input_folder, filename)
        img = cv2.imread(input_path)

        if img is None:
            print(f"⚠️ Erro ao carregar {filename}, a ignorar...")
            continue

        # Redimensionar para visualização
        resized = cv2.resize(img, (300, 300))

        # Calcular histogramas dos canais BGR
        colors = ('b', 'g', 'r')
        plt.figure(figsize=(8,4))
        plt.subplot(1,2,1)
        plt.imshow(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.title('Imagem')

        plt.subplot(1,2,2)
        for i, color in enumerate(colors):
            hist = cv2.calcHist([resized], [i], None, [256], [0,256])
            plt.plot(hist, color=color)
            plt.xlim([0,256])
        plt.title('Histograma (BGR)')
        plt.xlabel('Intensidade')
        plt.ylabel('Nº de pixels')

        plt.tight_layout()
        out_path = os.path.join(output_folder, f"{os.path.splitext(filename)[0]}_hist.png")
        plt.savefig(out_path)
        plt.close()
        print(f"✅ Histograma guardado: {out_path}")
