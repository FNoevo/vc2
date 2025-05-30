import cv2
import numpy as np
from tensorflow.keras.models import load_model

# === Carregar o modelo ===
model = load_model('contador_pessoas.h5')

# === Caminho para a imagem de teste ===
img_path = 'part_B/test_data/images/IMG_1.jpg'  # Altera conforme necessário

# === Ler e preparar imagem ===
img = cv2.imread(img_path)
if img is None:
    raise FileNotFoundError(f"❌ Imagem não encontrada: {img_path}")

img = cv2.resize(img, (128, 128))
img = img.astype('float32') / 255.0
img = np.expand_dims(img, axis=0)  # (1, 128, 128, 3)

# === Prever contagem ===
pred = model.predict(img)[0][0]
print(f"🔢 Contagem prevista: {round(pred)} pessoas")
