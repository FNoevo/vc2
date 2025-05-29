import os
import scipy.io
import cv2
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam

# === Leitura de imagens e ground-truth ===
IMG_DIR = 'part_B/train_data/images'
GT_DIR = 'part_B/train_data/ground-truth'

X = []
y = []

for filename in tqdm(os.listdir(IMG_DIR)):
    if not filename.endswith('.jpg'):
        continue

    img_path = os.path.join(IMG_DIR, filename)
    mat_path = os.path.join(GT_DIR, 'GT_' + filename.replace('.jpg', '.mat'))

    img = cv2.imread(img_path)
    if img is None:
        continue
    img = cv2.resize(img, (128, 128))
    X.append(img)

    mat = scipy.io.loadmat(mat_path)
    points = mat["image_info"][0][0][0][0][0]
    count = len(points)
    y.append(count)

X = np.array(X, dtype='float32') / 255.0  # Normalização
y = np.array(y)

# === Separar treino/validação ===
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# === Data Augmentation offline ===
X_aug = []
y_aug = []

for img, label in zip(X_train, y_train):
    X_aug.append(img)  # original
    y_aug.append(label)

    flipped_h = cv2.flip(img, 1)  # horizontal
    X_aug.append(flipped_h)
    y_aug.append(label)

    flipped_v = cv2.flip(img, 0)  # vertical
    X_aug.append(flipped_v)
    y_aug.append(label)

    flipped_both = cv2.flip(img, -1)  # horizontal + vertical
    X_aug.append(flipped_both)
    y_aug.append(label)

X_train = np.array(X_aug)
y_train = np.array(y_aug)

print(f"✅ Dados aumentados: {len(X_train)} imagens de treino")

# === Modelo CNN ===
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(1)
])

model.compile(optimizer=Adam(learning_rate=0.001),
              loss='mean_squared_error',
              metrics=['mae'])

# === Treinar modelo ===
model.fit(X_train, y_train, epochs=20, batch_size=16, validation_data=(X_val, y_val))

# === Guardar modelo ===
model.save('contador_pessoas.h5')
print(f"Imagens de treino após augmentation: {len(X_train)}")
print(f"Imagens de validação/teste: {len(X_val)}")

