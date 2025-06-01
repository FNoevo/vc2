import os
import cv2
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

# === Caminhos ===
IMG_DIR = 'part_B/train_data/images'
DENSITY_DIR = 'part_B/train_data/density_maps'

IMG_SIZE = 224
X = []
Y = []

# === Leitura de imagens e mapas de densidade ===
for filename in tqdm(sorted(os.listdir(IMG_DIR))):
    if not filename.endswith('.jpg'):
        continue

    img_path = os.path.join(IMG_DIR, filename)
    den_path = os.path.join(DENSITY_DIR, filename.replace('.jpg', '.npy'))

    if not os.path.exists(den_path):
        continue

    img = cv2.imread(img_path)
    if img is None:
        continue

    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE)).astype('float32') / 255.0
    dens = np.load(den_path)

    X.append(img)
    Y.append(dens)

X = np.array(X)
Y = np.array(Y)
Y = np.expand_dims(Y, axis=-1)  # output shape (H, W, 1)

# === Separar treino e validação ===
X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, random_state=42)

# === Modelo Encoder-Decoder (tipo mini U-Net) ===
def build_density_model(input_shape=(224, 224, 3)):
    inputs = Input(shape=input_shape)

    # Encoder
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)

    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)

    x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)

    x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)

    # Decoder
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)

    x = UpSampling2D((2, 2))(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)

    x = UpSampling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)

    outputs = Conv2D(1, (1, 1), activation='linear', padding='same')(x)

    model = Model(inputs, outputs)
    model.compile(optimizer=Adam(1e-4), loss='mse')
    return model


# === Guardar modelo ===
model.save('modelo_densitymap.keras')

# === Gráfico da perda ===
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Evolução da Perda - Mapa de Densidade')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
