import os
import cv2
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, BatchNormalization, ReLU, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
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

    if dens is None or dens.size == 0:
        print(f"⚠️ Erro ao carregar mapa: {filename}")
        continue

    dens = cv2.resize(dens, (IMG_SIZE // 8, IMG_SIZE // 8))  # CSRNet output: 1/8 da resolução

    X.append(img)
    Y.append(dens)

X = np.array(X)
Y = np.expand_dims(np.array(Y), axis=-1)  # output shape (H, W, 1)

# === Separar treino e validação ===
X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, random_state=42)

# === Modelo inspirado em CSRNet com dilated convolutions e Dropout ===
def build_density_model(input_shape=(224, 224, 3)):
    inputs = Input(shape=input_shape)

    x = Conv2D(64, (3, 3), padding='same')(inputs)
    x = ReLU()(x)
    x = Conv2D(64, (3, 3), padding='same')(x)
    x = ReLU()(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Conv2D(128, (3, 3), padding='same')(x)
    x = ReLU()(x)
    x = Conv2D(128, (3, 3), padding='same')(x)
    x = ReLU()(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Conv2D(256, (3, 3), padding='same')(x)
    x = ReLU()(x)
    x = Conv2D(256, (3, 3), padding='same')(x)
    x = ReLU()(x)
    x = Conv2D(256, (3, 3), padding='same')(x)
    x = ReLU()(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Conv2D(512, (3, 3), dilation_rate=2, padding='same')(x)
    x = ReLU()(x)
    x = Dropout(0.3)(x)
    x = Conv2D(512, (3, 3), dilation_rate=2, padding='same')(x)
    x = ReLU()(x)
    x = Dropout(0.3)(x)
    x = Conv2D(512, (3, 3), dilation_rate=2, padding='same')(x)
    x = ReLU()(x)

    output = Conv2D(1, (1, 1), activation='linear', padding='same')(x)

    model = Model(inputs, output)
    model.compile(optimizer=Adam(1e-4), loss='mae')
    return model

# === Instanciar e treinar o modelo ===
model = build_density_model()
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-7)

history = model.fit(
    X_train, Y_train,
    validation_data=(X_val, Y_val),
    epochs=100,
    batch_size=8,
    callbacks=[early_stop, reduce_lr],
    verbose=1
)

# === Guardar modelo ===
model.save('modelo_densitymap.keras')

# === Gráfico da perda ===
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Evolução da Perda - Mapa de Densidade (MAE)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
