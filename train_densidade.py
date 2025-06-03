import os
import cv2
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, BatchNormalization, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
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
    dens = np.load(den_path).astype('float32')

    if dens.shape != (IMG_SIZE, IMG_SIZE):
        dens = cv2.resize(dens, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_CUBIC)

    X.append(img)
    Y.append(dens)

X = np.array(X)
Y = np.expand_dims(np.array(Y), axis=-1)

print(f"✅ Dados carregados: {X.shape[0]} imagens")
print(f"🧠 Shape X: {X.shape}, Shape Y: {Y.shape}")
print(f"📊 Soma média de mapas: {np.mean([np.sum(y) for y in Y]):.2f}")

# === Separar treino e validação ===
X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, random_state=42)

# === Modelo melhorado com Dropout ===
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
    x = Dropout(0.3)(x)

    x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)

    # Decoder
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)

    x = UpSampling2D((2, 2))(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)

    x = UpSampling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)

    outputs = Conv2D(1, (1, 1), activation='linear', padding='same')(x)

    model = Model(inputs, outputs)
    model.compile(optimizer=Adam(1e-4), loss='mae', metrics=['mse'])
    return model

# === Criar modelo ===
model = build_density_model()

# === Callbacks ===
early_stop = EarlyStopping(patience=5, restore_best_weights=True, monitor='val_loss')
checkpoint = ModelCheckpoint('melhor_modelo.h5', save_best_only=True, monitor='val_loss', mode='min', verbose=1)
print("Saída esperada:", Y_train.shape)
print("Saída do modelo:", model.output_shape)

# === Treinar modelo ===
history = model.fit(
    X_train, Y_train,
    validation_data=(X_val, Y_val),
    epochs=50,
    batch_size=16,
    callbacks=[early_stop, checkpoint]
)

print("✅ Modelo treinado. O melhor foi guardado como 'melhor_modelo.h5'.")

# === Gráfico da perda ===
plt.figure()
plt.plot(history.history['loss'], label='Train MAE')
plt.plot(history.history['val_loss'], label='Val MAE')
plt.xlabel('Épocas')
plt.ylabel('Erro (MAE)')
plt.title('Evolução da Perda MAE')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("grafico_perda.png")
plt.show()
