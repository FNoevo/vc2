import os
import cv2
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, BatchNormalization
from tensorflow.keras.applications import VGG16
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# === Caminhos ===
IMG_DIR = 'part_B/train_data/images'
DENSITY_DIR = 'part_B/train_data/density_maps'
IMG_SIZE = 224

# === Leitura dos dados ===
X, Y = [], []
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

# === Separar treino e validação ===
X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, random_state=42)

# === Modelo CSRNet ===
def build_csrnet(input_shape=(224, 224, 3)):
    vgg = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)
    for layer in vgg.layers:
        layer.trainable = False

    x = vgg.output
    for filters, dilation in zip([512, 512, 512, 256, 128, 64], [2, 2, 2, 2, 2, 2]):
        x = Conv2D(filters, (3, 3), dilation_rate=dilation, activation='relu', padding='same')(x)
        x = BatchNormalization()(x)

    output = Conv2D(1, (1, 1), activation='linear', padding='same')(x)
    model = Model(inputs=vgg.input, outputs=output)
    model.compile(optimizer=Adam(1e-4), loss=tf.keras.losses.LogCosh(), metrics=['mse'])
    return model

model = build_csrnet()

# === Data Augmentation ===
datagen = ImageDataGenerator(horizontal_flip=True, rotation_range=10, zoom_range=0.1)

# === Callbacks ===
early_stop = EarlyStopping(patience=5, restore_best_weights=True, monitor='val_loss')
checkpoint = ModelCheckpoint('melhor_csrnet.h5', save_best_only=True, monitor='val_loss', mode='min', verbose=1)
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)

# === Treino ===
history = model.fit(
    datagen.flow(X_train, Y_train, batch_size=16),
    validation_data=(X_val, Y_val),
    epochs=50,
    callbacks=[early_stop, checkpoint, lr_scheduler]
)

# === Gráfico da perda ===
plt.figure()
plt.plot(history.history['loss'], label='Train LogCosh')
plt.plot(history.history['val_loss'], label='Val LogCosh')
plt.xlabel('Épocas')
plt.ylabel('Erro')
plt.title('Evolução da Perda (LogCosh)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("grafico_csrnet.png")
plt.show()
