import cv2, os
from PIL import Image, ExifTags
import numpy as np

# ─────────── PARÂMETROS ───────────
RESIZE_MAX_DIM = 512
CONTRAST = 1.3
BRIGHTNESS = 10
SHARPEN_AMOUNT = 1.0
# ───────────────────────────────────

input_folder  = os.path.join(os.path.dirname(__file__), "..", "imagens")
output_folder = os.path.join(os.path.dirname(__file__), "..", "resultados", "geometric_logic")
os.makedirs(output_folder, exist_ok=True) #cria a pasta de saida se nao existir

def read_image_with_orientation(path):
    img_pil = Image.open(path)
    try:
        for orientation in ExifTags.TAGS.keys():
            if ExifTags.TAGS[orientation] == 'Orientation':
                break
        exif = img_pil._getexif()
        if exif is not None:
            orientation_value = exif.get(orientation, None)
            if orientation_value == 3:
                img_pil = img_pil.rotate(180, expand=True)
            elif orientation_value == 6:
                img_pil = img_pil.rotate(270, expand=True)
            elif orientation_value == 8:
                img_pil = img_pil.rotate(90, expand=True)
    except Exception:
        pass
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

# Kernel de nitidez
kernel = np.array([[0, -1, 0],
                   [-1, 5 + SHARPEN_AMOUNT, -1],
                   [0, -1, 0]], dtype=np.float32)

for fn in os.listdir(input_folder): #carrega imagem com orientacao corrigida, redimensiona(512) aplica nitidez, constraste e brilho
    if not fn.lower().endswith((".jpg",".png",".jpeg")): continue
    img = read_image_with_orientation(os.path.join(input_folder, fn))
    if img is None:
        print(f"⚠️ Erro a carregar {fn}")
        continue

    # resize mantendo proporção
    h, w = img.shape[:2]
    scale = RESIZE_MAX_DIM / max(h, w)
    resized = cv2.resize(img, (int(w*scale), int(h*scale)))

    # Melhoria: nitidez, contraste e brilho
    sharpened = cv2.filter2D(resized, -1, kernel)
    improved = cv2.convertScaleAbs(sharpened, alpha=CONTRAST, beta=BRIGHTNESS)

    # Operação lógica: inverter as cores da imagem melhorada
    logic_result = cv2.bitwise_not(improved)

    # Comparação lado a lado
    comparison = np.hstack([resized, logic_result])

    name = os.path.splitext(fn)[0]
    cv2.imwrite(os.path.join(output_folder, f"{name}_logic.jpg"), logic_result)
    cv2.imwrite(os.path.join(output_folder, f"{name}_compare.jpg"), comparison)
    print(f"✅ {fn} → {output_folder}") #mostra o nome do arquivo processado e onde foi salvo
