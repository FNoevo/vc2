import cv2, numpy as np, os
from PIL import Image, ExifTags

# ─────────── PARÂMETROS ───────────
SHARPEN_AMOUNT = 1.0    # [experimenta 0.5–2.0]
CONTRAST = 1.2
BRIGHTNESS = 10
# ───────────────────────────────────

# constrói kernel de nitidez
kernel = np.array([[0, -1, 0],
                   [-1, 5 + SHARPEN_AMOUNT, -1],
                   [0, -1, 0]], dtype=np.float32)

input_folder  = os.path.join(os.path.dirname(__file__), "..", "imagens")
output_folder = os.path.join(os.path.dirname(__file__), "..", "resultados", "enhancement")
os.makedirs(output_folder, exist_ok=True)

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
    except Exception as e:
        pass
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

for fn in os.listdir(input_folder):
    if not fn.lower().endswith((".jpg",".png",".jpeg")): continue
    path = os.path.join(input_folder, fn)
    img = read_image_with_orientation(path)
    if img is None:
        print(f"Erro a ler {fn}")
        continue

    # Aplica nitidez
    sharpened = cv2.filter2D(img, -1, kernel)

    # Ajusta contraste e brilho
    enhanced = cv2.convertScaleAbs(sharpened, alpha=CONTRAST, beta=BRIGHTNESS)

    # Guarda imagem melhorada
    name = os.path.splitext(fn)[0]
    cv2.imwrite(f"{output_folder}/{name}_enhanced.jpg", enhanced)

    # (Opcional) Guarda comparação lado a lado
    comparison = np.hstack([img, enhanced])
    cv2.imwrite(f"{output_folder}/{name}_compare.jpg", comparison)
