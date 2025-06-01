import cv2, os

# ─────────── PARÂMETROS ───────────
USE_OTSU       = True
FIXED_THRESH   = 127     # usado se USE_OTSU=False
RESIZE_MAX_DIM = 512
# ───────────────────────────────────

input_folder  = os.path.join(os.path.dirname(__file__), "..", "imagens")
output_folder = os.path.join(os.path.dirname(__file__), "..", "resultados", "segmentation")
os.makedirs(output_folder, exist_ok=True)

for fn in os.listdir(input_folder):
    if not fn.lower().endswith((".jpg",".png",".jpeg")): continue
    img = cv2.imread(os.path.join(input_folder, fn), cv2.IMREAD_GRAYSCALE)
    if img is None:
        continue

    # redimensionar
    h, w = img.shape
    scale = RESIZE_MAX_DIM / max(h, w)
    img_resized = cv2.resize(img, (int(w*scale), int(h*scale)))

    # Segmentação
    if USE_OTSU:
        _, mask = cv2.threshold(img_resized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        _, mask = cv2.threshold(img_resized, FIXED_THRESH, 255, cv2.THRESH_BINARY)

    # Comparação lado a lado
    mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    compare = cv2.hconcat([cv2.cvtColor(img_resized, cv2.COLOR_GRAY2BGR), mask_bgr])

    name = os.path.splitext(fn)[0]
    cv2.imwrite(f"{output_folder}/{name}_segmented.jpg", mask)
    cv2.imwrite(f"{output_folder}/{name}_compare.jpg", compare)
    print(f"✅ {fn} segmentado e comparação guardada.")
