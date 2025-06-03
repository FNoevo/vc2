import cv2, os
import numpy as np

# ───── parâmetros ─────
BLUR_KSIZE   = (5,5)     # kernel do GaussianBlur
SIGMA        = 0.33      # para auto_canny
DILATE_ITERS = 1         # para “abrir” ligeiramente as bordas
ERODE_ITERS  = 1
# ──────────────────────

INPUT  = os.path.join(os.path.dirname(__file__), "..", "imagens")
OUTPUT = os.path.join(os.path.dirname(__file__), "..", "resultados", "edges_adaptive")
os.makedirs(OUTPUT, exist_ok=True)

def auto_canny(gray, sigma=SIGMA):
    v = np.median(gray)
    low  = int(max(0, (1.0 - sigma)*v))
    high = int(min(255, (1.0 + sigma)*v))
    return cv2.Canny(gray, low, high)

for fn in os.listdir(INPUT):
    if not fn.lower().endswith((".jpg","jpeg","png")): continue
    img = cv2.imread(os.path.join(INPUT, fn))
    if img is None: continue

    orig_h, orig_w = img.shape[:2]

    # ——— 1) escolha de ROI (automaticamente ou interactivamente) ———
    ROIS = {
      "tigre3.jpg":    (50, 50, 500, 300),
      "ponteee.jpg":   (100,100,800,400),
    }
    # 1) define e limite a ROI(x e y, coordenadas canto superior esquerdo do ROI e w,h são altura e largura do ROI)
    x, y, w, h = 0, 0, orig_w, orig_h
    x = max(0, min(x, orig_w-1))
    y = max(0, min(y, orig_h-1))
    w = max(1, min(w, orig_w - x))
    h = max(1, min(h, orig_h - y))
    sub = img[y:y+h, x:x+w]

    # ——— 2) blur para reduzir ruído ———
    gray = cv2.cvtColor(sub, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, BLUR_KSIZE, 0)

    # ——— 3) detecção de bordas adaptativa ———
    edges = auto_canny(blurred)

    # ——— 4) limpeza morfológica ———
    if DILATE_ITERS>0:
        edges = cv2.dilate(edges, None, iterations=DILATE_ITERS)
    if ERODE_ITERS>0:
        edges = cv2.erode(edges, None, iterations=ERODE_ITERS)

    # ——— 5) coloca o resultado de volta na imagem inteira (preto no resto) ———
    canvas = np.zeros_like(img)  # canvas do tamanho da imagem original
    canvas[y:y+h, x:x+w] = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

    out = os.path.join(OUTPUT, fn.replace(".", "_edges."))
    cv2.imwrite(out, canvas)
    print(f"✅ {fn} → {out}")
