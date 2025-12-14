import cv2
import numpy as np
import base64
from ultralytics import YOLO

# モデルのロード（初回のみ実行されるようにグローバルで保持）
MODEL_PATH = 'models/best.pt'
try:
    model = YOLO(MODEL_PATH)
except Exception:
    model = None
    print(f"Warning: {MODEL_PATH} not found. AI features will fail.")

HARVEST_THRESHOLDS = {
    'OK_ORANGE': 25,
    'NG_GREEN': 40
}

def get_harvest_label(mean_h):
    if mean_h < HARVEST_THRESHOLDS['OK_ORANGE']:
        return "OK (Ripe Orange)", (0, 255, 0)
    elif mean_h >= HARVEST_THRESHOLDS['NG_GREEN']:
        return "NG (Unripe Green)", (0, 0, 255)
    else:
        return "Check (Yellow)", (0, 255, 255)

def analyze_kaki_image(image_bytes):
    """
    バイト列の画像を受け取り、解析結果(テキスト)と加工画像(Base64)を返す
    """
    if model is None:
        return {"error": "Model not loaded"}

    # バイナリデータをOpenCV画像に変換
    nparr = np.frombuffer(image_bytes, np.uint8)
    original_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if original_image is None:
        return {"error": "Image decode failed"}

    # YOLO推論
    results = model(source=original_image, save=False, conf=0.1, iou=0.7, verbose=False)
    result = results[0]

    if result.masks is None:
        return {"result_text": "No Kaki Detected", "processed_image": None}

    # マスクデータの取得とリサイズ
    mask = result.masks.data[0].cpu().numpy()
    mask_resized = cv2.resize(mask, (original_image.shape[1], original_image.shape[0]), interpolation=cv2.INTER_LINEAR)
    final_mask = (mask_resized > 0.5).astype('uint8') * 255

    # バウンディングボックス計算
    contours, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return {"result_text": "Mask Error", "processed_image": None}
    
    x, y, w, h = cv2.boundingRect(contours[0])

    # HSV分析
    hsv_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2HSV)
    h_values = hsv_image[:, :, 0][final_mask > 0]
    
    if len(h_values) == 0:
        return {"result_text": "HSV Error", "processed_image": None}

    mean_h = int(np.mean(h_values))
    harvest_label, text_color = get_harvest_label(mean_h)
    text_to_draw = f"{harvest_label} (H={mean_h})"

    # --- 描画処理 ---
    annotated_img = original_image.copy()
    
    # マスク部分を緑に着色
    mask_color = np.zeros_like(annotated_img, dtype=np.uint8)
    mask_color[final_mask > 0] = (0, 255, 0)
    annotated_img = cv2.addWeighted(annotated_img, 1, mask_color, 0.5, 0)

    # 枠とテキスト
    cv2.rectangle(annotated_img, (x, y), (x + w, y + h), text_color, 4)
    (text_w, text_h), _ = cv2.getTextSize(text_to_draw, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    cv2.rectangle(annotated_img, (x, y - text_h - 10), (x + text_w + 10, y), text_color, -1)
    cv2.putText(annotated_img, text_to_draw, (x + 5, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # 画像をBase64文字列に変換して返す
    _, buffer = cv2.imencode('.jpg', annotated_img)
    base64_str = base64.b64encode(buffer).decode('utf-8')

    return {
        "result_text": text_to_draw,
        "mean_h": mean_h,
        "status": harvest_label,
        "processed_image": f"data:image/jpeg;base64,{base64_str}"
    }