import os
import json
import torch
import cv2
from torchvision import transforms

from model import build_pcb_model


BASE_DIR = os.path.dirname(os.path.abspath(__file__))


DATA_DIR = os.path.join(BASE_DIR, "data", "images")
CKPT_DIR = os.path.join(BASE_DIR, "checkpoints")

# CHANGE THIS to an existing image filename
IMAGE_NAME = "77000051_test_jpg.rf.1749d9aaf89a6f18ab4f112ea408699a.jpg"

IMAGE_PATH = os.path.join(DATA_DIR, IMAGE_NAME)
MODEL_PATH = os.path.join(CKPT_DIR, "pcb_defect.pth")


CLASSES = ["__bg__", "open", "short", "mousebite"]
SCORE_THRESHOLD = 0.4


def main():
  
    if not os.path.isfile(IMAGE_PATH):
        raise FileNotFoundError(f"Image not found: {IMAGE_PATH}")

    if not os.path.isfile(MODEL_PATH):
        raise FileNotFoundError(f"Model checkpoint not found: {MODEL_PATH}")

    device = "cuda" if torch.cuda.is_available() else "cpu"

  
    model = build_pcb_model(num_classes=len(CLASSES))
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()

    transform = transforms.ToTensor()


    img = cv2.imread(IMAGE_PATH)
    if img is None:
        raise RuntimeError(f"cv2.imread failed for: {IMAGE_PATH}")

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w, _ = img.shape

    
    with torch.no_grad():
        outputs = model(transform(img_rgb).unsqueeze(0).to(device))[0]

 
    results = []

    for box, label, score in zip(
        outputs["boxes"],
        outputs["labels"],
        outputs["scores"]
    ):
        if score < SCORE_THRESHOLD:
            continue

        x1, y1, x2, y2 = map(int, box.tolist())
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)

        area = max(0, x2 - x1) * max(0, y2 - y1)
        ratio = area / float(w * h)

        if ratio < 0.01:
            severity = "low"
        elif ratio < 0.05:
            severity = "medium"
        else:
            severity = "high"

        results.append({
            "defect_type": CLASSES[label],
            "confidence": round(float(score), 3),
            "center": [cx, cy],
            "severity": severity
        })


    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
