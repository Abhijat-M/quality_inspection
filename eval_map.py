import os
import xml.etree.ElementTree as ET
import torch
import numpy as np
from torchvision import transforms

from dataset import PCBDefectDataset
from model import build_pcb_model


DATA_ROOT = "data\\valid"
CKPT_PATH = "checkpoints\\pcb_defect.pth"

CLASSES = ["open", "short", "mousebite"]
CLASS_TO_IDX = {c: i + 1 for i, c in enumerate(CLASSES)}

IOU_THRESHOLD = 0.5
SCORE_THRESHOLD = 0.05



def compute_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

    union = area1 + area2 - inter
    return inter / union if union > 0 else 0



def voc_ap(rec, prec):
    ap = 0.0
    for t in np.arange(0.0, 1.1, 0.1):
        if np.sum(rec >= t) == 0:
            p = 0
        else:
            p = np.max(prec[rec >= t])
        ap += p / 11.0
    return ap



def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load dataset
    dataset = PCBDefectDataset(
        root=DATA_ROOT,
        classes=CLASSES,
        transforms=transforms.ToTensor()
    )

    # Load model
    model = build_pcb_model(len(CLASSES) + 1)
    model.load_state_dict(torch.load(CKPT_PATH, map_location=device))
    model.to(device).eval()

    # Ground truth cache
    gt_boxes = {c: {} for c in CLASSES}
    npos = {c: 0 for c in CLASSES}

    for idx in range(len(dataset)):
        sample = dataset[idx]
        if sample is None:
            continue

        _, target = sample
        img_id = idx

        for box, label in zip(target["boxes"], target["labels"]):
            cls = CLASSES[label - 1]
            gt_boxes.setdefault(cls, {}).setdefault(img_id, [])
            gt_boxes[cls][img_id].append({
                "bbox": box.tolist(),
                "detected": False
            })
            npos[cls] += 1

    aps = []

    # Per-class AP
    for cls in CLASSES:
        detections = []

        for idx in range(len(dataset)):
            sample = dataset[idx]
            if sample is None:
                continue

            image, _ = sample
            image = image.unsqueeze(0).to(device)

            with torch.no_grad():
                output = model(image)[0]

            for box, label, score in zip(
                output["boxes"],
                output["labels"],
                output["scores"]
            ):
                if score < SCORE_THRESHOLD:
                    continue
                if label.item() != CLASS_TO_IDX[cls]:
                    continue

                detections.append({
                    "img_id": idx,
                    "bbox": box.tolist(),
                    "score": score.item()
                })

        detections.sort(key=lambda x: -x["score"])

        tp = np.zeros(len(detections))
        fp = np.zeros(len(detections))

        for i, det in enumerate(detections):
            img_id = det["img_id"]

            if img_id not in gt_boxes[cls]:
                fp[i] = 1
                continue

            ious = [
                compute_iou(det["bbox"], gt["bbox"])
                for gt in gt_boxes[cls][img_id]
            ]

            max_iou = max(ious) if ious else 0
            max_idx = np.argmax(ious) if ious else -1

            if max_iou >= IOU_THRESHOLD:
                if not gt_boxes[cls][img_id][max_idx]["detected"]:
                    tp[i] = 1
                    gt_boxes[cls][img_id][max_idx]["detected"] = True
                else:
                    fp[i] = 1
            else:
                fp[i] = 1

        tp = np.cumsum(tp)
        fp = np.cumsum(fp)

        rec = tp / max(npos[cls], 1)
        prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)

        ap = voc_ap(rec, prec)
        aps.append(ap)

        print(f"AP ({cls}) = {ap:.4f}")

    mAP = np.mean(aps)
    print(f"\nmAP@0.5 = {mAP:.4f}")


if __name__ == "__main__":
    main()
