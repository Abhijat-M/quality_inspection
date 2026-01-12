


# Task 2: Automated Quality Inspection System (PCB Defects)

## Overview

This project implements an **automated visual quality inspection system** for printed circuit boards (PCBs).
The system detects, localizes, and classifies manufacturing defects using a deep learning–based object detection model. In addition to defect classification, the system estimates defect severity and reports precise spatial coordinates.

The design mirrors **real industrial inspection pipelines**, emphasizing robustness, interpretability, and reproducibility.

---

## Manufactured Item

**Printed Circuit Boards (PCBs)**

PCBs are fundamental components in electronic systems, where even minor defects can cause functional failure. Automated inspection systems are widely used in industry to replace manual visual inspection.

---

## Defect Types

Three representative PCB defect categories were selected from the DeepPCB dataset:

* **Open Circuit (`open`)**
  Broken or missing copper traces causing interrupted electrical connections.

* **Short Circuit (`short`)**
  Unintended electrical connections between traces.

* **Mousebite (`mousebite`)**
  Small irregular edge defects introduced during manufacturing.

Other defect types present in the dataset were intentionally ignored to reduce class imbalance and focus on the most common and critical failure modes.

---

## Dataset

* **Dataset:** DeepPCB
* **Source:** Roboflow
* **License:** CC BY 4.0
* **Annotation format:** Pascal VOC (XML)
* **Image resolution:** 640 × 640
* **Annotations:** Bounding boxes per defect instance

Although DeepPCB contains six defect categories, only three were used to satisfy task requirements and simplify evaluation.

---

## Model Architecture

The inspection system is based on **Faster R-CNN**, a two-stage object detection framework.

* **Backbone:** Custom CNN backbone (reused from Task 1)
* **Region Proposal Network (RPN):**

  * Anchor sizes: 16, 32, 64, 128
  * Aspect ratios: 0.5, 1.0, 2.0
* **Detection head:** Classification and bounding-box regression

This architecture provides high localization accuracy, which is essential for detecting small PCB defects.

---

## Training Configuration

* **Framework:** PyTorch
* **Optimizer:** SGD
* **Learning rate:** 0.005
* **Momentum:** 0.9
* **Weight decay:** 0.0005
* **Batch size:** 2
* **Epochs:** 30

Training was performed using bounding-box supervision on the DeepPCB dataset.

---

## Inference Output

For each detected defect, the system reports:

* Defect type
* Confidence score
* Center coordinates (x, y) in pixel space
* Estimated severity level

### Severity Estimation

Severity is estimated based on the relative size of the defect:

severity_ratio = defect_area / image_area

Severity levels:

* **Low:** ratio < 0.01
* **Medium:** 0.01 ≤ ratio < 0.05
* **High:** ratio ≥ 0.05

This heuristic is simple, interpretable, and commonly used in manufacturing quality control.

---

## Evaluation Metric

Performance is evaluated using **mean Average Precision (mAP@0.5 IoU)** following the Pascal VOC protocol.

* **IoU threshold:** 0.5
* **Evaluation split:** Held-out validation dataset

---

## Validation Results

AP (open) = 0.9014
AP (short) = 0.7954
AP (mousebite) = 0.9012

mAP@0.5 = **0.8660**

### Interpretation

* Open-circuit and mousebite defects achieve high precision due to their distinct visual patterns.
* Short-circuit defects show lower AP, which is expected due to their small size and visual similarity to normal copper traces.
* The overall validation mAP@0.5 of **0.866** indicates strong detection and localization performance with realistic generalization.

---

## How to Run

All scripts are executed as Python modules.

### Training

python -m quality_inspection.train

### Inference

python -m quality_inspection.detect_defects

### Evaluation (mAP)

python -m quality_inspection.eval_map

---

## Project Structure

quality_inspection/
├── checkpoints/
├── data/
│   ├── images/
│   └── annotations/
├── models/
│   └── backbone.py
├── dataset.py
├── model.py
├── train.py
├── detect_defects.py
└── eval_map.py

---

## Key Takeaways

* Implements a complete **automated PCB defect inspection pipeline**
* Performs defect detection, localization, classification, and severity estimation
* Uses industry-standard evaluation metrics (mAP@0.5)
* Designed with clean package structure and reproducible execution

---

## Conclusion

This task demonstrates a production-style visual inspection system for manufacturing quality control. The approach combines robust object detection, interpretable severity estimation, and standard evaluation protocols, making it suitable for real-world industrial deployment scenarios.


