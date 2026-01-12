import os
import xml.etree.ElementTree as ET
import torch
from PIL import Image


class PCBDefectDataset(torch.utils.data.Dataset):
    def __init__(self, root, classes, transforms=None):
        """
        root: quality_inspection/data
        classes: ["open", "short", "mousebite"]
        """
        self.root = root
        self.transforms = transforms

        self.classes = [c.lower() for c in classes]
        self.class_to_idx = {c: i + 1 for i, c in enumerate(self.classes)}

        self.img_dir = os.path.join(root, "images")
        self.ann_dir = os.path.join(root, "annotations")

        if not os.path.isdir(self.img_dir):
            raise FileNotFoundError(self.img_dir)
        if not os.path.isdir(self.ann_dir):
            raise FileNotFoundError(self.ann_dir)

       
        self.ids = [
            os.path.splitext(f)[0]
            for f in os.listdir(self.img_dir)
            if f.lower().endswith(".jpg")
        ]

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img_id = self.ids[idx]

        img_path = os.path.join(self.img_dir, img_id + ".jpg")
        ann_path = os.path.join(self.ann_dir, img_id + ".xml")

        image = Image.open(img_path).convert("RGB")
        tree = ET.parse(ann_path)
        root = tree.getroot()

        boxes = []
        labels = []

        for obj in root.findall("object"):
            name = obj.find("name").text.lower().strip()

            # ignore unwanted DeepPCB classes
            if name not in self.class_to_idx:
                continue

            b = obj.find("bndbox")
            xmin = int(b.find("xmin").text)
            ymin = int(b.find("ymin").text)
            xmax = int(b.find("xmax").text)
            ymax = int(b.find("ymax").text)

            if xmax <= xmin or ymax <= ymin:
                continue

            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(self.class_to_idx[name])

        
        if len(boxes) == 0:
            return None

        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.int64),
        }

        if self.transforms:
            image = self.transforms(image)

        return image, target
