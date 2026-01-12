import os
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from dataset import PCBDefectDataset
from model import build_pcb_model


DATA_ROOT = "data"

CLASSES = ["open", "short", "mousebite"]
NUM_CLASSES = len(CLASSES) + 1

EPOCHS = 30
BATCH_SIZE = 2
LR = 0.005


def collate_fn(batch):
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return [], []
    return tuple(zip(*batch))


def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    steps = 0

    for images, targets in loader:
        if len(images) == 0:
            continue

        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        loss = sum(loss_dict.values())

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        steps += 1

    return total_loss / max(steps, 1)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    transform = transforms.ToTensor()

    dataset = PCBDefectDataset(
        root=DATA_ROOT,
        classes=CLASSES,
        transforms=transform
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn
    )

    model = build_pcb_model(NUM_CLASSES).to(device)

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=LR,
        momentum=0.9,
        weight_decay=0.0005
    )

    os.makedirs("/checkpoints", exist_ok=True)

    for epoch in range(EPOCHS):
        loss = train_one_epoch(model, loader, optimizer, device)
        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {loss:.4f}")
        torch.save(
            model.state_dict(),
            "/checkpoints/pcb_defect.pth"
        )


if __name__ == "__main__":
    main()
