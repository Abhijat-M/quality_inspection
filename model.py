from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator

from models.backbone import CustomBackbone


def build_pcb_model(num_classes):
    backbone = CustomBackbone()

    anchor_generator = AnchorGenerator(
        sizes=((16, 32, 64, 128),),
        aspect_ratios=((0.5, 1.0, 2.0),)
    )

    model = FasterRCNN(
        backbone=backbone,
        num_classes=num_classes,
        rpn_anchor_generator=anchor_generator
    )

    return model
