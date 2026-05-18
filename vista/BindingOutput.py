"""Schema Pydantic per vincolare l'output JSON di Qwen-VL via lm-format-enforcer."""
from typing import Literal
from pydantic import BaseModel, Field, conlist


# Tassonomia delle label ammesse, allineata al PDF della challenge.
LABELS = Literal[
    "intact car",
    "crashed car",
    "overturned car",
    "emergency vehicle",
    "person standing",
    "person walking",
    "person running",
    "person sitting",
    "person lying down",
    "person injured",
    "person helping",
    "emergency responder",
]

# Maps each structured label to its challenge category.
LABEL_TO_CATEGORY: dict[str, str] = {
    "intact car":          "car",
    "crashed car":         "car",
    "overturned car":      "car",
    "emergency vehicle":   "emergency_vehicle",
    "person standing":     "person",
    "person walking":      "person",
    "person running":      "person",
    "person sitting":      "person",
    "person lying down":   "person",
    "person injured":      "person",
    "person helping":      "person",
    "emergency responder": "person",
}


class Detection(BaseModel):
    """Una singola detection: bounding box + label di stato."""
    bbox_2d: conlist(int, min_length=4, max_length=4) = Field(
        ..., description="Bounding box [x1, y1, x2, y2] in pixel coordinates."
    )
    label: LABELS = Field(
        ..., description="Status label from the allowed taxonomy."
    )


class DetectionList(BaseModel):
    """Wrapper per lm-format-enforcer (full-frame detection mode)."""
    detections: list[Detection] = Field(default_factory=list)


class CropCaption(BaseModel):
    """Caption for a single YOLO crop (used in batch crop mode)."""
    label: LABELS = Field(..., description="Status label for this crop.")


class CropCaptionList(BaseModel):
    """Ordered list of captions for a batch of YOLO crops.

    The i-th entry corresponds to the i-th crop passed in the prompt.
    """
    captions: list[CropCaption] = Field(default_factory=list)