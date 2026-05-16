import torch
from PIL import Image
from io import BytesIO
import base64


def set_seed(seed: int):
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        

# Useless categories to ignore with respect to emergency detection
IGNORE_CATEGORIES = {
    "dog",
    "cat"
    "parking meter",
    "bench",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
}

# ============================================================
# Image utilities
# ============================================================

def resize_image(image: Image.Image, target_size: int) -> Image.Image:
    w, h = image.size
    if max(w, h) <= target_size:
        return image
    scale = target_size / max(w, h)
    return image.resize((int(w * scale), int(h * scale)))

def image_to_base64(image: Image.Image) -> str:
    buf = BytesIO()
    image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")

# ============================================================
# Logging
# ============================================================

def log(msg: str):
    print(f"[INFO] {msg}", flush=True)


def get_emergency_level(label: str) -> int:
    # Define keywords for emergency levels, 3 is highest
    # 1 should include people helping others, 2 
    emergency_keywords = {
        "fire": 3,
        "smoke": 3,
        "collapsed": 3,
        "injured": 3,
        "help": 2,
        "accident": 2,
        "needing": 2,
        "flood": 3,
        "trapped": 3,
        "danger": 2,
        "emergency": 3,
        "helping": 1,
        "rescue": 1,
        "calling": 3,
    }
    # return max level found in label
    max_level = 0
    for keyword, level in emergency_keywords.items():
        if keyword in label.lower():
            max_level = max(max_level, level)
    return max_level
