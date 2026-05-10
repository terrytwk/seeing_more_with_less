import numpy as np
import torch
from PIL import Image
from transformers import DetrImageProcessor, DetrForObjectDetection

try:
    from inference.model_loading import resolve_pretrained_reference
except ImportError:
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from inference.model_loading import resolve_pretrained_reference


def resolve_device(device):
    if device != "auto":
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class DETRRunner:
    def __init__(self, model_path, device="auto", threshold=0.5):
        self.device = resolve_device(device)
        model_path = resolve_pretrained_reference(
            model_path,
            model_name="DETR",
            recommended_id="facebook/detr-resnet-101",
        )
        self.processor = DetrImageProcessor.from_pretrained(model_path)
        self.model = DetrForObjectDetection.from_pretrained(model_path).to(self.device)
        self.model.eval()
        self.threshold = threshold

    @torch.no_grad()
    def predict(self, image):
        """
        image: H x W x 3 numpy array (RGB) or PIL Image.
        Returns list of {"label": int, "score": float, "bbox": [x, y, w, h]}.
        Label integers are COCO category IDs (HuggingFace DETR label n == COCO category ID n).
        """
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        w, h = image.size
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        outputs = self.model(**inputs)
        target_sizes = torch.tensor([[h, w]], device=self.device)
        results = self.processor.post_process_object_detection(
            outputs, threshold=self.threshold, target_sizes=target_sizes
        )[0]
        detections = []
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            x1, y1, x2, y2 = box.tolist()
            detections.append({
                "label": int(label),
                "score": float(score),
                "bbox": [x1, y1, x2 - x1, y2 - y1],
            })
        return detections
