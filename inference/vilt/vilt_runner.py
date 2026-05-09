import numpy as np
import torch
from PIL import Image
from transformers import ViltProcessor, ViltForQuestionAnswering


def resolve_device(device):
    if device != "auto":
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class ViLTRunner:
    def __init__(self, model_path, device="auto"):
        self.device = resolve_device(device)
        self.processor = ViltProcessor.from_pretrained(model_path)
        self.model = ViltForQuestionAnswering.from_pretrained(model_path).to(self.device)
        self.model.eval()

    @torch.no_grad()
    def predict(self, image, question):
        """
        image: H x W x 3 numpy array (RGB) or PIL Image.
        question: string.
        Returns the top predicted answer string.
        """
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        inputs = self.processor(image, question, return_tensors="pt").to(self.device)
        outputs = self.model(**inputs)
        predicted_idx = outputs.logits.argmax(-1).item()
        return self.model.config.id2label[predicted_idx]
