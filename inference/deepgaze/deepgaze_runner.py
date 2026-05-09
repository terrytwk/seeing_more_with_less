import numpy as np
import torch
from scipy.ndimage import zoom
from scipy.special import logsumexp


def resolve_device(device):
    if device != "auto":
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def prepare_centerbias(image_shape, centerbias_path=None):
    height, width = image_shape[:2]
    if centerbias_path:
        centerbias_template = np.load(centerbias_path)
    else:
        centerbias_template = np.zeros((height, width), dtype=np.float64)

    if centerbias_template.shape != (height, width):
        centerbias = zoom(
            centerbias_template,
            (height / centerbias_template.shape[0], width / centerbias_template.shape[1]),
            order=0,
            mode="nearest",
        )
    else:
        centerbias = centerbias_template.astype(np.float64, copy=False)

    centerbias = np.asarray(centerbias, dtype=np.float64)
    centerbias -= logsumexp(centerbias)
    return centerbias


class DeepGazeIIERunner:
    def __init__(self, device="auto"):
        import deepgaze_pytorch

        self.device = resolve_device(device)
        self.model = deepgaze_pytorch.DeepGazeIIE(pretrained=True)
        if self.device.type == "mps":
            # MPS does not support float64 tensors. DeepGaze includes some
            # float64 buffers, so convert them before moving the model.
            self.model = self.model.float()
        self.model = self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def predict_log_density(self, image, centerbias_path=None):
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("image must be an RGB array with shape HxWx3")

        centerbias = prepare_centerbias(image.shape, centerbias_path=centerbias_path)
        image_batch = np.expand_dims(np.ascontiguousarray(image.transpose(2, 0, 1)), axis=0)
        centerbias_batch = np.expand_dims(centerbias, axis=0)
        image_tensor = torch.from_numpy(image_batch).to(dtype=torch.float32, device=self.device)
        centerbias_tensor = torch.from_numpy(centerbias_batch).to(dtype=torch.float32, device=self.device)

        prediction = self.model(image_tensor, centerbias_tensor)
        return prediction.detach().cpu().numpy()[0, 0]

    def predict_probability(self, image, centerbias_path=None):
        log_density = self.predict_log_density(image, centerbias_path=centerbias_path)
        log_density = log_density - logsumexp(log_density)
        return np.exp(log_density)
