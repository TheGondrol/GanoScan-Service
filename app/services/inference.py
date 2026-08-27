"""Model loading, preprocessing and prediction.

Loading priority:
  1. model_jit.pt      (TorchScript, recommended)   -> mode "torchscript"
  2. best_model.pth    (state_dict via GanodermaCNN) -> mode "state_dict"
  3. no files present  -> mode "random"  (random predictions, runs with no model)

Random mode lets the whole stack run and be integrated *before* any model is
trained. torch / opencv are imported lazily, so random mode needs only
FastAPI + Pillow + NumPy.
"""

import io
import json

import numpy as np
from PIL import Image

from app.core.config import settings


class GanodermaModel:
    def __init__(self):
        self.classes = list(settings.default_classes)
        self.image_size = tuple(settings.default_image_size)  # (H, W)
        self.backbone = settings.default_backbone
        self.accuracy = None
        self.mode = "random"
        self._model = None
        self._device = "cpu"

        self._load_model_info()
        self._load_weights()

    # ---- loading -------------------------------------------------------
    def _load_model_info(self):
        info_path = settings.model_dir_path / settings.model_info_name
        if not info_path.exists():
            return
        try:
            info = json.loads(info_path.read_text())
            self.classes = info.get("classes", self.classes)
            img = info.get("image_size", self.image_size)
            self.image_size = (int(img[0]), int(img[1]))
            self.backbone = info.get("backbone", self.backbone)
            self.accuracy = info.get("test_accuracy")
        except Exception as e:  # noqa: BLE001
            print(f"[model] could not read {settings.model_info_name}: {e}")

    def _load_weights(self):
        jit_path = settings.model_dir_path / settings.jit_model_name
        state_path = settings.model_dir_path / settings.state_dict_name

        if jit_path.exists():
            try:
                import torch

                self._device = "cuda" if torch.cuda.is_available() else "cpu"
                self._model = torch.jit.load(str(jit_path), map_location=self._device)
                self._model.eval()
                self.mode = "torchscript"
                print(f"[model] loaded TorchScript from {jit_path} on {self._device}")
                return
            except Exception as e:  # noqa: BLE001
                print(f"[model] failed to load TorchScript ({e}); trying state_dict")

        if state_path.exists():
            try:
                import torch

                from app.services.model_arch import GanodermaCNN

                self._device = "cuda" if torch.cuda.is_available() else "cpu"
                model = GanodermaCNN(
                    num_classes=len(self.classes),
                    backbone=self.backbone,
                    pretrained=False,
                )
                state = torch.load(str(state_path), map_location=self._device)
                if isinstance(state, dict) and "model_state_dict" in state:
                    state = state["model_state_dict"]
                model.load_state_dict(state)
                model.to(self._device).eval()
                self._model = model
                self.mode = "state_dict"
                print(f"[model] loaded state_dict from {state_path} on {self._device}")
                return
            except Exception as e:  # noqa: BLE001
                print(f"[model] failed to load state_dict ({e}); using random mode")

        print("[model] no weights found -> RANDOM mode (drop model files into "
              f"{settings.model_dir_path} and restart for real predictions)")

    @property
    def is_random(self) -> bool:
        return self._model is None

    # ---- prediction ----------------------------------------------------
    def _preprocess(self, image_bytes):
        """Match the notebook predictor exactly: BGR->RGB, resize (bilinear),
        ImageNet normalize, CHW float tensor."""
        import cv2
        import torch

        arr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)  # BGR
        if img is None:
            raise ValueError("could not decode image")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = self.image_size
        img = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)
        img = img.astype(np.float32) / 255.0
        img = (img - np.array(settings.imagenet_mean, np.float32)) / np.array(settings.imagenet_std, np.float32)
        tensor = torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0)
        return tensor.to(self._device)

    def _random_probs(self):
        """Fresh random probabilities each call (no model required). Slight bias
        toward 'healthy' so demo data skews realistic rather than uniform."""
        rng = np.random.default_rng()  # seeded from OS entropy every call
        logits = rng.standard_normal(len(self.classes)) * 1.6
        logits[0] += 0.4
        exp = np.exp(logits - logits.max())
        return exp / exp.sum()

    def predict(self, image_bytes):
        """Return (probabilities dict, predicted_class, confidence)."""
        if self._model is None:
            probs = self._random_probs()
        else:
            import torch

            tensor = self._preprocess(image_bytes)
            with torch.no_grad():
                out = self._model(tensor)
                probs = torch.softmax(out, dim=1)[0].cpu().numpy()

        idx = int(np.argmax(probs))
        predicted = self.classes[idx]
        confidence = float(probs[idx])
        prob_map = {cls: float(p) for cls, p in zip(self.classes, probs)}
        return prob_map, predicted, confidence

    def image_dimensions(self, image_bytes):
        try:
            with Image.open(io.BytesIO(image_bytes)) as im:
                return im.size  # (w, h)
        except Exception:  # noqa: BLE001
            return None

    def info(self):
        return {
            "mode": self.mode,
            "loaded": self._model is not None,
            "backbone": self.backbone,
            "classes": self.classes,
            "imageSize": list(self.image_size),
            "device": self._device,
            "testAccuracy": self.accuracy,
        }


# ---- module-level singleton -------------------------------------------------
_model: GanodermaModel | None = None


def get_model() -> GanodermaModel:
    global _model
    if _model is None:
        _model = GanodermaModel()
    return _model
