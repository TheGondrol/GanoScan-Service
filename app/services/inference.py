"""Model loading, preprocessing and prediction.

Pipeline (mirrors the training notebook's inference_single_image() exactly,
including the random gamma draw on every call):

  decode -> BGR->RGB -> resize                        (stage 1: "Citra Asli")
  -> gamma correction, gamma ~ random.choice(settings.gamma_values)
                                                        (stage 2: "Gamma Correction")
  -> autoencoder enhancement, frozen, no_grad          (stage 3: "CNN-Based Enhancement")
  -> ImageNet normalize -> CHW tensor -> classifier -> softmax
                                                        (stage 4: "Klasifikasi (CNN)")

Loading priority, independently for each of the two models:
  1. classifier_jit.pt / autoencoder_jit.pt      (TorchScript, recommended)
  2. classifier_best.pth / autoencoder_best.pth  (state_dict via model_arch)
  3. missing -> that component is None

Real mode requires the classifier. The autoencoder is optional: if it fails
to load, inference still runs (the classifier sees the gamma-corrected image
without enhancement) but a clear warning is logged and the response's
"CNN-Based Enhancement" stage notes it was skipped — accuracy will be lower
than what the model was evaluated at, since the classifier was only ever
trained on AE-enhanced images.

Random mode (no classifier files found) lets the whole stack run and be
integrated *before* any model is trained. torch / opencv are imported lazily,
so random mode needs only FastAPI + Pillow + NumPy.
"""

import base64
import io
import json
import random

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
        self._model = None  # classifier
        self._ae_model = None  # autoencoder (optional)
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
        self._load_classifier()
        if self._model is not None:
            self._load_autoencoder()

    def _load_classifier(self):
        jit_path = settings.model_dir_path / settings.classifier_jit_name
        state_path = settings.model_dir_path / settings.classifier_state_name

        if jit_path.exists():
            try:
                import torch

                self._device = "cuda" if torch.cuda.is_available() else "cpu"
                self._model = torch.jit.load(str(jit_path), map_location=self._device)
                self._model.eval()
                self.mode = "torchscript"
                print(f"[model] loaded classifier TorchScript from {jit_path} on {self._device}")
                return
            except Exception as e:  # noqa: BLE001
                print(f"[model] failed to load classifier TorchScript ({e}); trying state_dict")

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
                print(f"[model] loaded classifier state_dict from {state_path} on {self._device}")
                return
            except Exception as e:  # noqa: BLE001
                print(f"[model] failed to load classifier state_dict ({e}); using random mode")

        print(
            "[model] no classifier weights found -> RANDOM mode (drop model files into "
            f"{settings.model_dir_path} and restart for real predictions)"
        )

    def _load_autoencoder(self):
        """Optional: real mode still works without it, just skips enhancement."""
        jit_path = settings.model_dir_path / settings.autoencoder_jit_name
        state_path = settings.model_dir_path / settings.autoencoder_state_name

        if jit_path.exists():
            try:
                import torch

                self._ae_model = torch.jit.load(str(jit_path), map_location=self._device)
                self._ae_model.eval()
                print(f"[model] loaded autoencoder TorchScript from {jit_path} on {self._device}")
                return
            except Exception as e:  # noqa: BLE001
                print(f"[model] failed to load autoencoder TorchScript ({e}); trying state_dict")

        if state_path.exists():
            try:
                import torch

                from app.services.model_arch import EnhancementAutoencoderV2

                model = EnhancementAutoencoderV2()
                state = torch.load(str(state_path), map_location=self._device)
                if isinstance(state, dict) and "model_state_dict" in state:
                    state = state["model_state_dict"]
                model.load_state_dict(state)
                model.to(self._device).eval()
                self._ae_model = model
                print(f"[model] loaded autoencoder state_dict from {state_path} on {self._device}")
                return
            except Exception as e:  # noqa: BLE001
                print(f"[model] failed to load autoencoder ({e}); enhancement stage will be skipped")

        print(
            "[model] no autoencoder weights found -> enhancement stage will be skipped "
            "(classifier was trained on enhanced images, so accuracy may suffer)"
        )

    @property
    def is_random(self) -> bool:
        return self._model is None

    @property
    def has_autoencoder(self) -> bool:
        return self._ae_model is not None

    # ---- image encoding --------------------------------------------------
    @staticmethod
    def _encode_jpeg_b64(img_rgb_uint8) -> str:
        """RGB uint8 HWC array -> base64-encoded JPEG string (no ``data:`` prefix)."""
        import cv2

        bgr = cv2.cvtColor(img_rgb_uint8, cv2.COLOR_RGB2BGR)
        ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
        if not ok:
            raise ValueError("failed to encode stage image")
        return base64.b64encode(buf.tobytes()).decode("ascii")

    # ---- pipeline ----------------------------------------------------------
    def _run_pipeline(self, image_bytes):
        """Decode -> resize -> gamma correction -> AE enhancement -> ImageNet
        normalize. Mirrors the training notebook's inference_single_image()
        exactly, including the random gamma draw on every call.

        Returns (resized_rgb, gamma, degraded_rgb, enhanced_rgb, tensor, ae_ran).
        """
        import cv2
        import torch

        arr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)  # BGR
        if img is None:
            raise ValueError("could not decode image")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = self.image_size
        resized = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)  # stage 1

        gamma = random.choice(settings.gamma_values)
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        degraded = cv2.LUT(resized.astype(np.uint8), table)  # stage 2

        degraded_01 = degraded.astype(np.float32) / 255.0

        if self._ae_model is not None:
            with torch.no_grad():
                degraded_t = torch.from_numpy(degraded_01.transpose(2, 0, 1)).unsqueeze(0).to(self._device)
                enhanced_01 = self._ae_model(degraded_t).squeeze(0).cpu().numpy().transpose(1, 2, 0)
            enhanced_01 = np.clip(enhanced_01, 0.0, 1.0)
            enhanced_uint8 = (enhanced_01 * 255.0).astype(np.uint8)  # stage 3
            ae_ran = True
        else:
            enhanced_01 = degraded_01
            enhanced_uint8 = degraded
            ae_ran = False

        mean = np.array(settings.imagenet_mean, np.float32)
        std = np.array(settings.imagenet_std, np.float32)
        normalized = (enhanced_01 - mean) / std
        tensor = torch.from_numpy(normalized.transpose(2, 0, 1).astype(np.float32)).unsqueeze(0).to(self._device)

        return resized, float(gamma), degraded, enhanced_uint8, tensor, ae_ran

    def _random_probs(self):
        """Fresh random probabilities each call (no model required). Slight bias
        toward 'healthy' so demo data skews realistic rather than uniform."""
        rng = np.random.default_rng()  # seeded from OS entropy every call
        logits = rng.standard_normal(len(self.classes)) * 1.6
        logits[0] += 0.4
        exp = np.exp(logits - logits.max())
        return exp / exp.sum()

    def predict(self, image_bytes):
        """Return (probabilities dict, predicted_class, confidence, pipeline).

        ``pipeline`` is None in random mode, otherwise a dict with the 3
        base64-encoded stage images, the gamma value actually used, and
        whether the autoencoder ran — enough for verdict.build_result() to
        shape the full API response."""
        if self._model is None:
            probs = self._random_probs()
            pipeline = None
        else:
            import torch

            resized, gamma, degraded, enhanced, tensor, ae_ran = self._run_pipeline(image_bytes)
            with torch.no_grad():
                out = self._model(tensor)
                probs = torch.softmax(out, dim=1)[0].cpu().numpy()
            pipeline = {
                "gamma": gamma,
                "ae_ran": ae_ran,
                "stage1_b64": self._encode_jpeg_b64(resized),
                "stage2_b64": self._encode_jpeg_b64(degraded),
                "stage3_b64": self._encode_jpeg_b64(enhanced),
            }

        idx = int(np.argmax(probs))
        predicted = self.classes[idx]
        confidence = float(probs[idx])
        prob_map = {cls: float(p) for cls, p in zip(self.classes, probs)}
        return prob_map, predicted, confidence, pipeline

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
            "autoencoderLoaded": self.has_autoencoder,
        }


# ---- module-level singleton -------------------------------------------------
_model: GanodermaModel | None = None


def get_model() -> GanodermaModel:
    global _model
    if _model is None:
        _model = GanodermaModel()
    return _model
