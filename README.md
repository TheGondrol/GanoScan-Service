# GanoScan Service — Backend API

**FastAPI** backend for the **GanoScan** Android app. It's a **stateless**
classifier: upload a leaf/stem photo, get back the model's verdict. Nothing is
persisted server-side — no database, no stored images — the Android app owns
its own scan history and stats locally on-device. Blocking work (model
inference) is offloaded to a thread pool so the async event loop stays
responsive; interactive API docs are auto-generated at **`/docs`**.

It maps the model's **2 classes** (`Healthy`, `Infected`) to the app's
verdict (`HEALTHY` / `INFECTED`) plus a severity (`none` / `infected`),
confidence, per-class probabilities and care recommendations.

Preprocessing actually runs the training notebook's full pipeline per
request — **gamma correction** (a random degradation, γ ∈ `{0.8, 1.2}`) →
**CNN autoencoder enhancement** → ImageNet normalize → classifier — and each
of the first 3 stages returns its own base64-encoded JPEG (`stages[i].image`)
so the app can show a real before/after comparison, not just stage labels.

> **Works before training finishes.** With no model files present the service
> runs in **random mode** (random predictions, no PyTorch needed) so the Android
> app can be integrated end-to-end today. Copy the trained model into `models/`
> and restart — real predictions, no code change.

---

## Quick start

### 1. Run now (random mode — no model, no PyTorch)

```bash
cd "GanoScan Service"
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements-core.txt
python run.py                                    # dev, auto-reload
# or: uvicorn app.main:app --host 0.0.0.0 --port 5005
```

Port **5005** (not 5000, which macOS AirPlay Receiver occupies). Open
`http://127.0.0.1:5005/docs` for interactive docs. Smoke test:

```bash
python test_api.py http://127.0.0.1:5005
```

### 2. Real predictions (after training)

Train in `../GanoScan Model`, then copy its exported artifacts into this
service's own `models/` folder (renaming away the notebook's numeric prefix,
e.g. `01_classifier_jit.pt` → `classifier_jit.pt`). There are now **two**
models — the classifier and the enhancement autoencoder:

```bash
cp "../GanoScan Model/models/01_classifier_jit.pt"    models/classifier_jit.pt
cp "../GanoScan Model/models/01_classifier_best.pth"   models/classifier_best.pth
cp "../GanoScan Model/models/01_autoencoder_jit.pt"    models/autoencoder_jit.pt
cp "../GanoScan Model/models/01_autoencoder_best.pth"  models/autoencoder_best.pth
```

`models/model_info.json` isn't a straight copy — the training project's
`results/01_model_info.json` uses a nested shape, while this service reads a
flat one (`classes`, `image_size`, `backbone`, `test_accuracy`). Build it by
hand from that file's values (see the committed `models/model_info.json` for
the current example) whenever you retrain.

The service loads from `models/` (self-contained — no dependency on the
training folder at runtime). Install the ML deps and restart:

```bash
pip install -r requirements.txt   # adds torch, torchvision, opencv
uvicorn app.main:app --host 0.0.0.0 --port 5005
```

Loading priority, independently for each model: **`classifier_jit.pt`** /
**`autoencoder_jit.pt`** (TorchScript, recommended) → `classifier_best.pth` /
`autoencoder_best.pth` (raw weights, rebuilt via `app/services/model_arch.py`)
→ skipped. The classifier is required for real mode; the autoencoder is
optional — if missing, the enhancement stage is skipped and a warning is
logged (`/health`'s `model.autoencoderLoaded` reflects this). Point elsewhere
with `GANOSCAN_MODEL_DIR=/path/to/models`.

Check what loaded:

```bash
curl -s http://127.0.0.1:5005/health | python3 -m json.tool
```

`"mode"` will be `torchscript`, `state_dict`, or `random`.

---

## API

Base URL (local): `http://127.0.0.1:5005`

| Method | Path | Purpose |
|--------|------|---------|
| `GET`  | `/health` | Service + model status (mode, backbone, classes, accuracy) |
| `POST` | `/predict` | Multipart image → classify, return the result (nothing stored) |
| `GET`  | `/docs` | Interactive OpenAPI docs (auto-generated) |

That's the whole surface. There's no `/history`, `/stats`, or `/scan/{id}` —
those live entirely on-device now (Room database in the Android app).

The full contract is checked in at [`openapi.yaml`](openapi.yaml) (OpenAPI
3.1, validated) — regenerate it after changing any route/schema with:

```bash
curl -s http://127.0.0.1:5005/openapi.json | python3 -c \
  "import json,sys,yaml; yaml.dump(json.load(sys.stdin), sys.stdout, sort_keys=False, allow_unicode=True)" \
  > openapi.yaml
```

### `POST /predict`

Multipart form:

| Field | Required | Notes |
|-------|----------|-------|
| `image` | ✅ | The photo (jpg/png/webp) |

```bash
curl -F "image=@leaf.jpg" http://127.0.0.1:5005/predict
```

Response (`200`):

```json
{
  "verdict": "INFECTED",
  "label": "Terinfeksi Ganoderma",
  "predictedClass": "Infected",
  "severity": "infected",
  "confidence": 0.9231,
  "probabilities": { "Healthy": 0.0769, "Infected": 0.9231 },
  "gamma": 0.8,
  "inputResolution": "1024×768 px",
  "recommendations": ["Tumbang & musnahkan pohon terinfeksi segera", "..."],
  "stages": [
    { "index": "1", "title": "Citra Asli", "sub": "input 1024×768 px", "image": "<base64 jpeg>" },
    { "index": "2", "title": "Gamma Correction", "sub": "γ = 0.8 · kontras diacak", "image": "<base64 jpeg>" },
    { "index": "3", "title": "CNN-Based Enhancement", "sub": "detail tekstur dipertajam", "image": "<base64 jpeg>" },
    { "index": "✓", "title": "Klasifikasi (CNN)", "sub": "output: Terinfeksi · 0.923", "done": true }
  ],
  "mock": false
}
```

`stages[i].image` (stages 1–3 only) is a base64-encoded JPEG of that
pipeline step's actual output for this request — no `data:` prefix. It's
`null` in random/mock mode, since nothing real ran. `gamma` is the value
that was actually randomly drawn for this request (one of
`GANOSCAN_GAMMA_VALUES`, default `[0.8, 1.2]`), not a fixed display constant.

The app is responsible for generating its own scan id/timestamp, saving the
photo (and, if it wants to keep them, the stage images) to local storage,
and inserting the record into its local database — none of that is this
service's concern anymore.

---

## Connecting the Android app

The app's base URL lives in
`GanoScan/app/src/main/java/com/ganoscan/app/network/ApiClient.kt`:

```kotlin
const val BASE_URL = "http://10.0.2.2:5005/"
```

| Running on | Set `BASE_URL` to | Also |
|------------|-------------------|------|
| **Emulator** | `http://10.0.2.2:5005/` (host loopback) | works out of the box |
| **Physical device** | `http://<your-PC-LAN-IP>:5005/` | add that IP to `res/xml/network_security_config.xml`, same Wi-Fi, run the server with `GANOSCAN_HOST=0.0.0.0` (default) |

Cleartext HTTP is allowed only for those dev hosts (see the network security
config); production traffic stays HTTPS-only.

Only `/predict` and `/health` need the network — the app's history, stats, and
photos all live on-device, so they're available offline even when this
service isn't reachable.

**API key:** once `GANOSCAN_API_KEY` is set here, the app's
`ApiClient.API_KEY` constant (same file) must be set to the identical value —
Retrofit attaches it as `X-API-Key` automatically. Mismatched or blank keys
show up as every request failing with 401.

---

## How preprocessing matches training

Inference mirrors the notebook's `inference_single_image()` exactly, so
results are consistent — including gamma correction and the autoencoder,
which really run per request (they used to be preprocessing steps described
only in the response's stage labels; they're not anymore):

1. `cv2` decode → **BGR→RGB** → resize to the model's `image_size` (bilinear).
2. **Gamma correction**: γ is picked at random from `GANOSCAN_GAMMA_VALUES`
   (default `[0.8, 1.2]`) on every request and applied via a `cv2.LUT`
   gamma table — this matches the notebook's own inference function, which
   re-degrades every image before enhancing it (the classifier was only ever
   trained on gamma-degraded → enhanced images, never on raw ones, so this is
   intentional, not a bug).
3. **CNN-based enhancement**: the gamma-corrected image (scaled to `[0,1]`)
   is passed through the frozen `EnhancementAutoencoderV2` U-Net.
4. ImageNet normalize (`mean=[0.485,0.456,0.406]`, `std=[0.229,0.224,0.225]`)
   on the enhanced image → CHW tensor → classifier → `softmax`.

Each of steps 1–3 is base64-JPEG-encoded and returned in the matching
`stages[i].image` field (see the example response above).

---

## Configuration (env vars)

All optional — see `.env.example`. Common ones:

| Var | Default | Meaning |
|-----|---------|---------|
| `GANOSCAN_MODEL_DIR` | `./models` | Where model files live |
| `GANOSCAN_PORT` | `5005` | Server port |
| `GANOSCAN_HOST` | `0.0.0.0` | Bind address |
| `GANOSCAN_GAMMA_VALUES` | `[0.8, 1.2]` | Candidates randomly drawn for the gamma-correction stage on every request |
| `GANOSCAN_API_KEY` | unset | If set, required as `X-API-Key` on every endpoint except `/health` and the docs |

---

## Project layout

Conventional FastAPI package structure:

```
GanoScan Service/
├── app/
│   ├── main.py                 # app factory: lifespan, CORS, routers
│   ├── core/
│   │   ├── config.py           # pydantic-settings (env: GANOSCAN_*)
│   │   └── security.py         # X-API-Key middleware
│   ├── api/
│   │   ├── deps.py             # DI (model singleton)
│   │   ├── router.py           # aggregates routers
│   │   └── routes/
│   │       ├── health.py       # GET /health
│   │       └── predict.py      # POST /predict
│   ├── schemas/scan.py         # Pydantic response models
│   └── services/
│       ├── inference.py        # model load / gamma+AE pipeline / predict (random default)
│       ├── model_arch.py       # GanodermaCNN + EnhancementAutoencoderV2 (for *_best.pth)
│       └── verdict.py          # 2-class -> app contract + recommendations
├── run.py                      # dev entrypoint (uvicorn --reload)
├── test_api.py                 # endpoint smoke test
├── requirements.txt            # full (with torch/opencv)
├── requirements-core.txt       # random-mode only
└── models/                     # copied in from ../GanoScan Model (git-tracked, see .gitignore)
```

## Production note

`python run.py` runs Uvicorn with `--reload` (dev). For deployment run without
reload and add workers, e.g. `uvicorn app.main:app --host 0.0.0.0 --port 5005 --workers 2`,
or under Gunicorn: `gunicorn -k uvicorn.workers.UvicornWorker -w 2 -b 0.0.0.0:5005 app.main:app`.

Being stateless, this service is trivial to scale horizontally (no shared
disk/DB to coordinate) — any number of instances behind a load balancer work
fine, since every request is independent.
