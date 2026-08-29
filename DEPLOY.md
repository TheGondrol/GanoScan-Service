# Deploying to Render (Docker, build-from-repo)

This deploys the service the way Render's Blueprint (`render.yaml`) expects:
Render clones this repo and builds `Dockerfile` itself on every push — no
Docker Hub or any other registry involved.

## 1. Push this repo to GitHub

This folder is already a local git repo with an initial commit (model
weights included — see "Why the model files are committed" below). Create an
empty repo on GitHub, then:

```bash
cd "GanoScan Service"
git remote add origin https://github.com/<you>/ganoscan-service.git
git branch -M main
git push -u origin main
```

## 2. Create the Blueprint on Render

1. [render.com](https://render.com) → **New** → **Blueprint**.
2. Connect the GitHub repo you just pushed.
3. Render reads `render.yaml` and shows one service: `ganoscan-service`.
4. It'll prompt for the env var marked `sync: false` — paste in the
   **exact same value** already hardcoded in the Android app so it keeps
   working without a rebuild: `GANOSCAN_API_KEY` → `ApiClient.API_KEY` in
   `GanoScan/app/src/main/java/com/ganoscan/app/network/ApiClient.kt`
5. Deploy. First build installs torch/torchvision/opencv — expect several
   minutes, not seconds.
6. Check it worked: `curl https://ganoscan-service.onrender.com/health` should
   show `"mode": "torchscript"`, not `"random"`. (Render service names are
   globally unique across all users — if `ganoscan-service` is already taken,
   Render will ask you to pick another before it'll deploy.)

## 3. Point the Android app at it

In `ApiClient.kt`, change:
```kotlin
const val BASE_URL = "http://10.0.2.2:5005/"
```
to your Render URL (`https://ganoscan-service.onrender.com/`), rebuild the app.
Since it's HTTPS with a real cert, no `network_security_config.xml` change is
needed — that file's cleartext exceptions are for the dev-only HTTP hosts.

## Free plan trade-offs (worth knowing before you demo this)

`render.yaml` is set to `plan: free`. That's real money saved, but it comes
with two things that matter specifically for a torch+opencv service:

- **Spins down after 15 minutes of no traffic**, and cold-starts on the next
  request. A cold start here means booting Python, importing torch/opencv,
  and loading the model from disk — likely tens of seconds, not instant.
  The Android app's OkHttp client currently has a 15s connect / 30s read
  timeout (`ApiClient.kt`); a request that hits a fully cold instance can
  plausibly exceed that and fail with a timeout on the *first* try after
  idle, then work fine on retry once the instance is warm. If you're doing a
  live demo, hit `/health` yourself a minute beforehand to warm it up.
- **Limited RAM** on the free instance type — torch + opencv + a loaded
  resnet50 should fit, but there's little headroom. If the container
  restarts unexpectedly or `/health` reports `"mode": "random"` after a
  deploy that should have loaded the real model, check Render's logs for an
  out-of-memory kill before assuming it's a code bug. Current RAM/CPU specs
  are on Render's pricing page — they've changed over time, so don't rely on
  a number from anywhere else, including this file.

The disk itself isn't a concern either way: the service is stateless (no DB,
no stored images), so there's nothing that needs to survive a redeploy or
restart on the server side. Every request is independent.

## Why the model files are committed to git

Render builds from a fresh clone of the repo — it never sees this machine's
`models/` folder unless it's actually in the repo. `.gitignore` was adjusted
to allow exactly `models/model_jit.pt`, `models/best_model.pth`, and
`models/model_info.json` (nothing else that might land in that folder). Both
weight files are individually under GitHub's 100MB hard limit but over its
50MB warning threshold — that warning is expected and harmless here. If you
retrain and the files grow past 100MB, you'd need Git LFS instead.
