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
3. Render reads `render.yaml` and shows one service: `ganoscan-backend`.
4. It'll prompt for the two env vars marked `sync: false` — paste in the
   **exact same values** already hardcoded in the Android app so it keeps
   working without a rebuild:
   - `GANOSCAN_API_KEY` → `ApiClient.API_KEY` in
     `GanoScan/app/src/main/java/com/ganoscan/app/network/ApiClient.kt`
   - `GANOSCAN_ADMIN_TOKEN` → whatever you set in your local `.env`
     (only needed if you actually plan to call the `DELETE` endpoints
     against the deployed service)
5. Deploy. First build installs torch/torchvision/opencv — expect several
   minutes, not seconds.
6. Check it worked: `curl https://<your-service>.onrender.com/health` should
   show `"mode": "torchscript"`, not `"random"`.

## 3. Point the Android app at it

In `ApiClient.kt`, change:
```kotlin
const val BASE_URL = "http://10.0.2.2:5005/"
```
to your Render URL (`https://<your-service>.onrender.com/`), rebuild the app.
Since it's HTTPS with a real cert, no `network_security_config.xml` change is
needed — that file's cleartext exceptions are for the dev-only HTTP hosts.

## Why the model files are committed to git

Render builds from a fresh clone of the repo — it never sees this machine's
`models/` folder unless it's actually in the repo. `.gitignore` was adjusted
to allow exactly `models/model_jit.pt`, `models/best_model.pth`, and
`models/model_info.json` (nothing else that might land in that folder). Both
weight files are individually under GitHub's 100MB hard limit but over its
50MB warning threshold — that warning is expected and harmless here. If you
retrain and the files grow past 100MB, you'd need Git LFS instead.

## The one real gap: no persistent disk by default

Render's default web service disk is **ephemeral** — `scans.db` and
`uploads/` (created fresh inside the container at startup) are wiped on
every redeploy or restart. The model files are fine (they're baked into the
image itself, not runtime state), but scan history is not.

If you need history to survive redeploys, add a
[Render Disk](https://render.com/docs/disks) (paid — check current pricing
on your dashboard) and point the app at it via env vars:

```yaml
# add under the service in render.yaml
disk:
  name: ganoscan-data
  mountPath: /app/data
  sizeGB: 1
```
```bash
# additional env vars
GANOSCAN_DB=/app/data/scans.db
GANOSCAN_UPLOADS=/app/data/uploads
```

Without this, the service still works fine for demoing predictions — it just
starts with an empty history after every redeploy/restart.
