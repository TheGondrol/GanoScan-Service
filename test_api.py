"""Smoke test: generates a throwaway image and exercises the (stateless) API.
Run the server first, then: python test_api.py [base_url] [api_key]"""

import io
import sys
import json
import urllib.request

from PIL import Image

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:5005"
API_KEY = sys.argv[2] if len(sys.argv) > 2 else ""


def _get(path):
    req = urllib.request.Request(BASE + path, headers=_headers())
    with urllib.request.urlopen(req) as r:
        return r.status, json.load(r)


def _headers(extra: dict | None = None) -> dict:
    headers = {"X-API-Key": API_KEY} if API_KEY else {}
    if extra:
        headers.update(extra)
    return headers


def _post_image(path, color):
    img = Image.new("RGB", (256, 256), color)
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    body, ct = _multipart(buf.getvalue())
    req = urllib.request.Request(BASE + path, data=body, headers=_headers({"Content-Type": ct}))
    with urllib.request.urlopen(req) as r:
        return r.status, json.load(r)


def _multipart(image_bytes, field="image", filename="test.jpg"):
    boundary = "----ganoscanBoundary"
    lines = [
        f"--{boundary}",
        f'Content-Disposition: form-data; name="{field}"; filename="{filename}"',
        "Content-Type: image/jpeg",
        "",
    ]
    pre = ("\r\n".join(lines) + "\r\n").encode()
    post = f"\r\n--{boundary}--\r\n".encode()
    return pre + image_bytes + post, f"multipart/form-data; boundary={boundary}"


def main():
    print("· /health"); print(json.dumps(_get("/health")[1]["model"], indent=2))
    for color in [(40, 120, 60), (170, 90, 40), (200, 200, 120)]:
        status, scan = _post_image("/predict", color)
        print(f"· /predict ({color}) -> {status} {scan['predictedClass']} "
              f"{scan['verdict']} {scan['confidence']:.3f} mock={scan['mock']}")
    print("\nAll endpoints OK ✅")


if __name__ == "__main__":
    main()
