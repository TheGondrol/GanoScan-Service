"""Smoke test: generates a throwaway image and exercises every endpoint.
Run the server first (python app.py), then: python test_api.py"""

import io
import sys
import json
import urllib.request

from PIL import Image

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:5000"


def _get(path):
    with urllib.request.urlopen(BASE + path) as r:
        return r.status, json.load(r)


def _post_image(path, color):
    img = Image.new("RGB", (256, 256), color)
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    body, ct = _multipart(buf.getvalue())
    req = urllib.request.Request(BASE + path, data=body, headers={"Content-Type": ct})
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
        print(f"· /predict ({color}) -> {status} {scan['id']} "
              f"{scan['verdict']} {scan['confidence']:.3f} mock={scan['mock']}")
        last = scan["id"]
    print("· /stats", _get("/stats")[1])
    print("· /history count:", _get("/history")[1]["count"])
    print(f"· /scan/{last} ->", _get(f"/scan/{last}")[1]["label"])
    print("\nAll endpoints OK ✅")


if __name__ == "__main__":
    main()
