FROM python:3.11-slim

WORKDIR /app

# opencv-python-headless still needs libglib2.0 at runtime even without GUI support.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY run.py ./
COPY models ./models
COPY docker-entrypoint.sh /usr/local/bin/
RUN chmod +x /usr/local/bin/docker-entrypoint.sh

EXPOSE 5005

# Render injects $PORT and expects the app to bind to it; default to 5005 for
# plain `docker run` elsewhere. The entrypoint script expands the variable
# and `exec`s uvicorn so it runs as PID 1 and receives SIGTERM directly.
ENTRYPOINT ["docker-entrypoint.sh"]
