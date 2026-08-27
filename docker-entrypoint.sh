#!/bin/sh
# `exec` replaces this shell with uvicorn (PID 1), so it receives SIGTERM
# directly for a clean shutdown instead of the shell swallowing it.
exec uvicorn app.main:app --host 0.0.0.0 --port "${PORT:-5005}"
