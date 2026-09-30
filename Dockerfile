FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 PORT=8000
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# only what's needed at runtime (training data stays out of the image)
COPY server/ server/
COPY UI/ UI/

EXPOSE 8000
# 1 worker + threads: the models are loaded once per worker
CMD gunicorn --chdir server -w 1 --threads 4 -b 0.0.0.0:${PORT} --timeout 60 server:app
