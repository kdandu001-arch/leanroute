FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 HF_HOME=/models
WORKDIR /srv

# CPU-only PyTorch keeps the image small. On ARM (Oracle Ampere) PyPI wheels are already CPU-only.
ARG TORCH_INDEX=https://download.pytorch.org/whl/cpu
RUN pip install torch --index-url ${TORCH_INDEX} || pip install torch

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY app ./app

# Download Laya weights at build time so containers start fast (set to 0 to skip)
ARG PREFETCH_MODEL=1
RUN if [ "$PREFETCH_MODEL" = "1" ]; then python -c "import laya; laya.Router(preload=True)"; fi

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
