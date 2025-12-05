# Imagen Python slim - liviana
FROM python:3.11-slim

# Variables de entorno
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Dependencias del sistema solo para OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libgomp1 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Instalar dependencias Python (versión liviana - sin PyTorch)
RUN pip install --no-cache-dir \
    fastapi==0.123.9 \
    uvicorn==0.38.0 \
    python-multipart==0.0.20 \
    opencv-python-headless==4.12.0.88 \
    pillow==12.0.0 \
    numpy==2.2.6 \
    onnxruntime==1.23.2

# Copiar código y modelo ONNX
COPY app.py detector_onnx.py yolov5su.onnx ./

# Puerto
EXPOSE 8080

# Comando
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
