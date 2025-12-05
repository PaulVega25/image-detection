# Usar imagen Python slim
FROM python:3.11-slim

# Variables de entorno
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Instalar dependencias del sistema
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Copiar solo requirements primero (cache)
COPY requirements.txt .

# Instalar dependencias en pasos separados para reducir memoria
RUN pip install --no-cache-dir numpy==2.2.6
RUN pip install --no-cache-dir torch==2.9.1 --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir opencv-python-headless==4.12.0.88
RUN pip install --no-cache-dir fastapi==0.123.9 uvicorn==0.38.0 python-multipart==0.0.20
RUN pip install --no-cache-dir ultralytics==8.3.235 pandas==2.3.3 pillow==12.0.0

# Copiar código
COPY app.py train_model.py ./

# Puerto
EXPOSE 8080

# Comando
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
