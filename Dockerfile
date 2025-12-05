# Usar imagen Python slim (más liviana)
FROM python:3.13-slim

# Variables de entorno
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Instalar solo dependencias del sistema necesarias para OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Copiar requirements
COPY requirements.txt .

# Instalar dependencias Python
RUN pip install --no-cache-dir -r requirements.txt

# Copiar código
COPY app.py train_model.py ./

# Puerto
EXPOSE 8080

# Comando
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
