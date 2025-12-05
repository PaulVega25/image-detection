from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import numpy as np
from PIL import Image
import io
import cv2
from typing import Dict
import sys
from pathlib import Path

# Importar el detector ONNX liviano
sys.path.insert(0, str(Path(__file__).parent))
from detector_onnx import PersonDetectorONNX

app = FastAPI(title="API de Detección de Personas", version="1.0.0")

# Crear el detector al iniciar la aplicación
try:
    detector = PersonDetectorONNX("yolov5su.onnx")
except Exception as e:
    print(f"Error al cargar el modelo: {e}")
    detector = None


@app.get("/")
async def root():
    """Endpoint raíz con información de la API"""
    return {
        "mensaje": "API de Detección de Personas en Imágenes",
        "descripcion": "Detecta si una imagen contiene al menos 1 persona (valida hechos delictivos)",
        "uso": "POST /analizar-imagen con una imagen"
    }


@app.post("/analizar-imagen")
async def analizar_imagen(imagen: UploadFile = File(...)) -> Dict:
    """
    Analiza una imagen para detectar si contiene al menos 1 persona.
    Valida que la imagen sea de un hecho delictivo (no objetos, animales, etc.).
    
    Args:
        imagen: Archivo de imagen (JPG, PNG, etc.)
    
    Returns:
        Dict con resultado del análisis
    """
    if detector is None:
        raise HTTPException(
            status_code=503,
            detail="Modelo no disponible. Ejecute train_model.py para entrenar el modelo."
        )
    
    # Validar tipo de archivo
    if not imagen.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="El archivo debe ser una imagen válida"
        )
    
    try:
        # Leer la imagen
        contenido = await imagen.read()
        img = Image.open(io.BytesIO(contenido))
        
        # Convertir a RGB si es necesario
        if img.mode != "RGB":
            img = img.convert("RGB")
        
        # Convertir PIL Image a numpy array para OpenCV
        img_array = np.array(img)
        img_cv = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        
        # Realizar la detección
        resultado = detector.detectar_personas(img_cv)
        
        # Determinar si es válida (al menos 1 persona)
        es_valida = resultado["numero_personas"] >= 1
        
        return {
            "valida": es_valida,
            "mensaje": (
                "Imagen válida: contiene persona(s)."
                if es_valida
                else "Imagen no válida: no contiene personas."
            )
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error al procesar la imagen: {str(e)}"
        )


@app.get("/health")
async def health_check():
    """Endpoint de salud para verificar el estado del servicio"""
    return {
        "status": "ok",
        "modelo_cargado": detector is not None
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
