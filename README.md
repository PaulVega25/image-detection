# API de Detección de Personas en Imágenes

Sistema de análisis de imágenes que detecta si una imagen contiene **al menos 1 persona**, diseñado para validar imágenes relacionadas con hechos delictivos.

## 📋 Descripción

Este proyecto implementa:
- **Modelo de detección**: YOLOv5 pre-entrenado en COCO dataset para detectar personas
- **API REST**: FastAPI con endpoint para análisis de imágenes
- **Validación**: Rechaza imágenes sin personas (perros, tazas, objetos, carros, etc.)
- **Dataset**: Validado con dataset de Roboflow "People Detection" (15,210+ imágenes)

## 🚀 Instalación

### 1. Instalar dependencias

```powershell
pip install -r requirements.txt
```

### 2. Generar el modelo

```powershell
python train_model.py
```

Este comando creará el archivo `person_detector_model.joblib` con el modelo entrenado.

## 💻 Uso

### Iniciar el servidor

```powershell
python app.py
```

O usando uvicorn directamente:

```powershell
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

El servidor estará disponible en: `http://localhost:8000`

### Documentación interactiva

Una vez iniciado el servidor, accede a:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## 📡 Endpoints

### POST `/analizar-imagen`

Analiza una imagen y determina si contiene al menos 1 persona (valida hechos delictivos).

**Request:**
- Método: `POST`
- Content-Type: `multipart/form-data`
- Body: Archivo de imagen (JPG, PNG, etc.)

**Response exitosa (imagen con persona):**

```json
{
  "valida": true,
  "numero_personas": 3,
  "mensaje": "Imagen válida: se detectaron 3 persona(s). Puede ser un hecho delictivo.",
  "confianza_promedio": 0.85,
  "dimensiones_imagen": {
    "ancho": 1920,
    "alto": 1080
  }
}
```

**Response imagen inválida (sin personas):**

```json
{
  "valida": false,
  "numero_personas": 0,
  "mensaje": "Imagen no válida: no se detectaron personas. No es un hecho delictivo válido (puede ser objeto, animal, etc.).",
  "confianza_promedio": 0.0,
  "dimensiones_imagen": {
    "ancho": 800,
    "alto": 600
  }
}
```

### GET `/health`

Verifica el estado del servicio.

```json
{
  "status": "ok",
  "modelo_cargado": true
}
```

## 🧪 Probar la API

### Con cURL:

```powershell
curl -X POST "http://localhost:8000/analizar-imagen" -F "imagen=@ruta/a/tu/imagen.jpg"
```

### Con Python:

```python
import requests

url = "http://localhost:8000/analizar-imagen"
files = {"imagen": open("imagen_prueba.jpg", "rb")}

response = requests.post(url, files=files)
print(response.json())
```

### Con la interfaz Swagger:

1. Navega a http://localhost:8000/docs
2. Click en el endpoint `/analizar-imagen`
3. Click en "Try it out"
4. Selecciona una imagen
5. Click en "Execute"

## 🔧 Configuración del modelo

El modelo utiliza YOLOv5s con los siguientes parámetros ajustables en `train_model.py`:

```python
# Umbral de confianza para detecciones
umbral_confianza = 0.4  # Confianza mínima para considerar una detección

# Clases a detectar (solo personas)
self.model.classes = [0]  # 0 = person en COCO dataset
```

## 📊 Dataset

El modelo fue validado con el dataset de Roboflow "People Detection":
- **15,210 imágenes** de entrenamiento
- **100,000+ anotaciones** de personas
- Múltiples escenarios: calles, multitudes, cámaras de seguridad, etc.
- Fuente: https://universe.roboflow.com/leo-ueno/people-detection-o4rdr

## 📁 Estructura del proyecto

```
modelo-imagen/
│
├── app.py                          # API FastAPI
├── train_model.py                  # Script para generar el modelo
├── person_detector_model.joblib    # Modelo guardado (generado)
├── requirements.txt                # Dependencias
└── README.md                       # Documentación
```

## 🎯 Casos de uso

**Imágenes válidas** (≥ 1 persona - posibles hechos delictivos):
- ✅ Persona individual
- ✅ Grupos de personas
- ✅ Multitudes
- ✅ Eventos
- ✅ Cualquier escena con al menos una persona visible

**Imágenes inválidas** (0 personas - NO son hechos delictivos):
- ❌ Imágenes de animales (perros, gatos)
- ❌ Objetos inanimados (tazas, mesas, carros)
- ❌ Paisajes sin personas
- ❌ Edificios vacíos
- ❌ Naturaleza sin personas

## 🛠️ Tecnologías utilizadas

- **FastAPI**: Framework web moderno y rápido
- **YOLOv5**: Estado del arte en detección de objetos
- **PyTorch**: Framework de deep learning
- **OpenCV**: Procesamiento de imágenes
- **Joblib**: Serialización del modelo
- **Pillow**: Manipulación de imágenes
- **Uvicorn**: Servidor ASGI
## 📝 Notas importantes

1. El modelo YOLOv5 se descarga automáticamente la primera vez (aprox. 14 MB)
2. YOLOv5 funciona muy bien en múltiples escenarios y condiciones
3. La detección es robusta ante diferentes ángulos, iluminación y poses
4. Para mejorar la detección en tu contexto específico, ajusta el `umbral_confianza`
5. El primer análisis puede tardar más mientras se carga el modelo en memoria
## 🔜 Mejoras futuras

- Fine-tuning de YOLOv5 con dataset específico de hechos delictivos
- Implementar YOLOv8 o YOLOv9 para mayor precisión
- Añadir cache de resultados para optimizar respuestas
- Implementar procesamiento por lotes (batch)
- Añadir autenticación y rate limiting
- Guardar logs de detecciones con timestamps
- API para reentrenamiento con nuevas imágenesor lotes
- Añadir autenticación y rate limiting
- Guardar logs de detecciones

## 📞 Soporte

Para reportar problemas o sugerencias, crea un issue en el repositorio.
