import cv2
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict
import torch


class PersonDetector:
    """  
    Detector de personas usando YOLO (YOLOv5) de PyTorch.
    Detecta personas para validar que la imagen contenga al menos una persona.
    Útil para filtrar imágenes de objetos, animales o paisajes sin personas.
    """
    
    def __init__(self):
        # Cargar modelo YOLOv5 pre-entrenado
        print("Cargando modelo YOLOv5...")
        self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        # Configurar para detectar solo personas (clase 0 en COCO)
        self.model.classes = [0]  # 0 = person en COCO dataset
        print("✓ Modelo YOLOv5 cargado exitosamente")
        
    def detectar_personas(self, imagen: np.ndarray, umbral_confianza: float = 0.4) -> Dict:
        """
        Detecta personas en una imagen usando YOLO.
        
        Args:
            imagen: Imagen en formato numpy array (BGR)
            umbral_confianza: Umbral mínimo de confianza (default: 0.4)
        
        Returns:
            Dict con información sobre las personas detectadas
        """
        # Convertir BGR a RGB (YOLO espera RGB)
        imagen_rgb = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)
        
        # Realizar detección
        results = self.model(imagen_rgb)
        
        # Extraer detecciones
        detections = results.pandas().xyxy[0]
        
        # Filtrar por confianza
        detecciones_validas = detections[detections['confidence'] >= umbral_confianza]
        
        # Extraer información
        detecciones_list = []
        confianzas_list = []
        
        for _, row in detecciones_validas.iterrows():
            x1, y1, x2, y2 = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
            w, h = x2 - x1, y2 - y1
            detecciones_list.append((x1, y1, w, h))
            confianzas_list.append(float(row['confidence']))
        
        confianza_promedio = np.mean(confianzas_list) if confianzas_list else 0.0
        
        return {
            "numero_personas": len(detecciones_list),
            "detecciones": detecciones_list,
            "confianza_promedio": float(confianza_promedio),
            "confianzas": confianzas_list
        }
    
    def visualizar_detecciones(self, imagen: np.ndarray, guardar_como: str = None) -> np.ndarray:
        """
        Visualiza las detecciones dibujando rectángulos en la imagen.
        
        Args:
            imagen: Imagen en formato numpy array (BGR)
            guardar_como: Ruta opcional para guardar la imagen
        
        Returns:
            Imagen con las detecciones dibujadas
        """
        resultado = self.detectar_personas(imagen)
        img_con_detecciones = imagen.copy()
        
        for i, (x, y, w, h) in enumerate(resultado["detecciones"]):
            # Dibujar rectángulo
            cv2.rectangle(img_con_detecciones, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Añadir etiqueta con confianza
            confianza = resultado["confianzas"][i]
            etiqueta = f"Persona {i+1}: {confianza:.2f}"
            cv2.putText(
                img_con_detecciones,
                etiqueta,
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2
            )
        
        if guardar_como:
            cv2.imwrite(guardar_como, img_con_detecciones)
        
        return img_con_detecciones

def analizar_dataset(dataset_path: str, detector: PersonDetector):
    """
    Analiza el dataset para entender la distribución de personas por imagen.
    
    Args:
        dataset_path: Ruta al directorio del dataset
        detector: Instancia del detector
    """
    print("\n📊 Analizando dataset...")
    
    # Leer anotaciones
    annotations_path = Path(dataset_path) / "_annotations.csv"
    
    if not annotations_path.exists():
        print(f"⚠ No se encontró el archivo de anotaciones en: {annotations_path}")
        return
    
    df = pd.read_csv(annotations_path)
    
    # Contar personas por imagen
    personas_por_imagen = df.groupby('filename').size()
    
    # Estadísticas
    total_imagenes = len(personas_por_imagen)
    imagenes_con_1_o_mas = (personas_por_imagen >= 1).sum()
    imagenes_sin_personas = (personas_por_imagen < 1).sum()
    
    print(f"\n📈 Estadísticas del dataset:")
    print(f"  Total de imágenes únicas: {total_imagenes}")
    print(f"  Imágenes con ≥1 persona: {imagenes_con_1_o_mas} ({imagenes_con_1_o_mas/total_imagenes*100:.1f}%)")
    print(f"  Imágenes sin personas: {imagenes_sin_personas} ({imagenes_sin_personas/total_imagenes*100:.1f}%)")
    print(f"  Promedio de personas por imagen: {personas_por_imagen.mean():.2f}")
    print(f"  Máximo de personas en una imagen: {personas_por_imagen.max()}")
    print(f"  Mínimo de personas en una imagen: {personas_por_imagen.min()}")
    
    # Probar detección en algunas imágenes de muestra
    print(f"\n🔍 Probando detección en imágenes de muestra...")
    
    imagenes_muestra = personas_por_imagen.head(5).index.tolist()
    dataset_dir = Path(dataset_path)
    
    aciertos = 0
    total_pruebas = 0
    
    for img_name in imagenes_muestra:
        img_path = dataset_dir / img_name
        if img_path.exists():
            personas_reales = int(personas_por_imagen[img_name])
            img = cv2.imread(str(img_path))
            
            if img is not None:
                resultado = detector.detectar_personas(img)
                personas_detectadas = resultado['numero_personas']
                
                # Comparar clasificación (>=1 vs 0)
                real_valida = personas_reales >= 1
                detectada_valida = personas_detectadas >= 1
                
                if real_valida == detectada_valida:
                    aciertos += 1
                
                total_pruebas += 1
                
                print(f"  {img_name[:40]:40} | Real: {personas_reales:2d} | Detectado: {personas_detectadas:2d} | {'✓' if real_valida == detectada_valida else '✗'}")
    
    if total_pruebas > 0:
        precision = (aciertos / total_pruebas) * 100
        print(f"\n  Precisión en muestra: {precision:.1f}% ({aciertos}/{total_pruebas})")


def main():
    """Crear y guardar el modelo"""
    print("="*60)
    print("🚀 ENTRENAMIENTO DEL MODELO DE DETECCIÓN DE PERSONAS")
    print("="*60)
    
    # Crear el detector
    detector = PersonDetector()
    
    # Analizar el dataset si existe
    dataset_path = "../dataset-images/train/train"
    if Path(dataset_path).exists():
        analizar_dataset(dataset_path, detector)
    else:
        print(f"\n⚠ Dataset no encontrado en: {dataset_path}")
        print("El modelo se guardará con YOLO pre-entrenado de todas formas.")
    
    # Guardar el modelo usando joblib
    print("\n💾 Guardando modelo...")
    modelo_path = "person_detector_model.joblib"
    joblib.dump(detector, modelo_path)
    
    print(f"\n✓ Modelo guardado exitosamente en: {modelo_path}")
    print("\n📋 Información del modelo:")
    print("   - Tipo: YOLOv5s")
    print("   - Pre-entrenado en COCO dataset")
    print("   - Optimizado para detección de personas en múltiples contextos")
    print("   - Umbral de confianza: 0.4")
    print("\n🎯 El modelo valida que haya al menos 1 persona en la imagen")
    print("   ✓ Rechaza: objetos, animales, carros, paisajes sin personas")
    print("   ✓ Acepta: cualquier imagen con al menos una persona visible")
    print("\n▶ Puedes iniciar la API ejecutando:")
    print("   python app.py")
    print("\n" + "="*60)


if __name__ == "__main__":
    main()
