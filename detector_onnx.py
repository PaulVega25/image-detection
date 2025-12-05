"""
Detector de personas liviano usando ONNX Runtime (sin PyTorch)
Similar al proyecto de audio - solo inferencia con modelo pre-entrenado
"""
import cv2
import numpy as np
import onnxruntime as ort
from typing import Dict, List


class PersonDetectorONNX:
    """
    Detector de personas usando YOLOv5 en formato ONNX.
    Mucho más liviano que PyTorch (~200MB vs ~2.5GB)
    """
    
    def __init__(self, model_path: str = "yolov5su.onnx"):
        """Cargar modelo ONNX pre-entrenado"""
        print("Cargando modelo ONNX...")
        self.session = ort.InferenceSession(
            model_path,
            providers=['CPUExecutionProvider']
        )
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [out.name for out in self.session.get_outputs()]
        print("✓ Modelo ONNX cargado exitosamente")
        
    def preprocess(self, imagen: np.ndarray) -> np.ndarray:
        """Preprocesar imagen para YOLO"""
        # Convertir BGR a RGB
        img = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)
        
        # Resize manteniendo aspecto
        img_resized = cv2.resize(img, (640, 640))
        
        # Normalizar y reordenar dimensiones
        img_normalized = img_resized.astype(np.float32) / 255.0
        img_transposed = np.transpose(img_normalized, (2, 0, 1))
        img_batch = np.expand_dims(img_transposed, axis=0)
        
        return img_batch
    
    def postprocess(self, outputs: np.ndarray, umbral_confianza: float = 0.4) -> List[Dict]:
        """Procesar salidas de YOLO y filtrar personas"""
        detecciones = []
        
        # outputs es una lista, tomar el primer elemento
        output = outputs[0] if isinstance(outputs, (list, tuple)) else outputs
        
        # Shape: (1, 84, 8400) -> queremos (8400, 84)
        # [batch, features, boxes] -> [boxes, features]
        if len(output.shape) == 3:
            predictions = output[0].T  # (84, 8400) -> (8400, 84)
        else:
            predictions = output
        
        for pred in predictions:
            # pred shape: (84,) = [x, y, w, h, conf_obj] + [80 confianzas de clases]
            # Pero en YOLOv5u el formato es diferente: [x, y, w, h] + [80 confianzas]
            if len(pred) < 84:
                continue
            
            # Extraer bbox
            x_center, y_center, width, height = pred[:4]
            
            # Las confianzas de clases están en pred[4:84]
            class_confidences = pred[4:]
            
            # Persona es clase 0
            person_conf = class_confidences[0]
            
            if person_conf >= umbral_confianza:
                detecciones.append({
                    'bbox': [
                        float(x_center - width/2),
                        float(y_center - height/2),
                        float(x_center + width/2),
                        float(y_center + height/2)
                    ],
                    'confianza': float(person_conf),
                    'clase': 'persona'
                })
        
        return detecciones
    
    def detectar_personas(self, imagen: np.ndarray, umbral_confianza: float = 0.4) -> Dict:
        """
        Detecta personas en una imagen.
        
        Args:
            imagen: Imagen en formato numpy array (BGR)
            umbral_confianza: Umbral mínimo de confianza (default: 0.4)
        
        Returns:
            Dict con numero_personas, detecciones y confianzas
        """
        # Preprocesar
        input_data = self.preprocess(imagen)
        
        # Inferencia
        outputs = self.session.run(self.output_names, {self.input_name: input_data})
        
        # Postprocesar
        detecciones = self.postprocess(outputs[0], umbral_confianza)
        
        return {
            "numero_personas": len(detecciones),
            "detecciones": detecciones,
            "confianzas": [d['confianza'] for d in detecciones]
        }
