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
    
    def postprocess(self, outputs: np.ndarray, umbral_confianza: float = 0.4, iou_threshold: float = 0.45) -> List[Dict]:
        """Procesar salidas de YOLO y filtrar personas con NMS"""
        detecciones = []
        
        # outputs es una lista, tomar el primer elemento
        output = outputs[0] if isinstance(outputs, (list, tuple)) else outputs
        
        # Shape: (1, 84, 8400) -> queremos (8400, 84)
        # [batch, features, boxes] -> [boxes, features]
        if len(output.shape) == 3:
            predictions = output[0].T  # (84, 8400) -> (8400, 84)
        else:
            predictions = output
        
        # Recopilar todas las detecciones con confianza suficiente
        boxes = []
        scores = []
        
        for pred in predictions:
            if len(pred) < 84:
                continue
            
            # Extraer bbox
            x_center, y_center, width, height = pred[:4]
            
            # Las confianzas de clases están en pred[4:84]
            class_confidences = pred[4:]
            
            # Persona es clase 0
            person_conf = float(class_confidences[0])
            
            if person_conf >= umbral_confianza:
                boxes.append([
                    float(x_center - width/2),
                    float(y_center - height/2),
                    float(width),
                    float(height)
                ])
                scores.append(person_conf)
        
        # Aplicar Non-Maximum Suppression si hay detecciones
        if len(boxes) > 0:
            boxes_array = np.array(boxes, dtype=np.float32)
            scores_array = np.array(scores, dtype=np.float32)
            
            # NMS usando OpenCV
            indices = cv2.dnn.NMSBoxes(
                bboxes=boxes_array.tolist(),
                scores=scores_array.tolist(),
                score_threshold=umbral_confianza,
                nms_threshold=iou_threshold
            )
            
            # Crear lista de detecciones finales
            if len(indices) > 0:
                for i in indices.flatten():
                    box = boxes[i]
                    detecciones.append({
                        'bbox': [
                            box[0],
                            box[1],
                            box[0] + box[2],
                            box[1] + box[3]
                        ],
                        'confianza': scores[i],
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
