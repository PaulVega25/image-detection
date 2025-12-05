"""
Script para exportar YOLOv5 a formato ONNX (liviano para producción)
"""
from ultralytics import YOLO
from pathlib import Path

print("Exportando YOLOv5 a ONNX...")

# Cargar modelo YOLOv5
model = YOLO('yolov5s.pt')

# Exportar a ONNX
model.export(format='onnx', imgsz=640, simplify=True)

print("✓ Modelo exportado a yolov5s.onnx")
onnx_file = Path('yolov5s.onnx')
if onnx_file.exists():
    print(f"Tamaño: {onnx_file.stat().st_size / (1024*1024):.1f} MB")
else:
    print("Archivo no encontrado, revisando nombres...")
