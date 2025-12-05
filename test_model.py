"""
Script de prueba para el modelo de detección de personas.
Permite probar el modelo con imágenes locales antes de usar la API.
"""

import cv2
import joblib
import sys
from pathlib import Path


def probar_imagen(ruta_imagen: str):
    """
    Prueba el modelo con una imagen local.
    
    Args:
        ruta_imagen: Ruta a la imagen de prueba
    """
    # Verificar que existe el modelo
    if not Path("person_detector_model.joblib").exists():
        print("❌ Error: Modelo no encontrado.")
        print("Ejecuta primero: python train_model.py")
        return
    
    # Verificar que existe la imagen
    if not Path(ruta_imagen).exists():
        print(f"❌ Error: Imagen no encontrada en: {ruta_imagen}")
        return
    
    print(f"\n🔍 Analizando imagen: {ruta_imagen}\n")
    
    # Cargar el modelo
    detector = joblib.load("person_detector_model.joblib")
    
    # Cargar la imagen
    imagen = cv2.imread(ruta_imagen)
    
    if imagen is None:
        print(f"❌ Error: No se pudo cargar la imagen")
        return
    
    # Realizar la detección
    resultado = detector.detectar_personas(imagen)
    
    # Mostrar resultados
    num_personas = resultado["numero_personas"]
    print(f"{'='*50}")
    print(f"Personas detectadas: {num_personas}")
    print(f"Confianza promedio: {resultado['confianza_promedio']:.2f}")
    print(f"{'='*50}\n")
    
    # Determinar validez
    es_valida = num_personas >= 1
    
    if es_valida:
        print(f"✅ IMAGEN VÁLIDA - Posible hecho delictivo")
        print(f"   Se detectaron {num_personas} persona(s)")
    else:
        print(f"❌ IMAGEN NO VÁLIDA - No es hecho delictivo")
        print(f"   No se detectaron personas")
        print(f"   La imagen puede ser de objetos, animales, paisajes, etc.")
    
    # Crear imagen con las detecciones marcadas
    nombre_salida = Path(ruta_imagen).stem + "_resultado.jpg"
    detector.visualizar_detecciones(imagen, nombre_salida)
    print(f"\n💾 Resultado guardado en: {nombre_salida}")
    
    # Mostrar detalle de cada detección
    if resultado["detecciones"]:
        print(f"\nDetalle de detecciones:")
        for i, ((x, y, w, h), conf) in enumerate(zip(resultado["detecciones"], resultado["confianzas"])):
            print(f"  Persona {i+1}: posición=({x}, {y}), tamaño=({w}x{h}), confianza={conf:.3f}")


def main():
    if len(sys.argv) < 2:
        print("Uso: python test_model.py <ruta_imagen>")
        print("\nEjemplo:")
        print("  python test_model.py imagen_prueba.jpg")
        return
    
    ruta_imagen = sys.argv[1]
    probar_imagen(ruta_imagen)


if __name__ == "__main__":
    main()
