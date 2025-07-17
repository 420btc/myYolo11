#!/usr/bin/env python3
"""
Script de inicio para la aplicación YOLO11 con Streamlit
"""

import subprocess
import sys
import os

def check_dependencies():
    """Verificar que las dependencias estén instaladas"""
    try:
        import streamlit
        import ultralytics
        import cv2
        print("✅ Todas las dependencias están instaladas")
        return True
    except ImportError as e:
        print(f"❌ Dependencia faltante: {e}")
        print("🔧 Instalando dependencias...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        return True

def main():
    """Función principal"""
    print("🚀 Iniciando aplicación YOLO11...")
    
    # Verificar dependencias
    if not check_dependencies():
        print("❌ Error al instalar dependencias")
        return
    
    # Cambiar al directorio del script
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    # Ejecutar Streamlit
    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", "app.py"])
    except KeyboardInterrupt:
        print("\n👋 Aplicación cerrada por el usuario")
    except Exception as e:
        print(f"❌ Error al ejecutar la aplicación: {e}")

if __name__ == "__main__":
    main() 