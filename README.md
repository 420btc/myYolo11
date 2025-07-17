# 🎯 YOLO11 Object Detection App

Una aplicación web simple y moderna para detección de objetos en tiempo real usando YOLO11 y Streamlit.

## 🚀 Características

- **Detección en tiempo real** con cámara web
- **Procesamiento de videos** subidos
- **5 modelos YOLO11** diferentes (Nano a Extra Large)
- **Configuración ajustable** de confianza e IoU
- **Interfaz moderna** y fácil de usar
- **Detección de 80+ clases** de objetos

## 📋 Requisitos

- Python 3.8 o superior
- Cámara web (opcional, para detección en tiempo real)

## 🛠️ Instalación

### Opción 1: Instalación Automática
```bash
python run.py
```

### Opción 2: Instalación Manual
```bash
# Instalar dependencias
pip install -r requirements.txt

# Ejecutar la aplicación
streamlit run app.py
```

## 🎮 Uso

1. **Ejecuta la aplicación**:
   ```bash
   python run.py
   ```

2. **Abre tu navegador** en `http://localhost:8501`

3. **Configura los parámetros**:
   - Selecciona el modelo YOLO11
   - Ajusta la confianza (0.0 - 1.0)
   - Configura el umbral IoU

4. **Elige la fuente**:
   - **Cámara Web**: Para detección en tiempo real
   - **Archivo de Video**: Para procesar videos subidos

5. **¡Inicia la detección!**

## 🎯 Modelos Disponibles

| Modelo | Velocidad | Precisión | Uso Recomendado |
|--------|-----------|-----------|-----------------|
| YOLO11n | ⚡⚡⚡⚡⚡ | ⭐⭐⭐ | Tiempo real, recursos limitados |
| YOLO11s | ⚡⚡⚡⚡ | ⭐⭐⭐⭐ | Equilibrio velocidad-precisión |
| YOLO11m | ⚡⚡⚡ | ⭐⭐⭐⭐⭐ | Buena precisión general |
| YOLO11l | ⚡⚡ | ⭐⭐⭐⭐⭐⭐ | Alta precisión |
| YOLO11x | ⚡ | ⭐⭐⭐⭐⭐⭐⭐ | Máxima precisión |

## 🔧 Configuración

### Parámetros Principales

- **Confianza**: Umbral mínimo para considerar una detección válida (0.0-1.0)
- **IoU**: Umbral de Intersección sobre Unión para filtrar detecciones superpuestas (0.0-1.0)

### Clases Detectadas

La aplicación puede detectar 80 clases diferentes del dataset COCO:
- Personas
- Animales (perros, gatos, caballos, etc.)
- Vehículos (autos, motos, bicicletas, etc.)
- Objetos cotidianos (teléfonos, laptops, etc.)
- Y muchos más...

## 📁 Estructura del Proyecto

```
YOLODetector/
├── app.py              # Aplicación principal
├── run.py              # Script de inicio
├── requirements.txt    # Dependencias
├── README.md          # Este archivo
├── .streamlit/        # Configuración de Streamlit
│   └── config.toml
└── runs/              # Videos procesados (se crea automáticamente)
```

## 🐛 Solución de Problemas

### Error: "No se pudo abrir la cámara"
- Verifica que tu cámara esté conectada
- Cierra otras aplicaciones que puedan estar usando la cámara
- Prueba con un archivo de video primero

### Error de CUDA/GPU
- La aplicación está configurada para usar CPU por defecto
- Si tienes problemas con CUDA, reinicia la aplicación
- El rendimiento en CPU es suficiente para la mayoría de casos

### Error: "Dependencia faltante"
```bash
pip install -r requirements.txt
```

### La aplicación se ejecuta lentamente
- Usa un modelo más pequeño (YOLO11n o YOLO11s)
- Aumenta el umbral de confianza
- Verifica que tienes suficiente RAM

## 🤝 Contribuciones

¡Las contribuciones son bienvenidas! Si encuentras algún problema o tienes sugerencias:

1. Abre un issue
2. Crea un pull request
3. Comparte tus ideas

## 📄 Licencia

Este proyecto está bajo la licencia MIT. Consulta el archivo LICENSE para más detalles.

## 🙏 Agradecimientos

- [Ultralytics](https://ultralytics.com/) por YOLO11
- [Streamlit](https://streamlit.io/) por el framework web
- [OpenCV](https://opencv.org/) por el procesamiento de video

---

**¡Disfruta detectando objetos con YOLO11!** 🎯✨ 