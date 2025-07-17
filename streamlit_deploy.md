# 🚀 Guía de Deployment - Streamlit Cloud

## Streamlit Cloud (GRATIS) - RECOMENDADO

### Paso 1: Preparar el repositorio
Tu repositorio ya está listo en: https://github.com/420btc/myYolo11.git

### Paso 2: Ir a Streamlit Cloud
1. Ve a: https://share.streamlit.io/
2. Inicia sesión con tu cuenta de GitHub
3. Haz clic en "New app"

### Paso 3: Configurar la aplicación
- **Repository**: 420btc/myYolo11
- **Branch**: main
- **Main file path**: app.py
- **App URL**: Elige un nombre único (ej: yolo11-detector)

### Paso 4: Deploy automático
- Streamlit Cloud detectará automáticamente `requirements.txt`
- Instalará todas las dependencias
- Desplegará tu aplicación

### Paso 5: Acceder a tu app
Tu aplicación estará disponible en:
`https://[tu-app-name].streamlit.app/`

## Ventajas de Streamlit Cloud:
✅ Gratis para repositorios públicos
✅ Deployment automático desde GitHub
✅ Soporte nativo para Streamlit
✅ Escalado automático
✅ SSL incluido
✅ Actualizaciones automáticas con git push

## Alternativas (más complejas):

### Railway (Gratis con límites)
1. Ve a: https://railway.app/
2. Conecta tu repositorio GitHub
3. Railway detectará automáticamente que es Python
4. Desplegará automáticamente

### Render (Gratis con límites)
1. Ve a: https://render.com/
2. Conecta tu repositorio GitHub
3. Selecciona "Web Service"
4. Configura:
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `streamlit run app.py --server.port $PORT --server.address 0.0.0.0`

### Heroku (Requiere configuración adicional)
Necesita archivos adicionales:
- `Procfile`: `web: streamlit run app.py --server.port $PORT --server.address 0.0.0.0`
- `runtime.txt`: `python-3.11.0` 