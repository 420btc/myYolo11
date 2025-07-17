import streamlit as st
import tempfile
import os
import numpy as np
from PIL import Image
import threading
import time

# Intentar importar dependencias con manejo de errores
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError as e:
    st.error(f"Error importando YOLO: {e}")
    YOLO_AVAILABLE = False

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError as e:
    st.error(f"Error importando OpenCV: {e}")
    CV2_AVAILABLE = False

def main():
    # Configuración de la página
    st.set_page_config(
        page_title="YOLO11 Object Detection",
        page_icon="🎯",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Título principal
    st.title("🎯 YOLO11 Detección de Objetos en Tiempo Real")
    st.markdown("### Powered by Ultralytics YOLO11")
    
    # Verificar dependencias
    if not YOLO_AVAILABLE:
        st.error("❌ YOLO11 no está disponible. Verifica que 'ultralytics' esté instalado.")
        st.info("💡 Instala las dependencias: `pip install ultralytics`")
        return
    
    if not CV2_AVAILABLE:
        st.error("❌ OpenCV no está disponible. Verifica que 'opencv-python-headless' esté instalado.")
        st.info("💡 Instala las dependencias: `pip install opencv-python-headless`")
        return
    
    # Sidebar con configuración
    st.sidebar.header("⚙️ Configuración")
    
    # Selección del modelo
    model_options = {
        "YOLO11n (Nano - Más rápido)": "yolo11n.pt",
        "YOLO11s (Small)": "yolo11s.pt", 
        "YOLO11m (Medium)": "yolo11m.pt",
        "YOLO11l (Large)": "yolo11l.pt",
        "YOLO11x (Extra Large - Más preciso)": "yolo11x.pt"
    }
    
    selected_model_name = st.sidebar.selectbox(
        "🔧 Seleccionar Modelo YOLO11",
        options=list(model_options.keys()),
        index=0,
        help="Nano es el más rápido, Extra Large es el más preciso"
    )
    
    selected_model = model_options[selected_model_name]
    
    # Configuración de confianza
    confidence = st.sidebar.slider(
        "🎯 Umbral de Confianza",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.05,
        help="Puntuación mínima para considerar una detección válida"
    )
    
    # Configuración de IoU
    iou = st.sidebar.slider(
        "📊 Umbral IoU",
        min_value=0.0,
        max_value=1.0,
        value=0.7,
        step=0.05,
        help="Umbral de Intersección sobre Unión para filtrar detecciones superpuestas"
    )
    
    # Tipo de entrada
    st.sidebar.header("📹 Fuente de Entrada")
    input_type = st.sidebar.radio(
        "Seleccionar fuente:",
        ["Cámara Web", "Captura de Imagen", "Archivo de Video", "Imagen"],
        help="Elige entre usar tu cámara web, capturar una imagen, subir un archivo o una imagen"
    )
    
    # Información del modelo seleccionado
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"**Modelo actual:** {selected_model_name}")
    st.sidebar.markdown(f"**Confianza:** {confidence}")
    st.sidebar.markdown(f"**IoU:** {iou}")
    
    # Inicializar estado de sesión
    if 'model' not in st.session_state:
        st.session_state.model = None
    if 'model_name' not in st.session_state:
        st.session_state.model_name = None
    if 'stop_camera' not in st.session_state:
        st.session_state.stop_camera = False
    
    # Cargar modelo si es necesario
    if st.session_state.model is None or st.session_state.model_name != selected_model:
        with st.spinner(f"Cargando modelo {selected_model_name}..."):
            try:
                st.session_state.model = YOLO(selected_model)
                # Forzar uso de CPU para evitar problemas de CUDA
                st.session_state.model.to('cpu')
                st.session_state.model_name = selected_model
                st.success(f"✅ Modelo {selected_model_name} cargado exitosamente (CPU)")
            except Exception as e:
                st.error(f"❌ Error cargando modelo: {str(e)}")
                st.session_state.model = None
    
    # Área principal
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("🚀 Detección de Objetos")
        
        if input_type == "Imagen":
            handle_image_input(confidence, iou)
        elif input_type == "Archivo de Video":
            handle_video_input(confidence, iou)
        elif input_type == "Captura de Imagen":
            handle_camera_capture(confidence, iou)
        else:  # Cámara Web
            handle_webcam_input(confidence, iou)
    
    with col2:
        st.header("ℹ️ Información")
        
        st.markdown("""
        **🎯 Modelos Disponibles:**
        - **Nano**: Más rápido, menor precisión
        - **Small**: Equilibrio velocidad-precisión
        - **Medium**: Buena precisión general
        - **Large**: Alta precisión
        - **Extra Large**: Máxima precisión
        
        **⚙️ Parámetros:**
        - **Confianza**: Umbral mínimo para detecciones
        - **IoU**: Filtro para detecciones superpuestas
        
        **📋 Clases Detectadas:**
        YOLO11 puede detectar 80 clases diferentes incluyendo:
        - Personas, animales
        - Vehículos (autos, motos, etc.)
        - Objetos cotidianos
        - Y muchos más...
        """)
        
        st.markdown("---")
        st.markdown("**🔧 Instrucciones:**")
        st.markdown("""
        1. Selecciona el modelo YOLO11
        2. Ajusta la confianza e IoU
        3. Elige la fuente de entrada
        4. Haz clic en el botón correspondiente
        """)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center'>
        <p>🚀 Powered by <strong>Ultralytics YOLO11</strong> | 
        📚 Built with <strong>Streamlit</strong></p>
    </div>
    """, unsafe_allow_html=True)

def handle_image_input(confidence, iou):
    """Manejar entrada de imagen"""
    st.subheader("📸 Detección en Imagen")
    
    uploaded_file = st.file_uploader(
        "Seleccionar imagen",
        type=['png', 'jpg', 'jpeg'],
        help="Sube una imagen para detección de objetos"
    )
    
    if uploaded_file is not None:
        if st.session_state.model is None:
            st.warning("⚠️ Por favor, espera a que se cargue el modelo")
            return
        
        # Mostrar imagen original
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Original")
            image = Image.open(uploaded_file)
            st.image(image, use_container_width=True)
        
        with col2:
            st.subheader("Detecciones")
            
            if st.button("🔍 Detectar Objetos", type="primary"):
                with st.spinner("Procesando imagen..."):
                    try:
                        # Ejecutar inferencia
                        results = st.session_state.model(image, conf=confidence, iou=iou, device='cpu')
                        
                        # Mostrar imagen con detecciones
                        annotated_image = results[0].plot()
                        annotated_pil = Image.fromarray(annotated_image)
                        st.image(annotated_pil, use_container_width=True)
                        
                        # Mostrar resultados
                        if len(results[0].boxes) > 0:
                            st.success(f"✅ Detectados {len(results[0].boxes)} objetos")
                            
                            # Detalles de detección
                            with st.expander("Detalles de Detección"):
                                for i, box in enumerate(results[0].boxes):
                                    class_id = int(box.cls[0])
                                    conf = float(box.conf[0])
                                    class_name = results[0].names[class_id]
                                    st.write(f"**{i+1}.** {class_name} - Confianza: {conf:.2f}")
                        else:
                            st.info("No se detectaron objetos")
                            
                    except Exception as e:
                        st.error(f"❌ Error procesando imagen: {str(e)}")

def handle_video_input(confidence, iou):
    """Manejar entrada de video"""
    st.subheader("📹 Detección en Video")
    
    uploaded_file = st.file_uploader(
        "Seleccionar archivo de video",
        type=['mp4', 'avi', 'mov', 'mkv', 'wmv'],
        help="Sube un archivo de video para detección de objetos"
    )
    
    if uploaded_file is not None:
        if st.session_state.model is None:
            st.warning("⚠️ Por favor, espera a que se cargue el modelo")
            return
        
        # Guardar archivo temporalmente
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
            tmp_file.write(uploaded_file.read())
            temp_path = tmp_file.name
        
        st.success(f"✅ Archivo subido: {uploaded_file.name}")
        
        if st.button("🎬 Procesar Video", type="primary"):
            with st.spinner("Procesando video..."):
                try:
                    # Ejecutar inferencia en video
                    results = st.session_state.model(temp_path, conf=confidence, iou=iou, save=True, device='cpu')
                    
                    st.success("✅ ¡Video procesado exitosamente!")
                    st.info("🎥 El video con detecciones se ha guardado en la carpeta 'runs'")
                    
                except Exception as e:
                    st.error(f"❌ Error al procesar el video: {str(e)}")
                finally:
                    # Limpiar archivo temporal
                    if os.path.exists(temp_path):
                        os.unlink(temp_path)

def init_camera():
    """Inicializar la cámara con múltiples métodos"""
    try:
        # Método 1: Índice 0 (cámara principal)
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret and frame is not None:
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                return cap, "✅ Cámara principal inicializada"
            cap.release()
        
        # Método 2: Probar otros índices
        for i in range(1, 4):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret and frame is not None:
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                    return cap, f"✅ Cámara {i} inicializada"
                cap.release()
        
        # Método 3: Probar con diferentes backends
        backends = [cv2.CAP_DSHOW, cv2.CAP_V4L2, cv2.CAP_GSTREAMER]
        for backend in backends:
            try:
                cap = cv2.VideoCapture(0, backend)
                if cap.isOpened():
                    ret, frame = cap.read()
                    if ret and frame is not None:
                        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                        return cap, f"✅ Cámara inicializada con backend {backend}"
                    cap.release()
            except:
                continue
        
        return None, "❌ No se encontró ninguna cámara disponible"
    except Exception as e:
        return None, f"❌ Error al inicializar la cámara: {str(e)}"

def handle_camera_capture(confidence, iou):
    """Capturar imagen desde la cámara web usando Streamlit"""
    st.subheader("📸 Captura de Imagen desde Cámara")
    
    if st.session_state.model is None:
        st.warning("⚠️ Por favor, espera a que se cargue el modelo")
        return
    
    st.info("""
    📷 **Captura de imagen desde cámara web:**
    - Esta función usa la cámara web del navegador
    - Funciona mejor que el streaming de video
    - Compatible con Streamlit Cloud
    """)
    
    # Usar st.camera_input para capturar imagen
    camera_image = st.camera_input("Toma una foto para detectar objetos")
    
    if camera_image is not None:
        # Convertir la imagen capturada
        image = Image.open(camera_image)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Imagen Original")
            st.image(image, use_container_width=True)
        
        with col2:
            st.subheader("Detecciones")
            
            if st.button("🔍 Detectar Objetos", type="primary", key="detect_camera"):
                with st.spinner("Procesando imagen capturada..."):
                    try:
                        # Ejecutar inferencia
                        results = st.session_state.model(image, conf=confidence, iou=iou, device='cpu')
                        
                        # Mostrar imagen con detecciones
                        annotated_image = results[0].plot()
                        annotated_pil = Image.fromarray(annotated_image)
                        st.image(annotated_pil, use_container_width=True)
                        
                        # Mostrar resultados
                        if len(results[0].boxes) > 0:
                            st.success(f"✅ Detectados {len(results[0].boxes)} objetos")
                            
                            # Detalles de detección
                            with st.expander("Detalles de Detección"):
                                for i, box in enumerate(results[0].boxes):
                                    class_id = int(box.cls[0])
                                    conf = float(box.conf[0])
                                    class_name = results[0].names[class_id]
                                    st.write(f"**{i+1}.** {class_name} - Confianza: {conf:.2f}")
                        else:
                            st.info("No se detectaron objetos")
                            
                    except Exception as e:
                        st.error(f"❌ Error procesando imagen: {str(e)}")

def handle_webcam_input(confidence, iou):
    """Manejar entrada de cámara web en tiempo real con streaming automático"""
    st.subheader("📷 Detección en Cámara Web - Streaming Continuo")
    
    if st.session_state.model is None:
        st.warning("⚠️ Por favor, espera a que se cargue el modelo")
        return
    
    # Información sobre la funcionalidad
    st.info("""
    🎥 **Streaming automático en tiempo real:**
    - Captura y procesa frames automáticamente cada 2 segundos
    - Detección continua sin necesidad de presionar botones
    - Usa la cámara web del navegador
    - Compatible con Streamlit Cloud
    """)
    
    # Inicializar estados de sesión
    if 'streaming_active' not in st.session_state:
        st.session_state.streaming_active = False
    if 'stream_frame_count' not in st.session_state:
        st.session_state.stream_frame_count = 0
    if 'last_capture_time' not in st.session_state:
        st.session_state.last_capture_time = 0
    
    # Botones de control
    col1, col2 = st.columns(2)
    
    with col1:
        if not st.session_state.streaming_active:
            start_button = st.button("🚀 Iniciar Streaming", type="primary")
        else:
            start_button = False
    
    with col2:
        if st.session_state.streaming_active:
            stop_button = st.button("⏹️ Detener Streaming", type="secondary")
        else:
            stop_button = False
    
    if stop_button:
        st.session_state.streaming_active = False
        st.session_state.stream_frame_count = 0
        st.success("📷 Streaming detenido")
        st.rerun()
    
    if start_button:
        st.session_state.streaming_active = True
        st.session_state.stream_frame_count = 0
        st.session_state.last_capture_time = time.time()
        st.rerun()
    
    # Si el streaming está activo
    if st.session_state.streaming_active:
        st.markdown("---")
        st.subheader("📹 Streaming Activo")
        
        # Crear contenedores para mostrar el streaming
        stream_container = st.container()
        stats_container = st.container()
        
        with stream_container:
            # Auto-captura cada 2 segundos
            current_time = time.time()
            if current_time - st.session_state.last_capture_time >= 2.0:
                st.session_state.last_capture_time = current_time
                st.session_state.stream_frame_count += 1
                
                # Usar camera_input con key única para auto-captura
                camera_key = f"stream_{st.session_state.stream_frame_count}"
                
                # Mostrar cámara para captura automática
                camera_image = st.camera_input(
                    f"📸 Frame #{st.session_state.stream_frame_count} - Captura automática cada 2 segundos",
                    key=camera_key,
                    help="La cámara captura automáticamente para análisis continuo"
                )
                
                if camera_image is not None:
                    # Procesar la imagen capturada
                    image = Image.open(camera_image)
                    
                    # Ejecutar inferencia
                    with st.spinner("Analizando frame..."):
                        try:
                            results = st.session_state.model(image, conf=confidence, iou=iou, device='cpu')
                            
                            # Mostrar imagen con detecciones
                            annotated_image = results[0].plot()
                            annotated_pil = Image.fromarray(annotated_image)
                            
                            # Mostrar resultado
                            st.image(annotated_pil, use_container_width=True)
                            
                            # Mostrar detecciones
                            detections = len(results[0].boxes)
                            
                            if detections > 0:
                                st.success(f"✅ Frame #{st.session_state.stream_frame_count}: {detections} objetos detectados")
                                
                                # Detalles de detección
                                with st.expander("📋 Detalles de Detección"):
                                    for i, box in enumerate(results[0].boxes):
                                        class_id = int(box.cls[0])
                                        conf = float(box.conf[0])
                                        class_name = results[0].names[class_id]
                                        st.write(f"**{i+1}.** {class_name} - Confianza: {conf:.2f}")
                            else:
                                st.info(f"Frame #{st.session_state.stream_frame_count}: No se detectaron objetos")
                            
                        except Exception as e:
                            st.error(f"❌ Error procesando frame: {str(e)}")
            
            # Auto-refresh para continuar el streaming
            if st.session_state.streaming_active:
                time.sleep(0.5)  # Pausa más corta para mejor responsividad
                st.rerun()
        
        # Estadísticas en tiempo real
        with stats_container:
            st.markdown("---")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Frames procesados", st.session_state.stream_frame_count)
            with col2:
                st.metric("Tiempo transcurrido", f"{int(time.time() - st.session_state.last_capture_time)}s")
            with col3:
                st.metric("Intervalo", "2 segundos")
        
        # Instrucciones
        st.markdown("---")
        st.info("""
        💡 **Streaming activo:**
        - La cámara captura automáticamente cada 2 segundos
        - Los frames se procesan y analizan automáticamente
        - Presiona "Detener Streaming" para finalizar
        - El streaming continúa hasta que lo detengas manualmente
        """)
    
    else:
        # Mostrar información cuando no está activo
        st.markdown("---")
        st.markdown("""
        ### 🎯 Diferencias entre funcionalidades:
        
        **📹 Cámara Web (Streaming):**
        - **Captura automática** cada 2 segundos
        - **Análisis continuo** sin intervención manual
        - **Streaming en tiempo real** hasta que lo detengas
        - Ideal para **monitoreo continuo**
        
        **📸 Captura de Imagen:**
        - **Una sola captura** manual
        - **Análisis detallado** de imagen individual
        - **Control total** sobre cuándo capturar
        - Ideal para **análisis específico**
        
        **💡 Usa Streaming para monitoreo continuo, Captura para análisis puntual**
        """)

if __name__ == "__main__":
    main()
