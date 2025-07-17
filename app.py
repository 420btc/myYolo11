import streamlit as st
import numpy as np
from PIL import Image
import tempfile
import os
import time

# Try to import dependencies
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False

def main():
    # Page configuration
    st.set_page_config(
        page_title="YOLO11 Object Detection",
        page_icon="🎯",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Main title
    st.title("🎯 YOLO11 Real-Time Object Detection")
    st.markdown("### Powered by Ultralytics YOLO11")
    
    # Check dependencies
    missing_deps = []
    if not CV2_AVAILABLE:
        missing_deps.append("opencv-python")
    if not ULTRALYTICS_AVAILABLE:
        missing_deps.append("ultralytics")
    
    if missing_deps:
        st.error(f"❌ Faltan dependencias: {', '.join(missing_deps)}")
        st.markdown("### Instalación de Dependencias")
        st.code("pip install ultralytics opencv-python", language="bash")
        
        # Show demo interface anyway
        st.warning("⚠️ Mostrando interfaz de demostración. Instala las dependencias para funcionalidad completa.")
        show_demo_interface()
        return
    
    # Sidebar configuration
    st.sidebar.header("Configuración")
    
    # Model selection
    model_options = {
        "YOLO11n (Nano)": "yolo11n.pt",
        "YOLO11s (Small)": "yolo11s.pt", 
        "YOLO11m (Medium)": "yolo11m.pt",
        "YOLO11l (Large)": "yolo11l.pt",
        "YOLO11x (Extra Large)": "yolo11x.pt"
    }
    
    selected_model = st.sidebar.selectbox(
        "Seleccionar Modelo YOLO",
        options=list(model_options.keys()),
        index=0,
        help="Elige la variante del modelo YOLO11. Nano es el más rápido, Extra Large es el más preciso."
    )
    
    # Confidence threshold
    confidence_threshold = st.sidebar.slider(
        "Umbral de Confianza",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.05,
        help="Puntuación mínima de confianza para las detecciones"
    )
    
    # Input source selection
    st.sidebar.header("Fuente de Entrada")
    input_source = st.sidebar.radio(
        "Elegir fuente de entrada:",
        ["Cámara Web", "Archivo de Video", "Imagen"],
        help="Selecciona si usar cámara web, subir video o imagen"
    )
    
    # Initialize session state
    if 'model' not in st.session_state:
        st.session_state.model = None
    if 'inference_running' not in st.session_state:
        st.session_state.inference_running = False
    if 'video_file_path' not in st.session_state:
        st.session_state.video_file_path = None
    
    # Load model button
    if st.sidebar.button("🔄 Cargar Modelo"):
        load_model(model_options[selected_model])
    
    # Main content area
    if input_source == "Imagen":
        handle_image_input(confidence_threshold)
    elif input_source == "Archivo de Video":
        handle_video_input(confidence_threshold)
    else:
        handle_webcam_input(confidence_threshold)
    
    # Footer with information
    st.markdown("---")
    st.markdown("""
    **Características:**
    - Detección de objetos en tiempo real usando modelos YOLO11
    - Soporte para cámara web, archivos de video e imágenes
    - Umbrales de confianza ajustables
    - Procesamiento de video en vivo con anotaciones
    
    **Modelos:** YOLO11n (más rápido) a YOLO11x (más preciso)
    """)

def load_model(model_path):
    """Load YOLO model"""
    try:
        with st.spinner(f"Cargando modelo {model_path}..."):
            st.session_state.model = YOLO(model_path)
        st.success(f"✅ Modelo {model_path} cargado exitosamente")
    except Exception as e:
        st.error(f"❌ Error cargando modelo: {str(e)}")
        st.session_state.model = None

def handle_image_input(confidence_threshold):
    """Handle image upload and processing"""
    st.header("Detección en Imagen")
    
    uploaded_file = st.file_uploader(
        "Subir una imagen",
        type=['png', 'jpg', 'jpeg'],
        help="Sube una imagen para detección de objetos"
    )
    
    if uploaded_file is not None and st.session_state.model is not None:
        # Display original image
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Imagen Original")
            image = Image.open(uploaded_file)
            st.image(image, use_column_width=True)
        
        with col2:
            st.subheader("Resultados de Detección")
            
            # Process image
            try:
                # Convert PIL image to numpy array
                img_array = np.array(image)
                
                # Run inference
                results = st.session_state.model(img_array, conf=confidence_threshold)
                
                # Plot results
                annotated_image = results[0].plot()
                
                # Convert back to PIL and display
                annotated_pil = Image.fromarray(annotated_image)
                st.image(annotated_pil, use_column_width=True)
                
                # Show detection results
                if len(results[0].boxes) > 0:
                    st.success(f"✅ Detectados {len(results[0].boxes)} objetos")
                    
                    # Display detection details
                    with st.expander("Detalles de Detección"):
                        for i, box in enumerate(results[0].boxes):
                            class_id = int(box.cls[0])
                            confidence = float(box.conf[0])
                            class_name = results[0].names[class_id]
                            st.write(f"**{i+1}.** {class_name} - Confianza: {confidence:.2f}")
                else:
                    st.info("No se detectaron objetos")
                    
            except Exception as e:
                st.error(f"❌ Error procesando imagen: {str(e)}")
    
    elif uploaded_file is not None and st.session_state.model is None:
        st.warning("⚠️ Por favor, carga un modelo primero")

def handle_video_input(confidence_threshold):
    """Handle video upload and processing"""
    st.header("Detección en Video")
    
    uploaded_file = st.file_uploader(
        "Subir un archivo de video",
        type=['mp4', 'avi', 'mov', 'mkv'],
        help="Sube un archivo de video para detección de objetos"
    )
    
    if uploaded_file is not None:
        # Save uploaded file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
            tmp_file.write(uploaded_file.read())
            st.session_state.video_file_path = tmp_file.name
        
        st.success(f"Video subido: {uploaded_file.name}")
        
        if st.session_state.model is not None:
            if st.button("🚀 Procesar Video"):
                process_video(st.session_state.video_file_path, confidence_threshold)
        else:
            st.warning("⚠️ Por favor, carga un modelo primero")

def handle_webcam_input(confidence_threshold):
    """Handle webcam input"""
    st.header("Detección en Cámara Web")
    
    if st.session_state.model is not None:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Cámara")
            camera_placeholder = st.empty()
        
        with col2:
            st.subheader("Detecciones")
            detection_placeholder = st.empty()
        
        # Control buttons
        col_start, col_stop = st.columns(2)
        
        with col_start:
            if st.button("🚀 Iniciar Cámara", disabled=st.session_state.inference_running):
                st.session_state.inference_running = True
                st.rerun()
        
        with col_stop:
            if st.button("⏹️ Detener Cámara", disabled=not st.session_state.inference_running):
                st.session_state.inference_running = False
                st.rerun()
        
        # Process webcam if running
        if st.session_state.inference_running:
            process_webcam(camera_placeholder, detection_placeholder, confidence_threshold)
    else:
        st.warning("⚠️ Por favor, carga un modelo primero")

def process_video(video_path, confidence_threshold):
    """Process video file"""
    try:
        cap = cv2.VideoCapture(video_path)
        
        # Video info
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        st.info(f"📹 Procesando video: {fps} FPS, {total_frames} frames")
        
        # Create placeholders
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Original")
            original_placeholder = st.empty()
        
        with col2:
            st.subheader("Detecciones")
            detection_placeholder = st.empty()
        
        progress_bar = st.progress(0)
        frame_count = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Update progress
            frame_count += 1
            progress = frame_count / total_frames
            progress_bar.progress(progress)
            
            # Process every 5th frame to reduce processing load
            if frame_count % 5 == 0:
                # Run inference
                results = st.session_state.model(frame, conf=confidence_threshold)
                annotated_frame = results[0].plot()
                
                # Convert BGR to RGB for display
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                annotated_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                
                # Display frames
                original_placeholder.image(frame_rgb, channels="RGB", use_column_width=True)
                detection_placeholder.image(annotated_rgb, channels="RGB", use_column_width=True)
                
                # Small delay to control playback speed
                time.sleep(0.1)
        
        cap.release()
        progress_bar.progress(1.0)
        st.success("✅ Video procesado completamente")
        
    except Exception as e:
        st.error(f"❌ Error procesando video: {str(e)}")

def process_webcam(camera_placeholder, detection_placeholder, confidence_threshold):
    """Process webcam stream"""
    try:
        # Try to open webcam
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            st.error("❌ No se pudo abrir la cámara")
            return
        
        # Process frames
        frame_count = 0
        while st.session_state.inference_running:
            ret, frame = cap.read()
            if not ret:
                st.error("❌ No se pudo leer de la cámara")
                break
            
            frame_count += 1
            
            # Process every 3rd frame to reduce load
            if frame_count % 3 == 0:
                # Run inference
                results = st.session_state.model(frame, conf=confidence_threshold)
                annotated_frame = results[0].plot()
                
                # Convert BGR to RGB for display
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                annotated_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                
                # Display frames
                camera_placeholder.image(frame_rgb, channels="RGB", use_column_width=True)
                detection_placeholder.image(annotated_rgb, channels="RGB", use_column_width=True)
            
            # Small delay
            time.sleep(0.03)
        
        cap.release()
        
    except Exception as e:
        st.error(f"❌ Error con la cámara: {str(e)}")
        st.session_state.inference_running = False

def show_demo_interface():
    """Show demo interface when dependencies are missing"""
    # Sidebar configuration
    st.sidebar.header("Configuración")
    
    # Model selection
    model_options = {
        "YOLO11n (Nano)": "yolo11n.pt",
        "YOLO11s (Small)": "yolo11s.pt", 
        "YOLO11m (Medium)": "yolo11m.pt",
        "YOLO11l (Large)": "yolo11l.pt",
        "YOLO11x (Extra Large)": "yolo11x.pt"
    }
    
    selected_model = st.sidebar.selectbox(
        "Seleccionar Modelo YOLO",
        options=list(model_options.keys()),
        index=0,
        help="Elige la variante del modelo YOLO11. Nano es el más rápido, Extra Large es el más preciso."
    )
    
    # Confidence threshold
    confidence_threshold = st.sidebar.slider(
        "Umbral de Confianza",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.05,
        help="Puntuación mínima de confianza para las detecciones"
    )
    
    # Input source selection
    st.sidebar.header("Fuente de Entrada")
    input_source = st.sidebar.radio(
        "Elegir fuente de entrada:",
        ["Cámara Web", "Archivo de Video", "Imagen"],
        help="Selecciona si usar cámara web, subir video o imagen"
    )
    
    # Main content area
    st.header("Interfaz de Demostración")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Entrada")
        if input_source == "Imagen":
            st.info("📸 Aquí subirías una imagen para detectar objetos")
        elif input_source == "Archivo de Video":
            st.info("📹 Aquí subirías un video para procesar")
        else:
            st.info("📷 Aquí se mostraría la cámara web")
    
    with col2:
        st.subheader("Resultados")
        st.info("🎯 Aquí se mostrarían los objetos detectados con anotaciones")
    
    # Control buttons
    st.header("Controles")
    col_start, col_stop = st.columns(2)
    
    with col_start:
        if st.button("🚀 Iniciar Detección"):
            st.success(f"Demo: Iniciaría detección con {selected_model} (confianza: {confidence_threshold})")
    
    with col_stop:
        if st.button("⏹️ Detener"):
            st.info("Demo: Detección detenida")
    
    # Implementation guide
    st.markdown("---")
    st.markdown("### Guía de Implementación Completa")
    
    # Installation instructions
    st.subheader("1. Instalación de Dependencias")
    st.code("""
# Instalar dependencias requeridas
pip install ultralytics opencv-python pillow torch torchvision streamlit

# O usando conda
conda install -c conda-forge ultralytics opencv pillow pytorch torchvision streamlit
""", language="bash")
    
    # Code example
    st.subheader("2. Código de Ejemplo")
    st.code("""
from ultralytics import YOLO
import cv2
import streamlit as st

# Cargar modelo
model = YOLO('yolo11n.pt')

# Procesar imagen
results = model(image, conf=0.5)
annotated_image = results[0].plot()

# Mostrar resultados
st.image(annotated_image)
""", language="python")
    
    # Features overview
    st.subheader("3. Características Disponibles")
    st.markdown("""
    **Cuando las dependencias estén instaladas:**
    - ✅ Detección de objetos en tiempo real
    - ✅ Soporte para 80+ clases de objetos (COCO dataset)
    - ✅ Procesamiento de imágenes, videos y cámara web
    - ✅ 5 modelos diferentes (velocidad vs precisión)
    - ✅ Controles de confianza ajustables
    - ✅ Anotaciones automáticas con cajas delimitadoras
    - ✅ Información detallada de cada detección
    """)

if __name__ == "__main__":
    main()
