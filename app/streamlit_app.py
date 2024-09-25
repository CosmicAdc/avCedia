import streamlit as st
import os
from langchain_code.utils import split_document
from langchain_community.document_loaders import PyPDFLoader

# Directorio para guardar los archivos subidos
UPLOAD_DIRECTORY = "app/static"

# Función para procesar la subida del archivo
def process_document(uploaded_file):
    if uploaded_file is not None:
        file_name = uploaded_file.name
        file_path = os.path.join(UPLOAD_DIRECTORY, file_name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        if file_name.endswith(".pdf"):
            loader = PyPDFLoader(file_path)
        elif file_name.endswith(".docx") or file_name.endswith(".doc"):
            loader = Docx2txtLoader(file_path)
        else:  
            loader = load_document(file_path)
        
        doc = loader.load()
        all_chunks = []
        for doc_page in doc:
            chunks = split_document(doc_page.page_content)
            all_chunks.extend(chunks)
        return all_chunks
    return []

# Configuración de la página de Streamlit
st.set_page_config(page_title="Generador de Preguntas y Respuestas", layout="wide")

st.title("Generador de Preguntas y Respuestas basado en LLM")

# Subida de archivo
uploaded_file = st.file_uploader("Sube tu documento", type=["txt", "pdf", "docx", "doc"])

# Procesar documento y almacenar chunks en session_state solo si no se ha procesado antes
if "chunks" not in st.session_state:
    st.session_state["chunks"] = []

if uploaded_file is not None and not st.session_state["chunks"]:
    chunks = process_document(uploaded_file)
    st.session_state["chunks"] = chunks  # Guardar los chunks procesados en session_state

chunks = st.session_state["chunks"]

# Función para eliminar un chunk
def eliminar_chunk(idx):
    # Usamos una lista temporal para no modificar el session_state directamente
    temp_chunks = st.session_state["chunks"]
    if 0 <= idx < len(temp_chunks):
        temp_chunks.pop(idx)
        st.session_state["chunks"] = temp_chunks  # Guardar cambios en session_state sin recargar todo
        st.rerun()  # Recargar solo para reflejar la eliminación

# Mostrar chunks si existen
if len(chunks) > 0:
    st.write(f"Documento dividido en {len(chunks)} chunks.")

    for idx, chunk in enumerate(chunks):
        col1, col2, col3, col4, col5, col6 = st.columns([2, 2, 2, 1, 1, 1]) 

        with col1:
            st.text_area(f"Texto del Chunk {idx+1}", chunk, height=200)
        
        with col2:
            st.text_area(f"Pregunta Generada para Chunk {idx+1}", height=200)

        with col3:
            st.text_area(f"Respuesta Generada para Chunk {idx+1}", height=200)

        with col4:
            selected_relevance = st.selectbox("Relevancia", ["Alta", "Mediana", "Baja"], index=0, key=f"relevance_{idx}")

        with col5:
            st.button("Reintentar", key=f"retry_{idx}")
            st.button("Guardar", key=f"save_{idx}")

        with col6: 
            if st.button("Eliminar", key=f"delete_{idx}"):
                eliminar_chunk(idx)  # Llamar a la función de eliminación

else:
    st.write("Esperando que se suba un documento...")
