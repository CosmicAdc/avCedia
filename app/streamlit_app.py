import streamlit as st
import os
from langchain_code.utils import split_document, load_document
from langchain_code.chains import batch_call_chain, invoke_call_chain,invoke_call_chain_answer,batch_call_chain_answer
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import Docx2txtLoader
import pandas as pd  # Asegúrate de importar pandas

# Directorio para guardar los archivos subidos
UPLOAD_DIRECTORY = "app/static"
NUM_CARACTERES_ELIMINAR= 80
NOMBRE_ARCHIVO_CSV = "QA_FINETUNNING_AV.csv"

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
            chunks = [chunk for chunk in chunks if len(chunk.page_content) >= NUM_CARACTERES_ELIMINAR]
            all_chunks.extend(chunks)
        return all_chunks
    return []

# Configuración de la página de Streamlit
st.set_page_config(page_title="Generador de Preguntas y Respuestas", layout="wide")

st.title("Generador de Preguntas y Respuestas basado en LLM")

# Subida de archivo
uploaded_file = st.file_uploader("Sube tu documento", type=["txt", "pdf", "docx", "doc"])
importancia = st.selectbox("Importancia del documento:", ["Alta", "Mediana", "Baja"], index=0)

# Procesar documento y almacenar chunks en session_state solo si no se ha procesado antes
if "chunks" not in st.session_state:
    st.session_state["chunks"] = []

if uploaded_file is not None and not st.session_state["chunks"]:
    chunks = process_document(uploaded_file)
    # Agregar importancia a cada chunk
    for chunk in chunks:
        chunk.metadata["importancia"] = importancia 
    st.session_state["chunks"] = chunks # Guardar los chunks procesados en session_state

chunks = st.session_state["chunks"]

# Inicializar session_state para preguntas si no existe
if "generated_questions" not in st.session_state:
    st.session_state["generated_questions"] = {}

if "generated_answers" not in st.session_state:
    st.session_state["generated_answers"] = {}


# Función para eliminar un chunk
def eliminar_chunk(idx):
    temp_chunks = st.session_state["chunks"]
    if 0 <= idx < len(temp_chunks):
        temp_chunks.pop(idx)  # Elimina el chunk por índice
        st.session_state["chunks"] = temp_chunks  # Guardar los cambios
        # Elimina la pregunta generada asociada si existe
        if idx in st.session_state["generated_questions"]:
            del st.session_state["generated_questions"][idx]
        # Forzar el refresco de los índices para las preguntas generadas
        st.session_state["generated_questions"] = {
            new_idx: q for new_idx, (old_idx, q) in enumerate(st.session_state["generated_questions"].items()) if old_idx != idx
        } # Eliminar también la pregunta generada

# Función para generar pregunta y actualizar en session_state
def call_invoke_question(idx, chunk):
    question = invoke_call_chain(chunk)  # Aquí haces la llamada para generar la pregunta
    st.session_state["generated_questions"][idx] = question  # Guardar en session_state

def call_batch_invoke_question(chunks):
    docs = [{"context": chunk.page_content} for chunk in chunks]  # Prepara los docs en el formato esperado
    questions = batch_call_chain(docs)  # Llama a la función batch
    return questions

def call_invoke_answer(idx, chunk, question):
    answer = invoke_call_chain_answer(chunk.page_content, question) 
    st.session_state["generated_answers"][idx] = answer  # Store in session_state

def call_batch_invoke_answer(chunks, questions):
    answers = batch_call_chain_answer([
        {"context": chunk.page_content, "question": questions[idx]} 
        for idx, chunk in enumerate(chunks)
    ])
    return answers



# Mostrar chunks si existen
if len(chunks) > 0:
    st.write(f"Documento dividido en {len(chunks)} chunks.")

    for idx, chunk in enumerate(chunks):
        col1, col2, col3, col4, col5, col6 = st.columns([2, 2, 2, 1, 1, 1])

        with col1:
            st.text_area(f"Texto del Chunk {idx+1}", chunk.page_content, height=200)
        
        with col2:
            pregunta = st.session_state["generated_questions"].get(idx, "")
            st.text_area(f"Pregunta Generada para Chunk {idx+1}", value=pregunta, height=200)

        with col3:
            respuesta = st.session_state["generated_answers"].get(idx, "")
            st.text_area(f"Respuesta Generada para Chunk {idx+1}", value=respuesta, height=200)


        with col4:
            importancia_seleccionada = st.session_state["chunks"][idx].metadata.get("importancia", "No definida")
            nueva_importancia = st.selectbox(
                f"Importancia del Chunk {idx+1}:", 
                ["Alta", "Mediana", "Baja"], 
                index=["Alta", "Mediana", "Baja"].index(importancia_seleccionada)
            )
            st.session_state["chunks"][idx].metadata["importancia"] = nueva_importancia

        with col5:
            if st.button("Generar Pregunta", key=f"generate_{idx}"):
                call_invoke_question(idx, chunk.page_content)
            if st.session_state["generated_questions"].get(idx):
                if st.button("Generar Respuesta", key=f"answer_{idx}",disabled=False):
                    call_invoke_answer(idx, chunk, st.session_state["generated_questions"][idx])
            else:
                st.button("Generar Respuesta", key=f"answer_{idx}",disabled=True)

        with col6: 
            if st.button("Eliminar", key=f"delete_{idx}"):
                eliminar_chunk(idx)
                st.rerun()
            hay_pregunta_y_respuesta = (
                idx in st.session_state["generated_questions"] and 
                st.session_state["generated_questions"][idx].strip() and
                idx in st.session_state["generated_answers"] and 
                st.session_state["generated_answers"][idx].strip()
            )
            if hay_pregunta_y_respuesta:
                if st.button("Guardar CSV", key=f"save_csv_{idx}"):
                    # Crear un diccionario con los nuevos datos
                    nuevos_datos = {
                        'context': st.session_state["chunks"][idx].page_content,
                        'importance': st.session_state["chunks"][idx].metadata.get("importancia", "No definida"),
                        'human': st.session_state["generated_questions"][idx],
                        'response': st.session_state["generated_answers"][idx]
                    }
                    
                    # Convertir en DataFrame como una fila nueva, asegurándonos que cada campo esté correctamente asignado a una columna
                    df_nuevos_datos = pd.DataFrame([nuevos_datos])

                    try:
                        # Intentar leer el CSV existente con el separador adecuado
                        df = pd.read_csv(NOMBRE_ARCHIVO_CSV)
                    except FileNotFoundError:
                        # Si el archivo no existe, creamos uno nuevo con las columnas definidas
                        df = pd.DataFrame(columns=['context', 'importance', 'human', 'response'])

                    # Concatenar los nuevos datos con el DataFrame existente
                    df = pd.concat([df, df_nuevos_datos], ignore_index=True)

                    # Guardar el DataFrame actualizado en el CSV, usando coma como delimitador
                    df.to_csv(NOMBRE_ARCHIVO_CSV, index=False, sep=';')

                    # Mensaje de éxito en la interfaz
                    st.success(f"Pregunta y respuesta del chunk {idx+1} guardadas en {NOMBRE_ARCHIVO_CSV}")



    col1_A, col2_A,col3_A = st.columns(3)
    with col1_A:
        if st.button("GENERAR PREGUNTAS"):
            questions = call_batch_invoke_question(chunks)  # Llamada a batch
            for idx, question in enumerate(questions):
                st.session_state["generated_questions"][idx] = question
        st.button("Actualizar Estado")

    with col2_A:
        # Verificar si todas las preguntas están generadas y no están vacías
        todas_las_preguntas_generadas = all(
            idx in st.session_state["generated_questions"] and 
            st.session_state["generated_questions"][idx].strip()  # Verifica si no están vacías
            for idx in range(len(chunks))
        )

        # Habilitar o deshabilitar el botón según la condición
        if todas_las_preguntas_generadas:
            if st.button("GENERAR RESPUESTAS", disabled=False):
                st.write("Generando respuestas para todos los chunks...")
                answers = call_batch_invoke_answer(chunks, st.session_state["generated_questions"])
                for idx, answer in enumerate(answers):
                    st.session_state["generated_answers"][idx] = answer
        else:
            st.button("GENERAR RESPUESTAS", disabled=True)

    with col3_A:
        # Verificar si todos los campos de preguntas y respuestas están completos directamente
        completos = all(
            idx in st.session_state["generated_questions"] and st.session_state["generated_questions"][idx].strip() and
            idx in st.session_state["generated_answers"] and st.session_state["generated_answers"][idx].strip()
            for idx in range(len(chunks))
        )

        # Botón para guardar todas las observaciones en el CSV si todos los campos están completos
      # Botón para guardar todas las observaciones en el CSV si todos los campos están completos
    if completos:
        if st.button("GUARDAR TODAS EN EL CSV", key="guardar_csv_completo"):
            data = {
                'context': [chunk.page_content for chunk in st.session_state["chunks"]],
                'importance': [chunk.metadata.get("importancia", "No definida") for chunk in st.session_state["chunks"]],
                'human': [st.session_state["generated_questions"].get(idx, "") for idx in range(len(st.session_state["chunks"]))],
                'response': [st.session_state["generated_answers"].get(idx, "") for idx in range(len(st.session_state["chunks"]))]
            }

            # Convertir los datos en un DataFrame
            df_nuevos_datos = pd.DataFrame(data)

            try:
                # Intentar leer el CSV existente
                df = pd.read_csv(NOMBRE_ARCHIVO_CSV, sep=';')
            except FileNotFoundError:
                # Si el archivo no existe, crear uno nuevo con las columnas definidas
                df = pd.DataFrame(columns=['context', 'importance', 'human', 'response'])

            # Concatenar los nuevos datos con el DataFrame existente
            df = pd.concat([df, df_nuevos_datos], ignore_index=True)

            # Guardar el DataFrame actualizado en el CSV, usando coma como delimitador
            df.to_csv(NOMBRE_ARCHIVO_CSV, index=False, sep=';')

            # Mostrar mensaje de éxito en la interfaz
            st.success(f"Todas las observaciones guardadas en {NOMBRE_ARCHIVO_CSV}")

        else:
            st.button("GUARDAR TODAS EN EL CSV", disabled=True, key="guardar_csv_completo")


else:
    st.write("Esperando que se suba un documento...")
