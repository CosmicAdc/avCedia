import streamlit as st
import pandas as pd
from langchain_code.data_augmentation_chains import invoke_call_chain,batch_call_chain

# Configuración de la página de Streamlit
st.set_page_config(page_title="Generador de Preguntas y Respuestas", layout="wide")

st.title("Generador de Preguntas y Respuestas basado en LLM")

# Cargar el archivo CSV
NOMBRE_ARCHIVO_CSV = "QA_FINETUNNING_AV_pt1.csv"
SAVE_ARCHIVO_CSV = "data_process/QA_FINETUNNING_AV_data_process.csv"
uploaded_file = st.file_uploader("Sube tu archivo CSV", type=["csv"])
st.button("Actualizar estado", key=f"update_states_2")


# Inicializamos la página actual en session_state si no existe
if "page" not in st.session_state:
    st.session_state.page = 0

# Número de observaciones por página
OBSERVACIONES_POR_PAGINA = 10

if uploaded_file is not None:
    # Leer el archivo CSV
    df = pd.read_csv(uploaded_file, sep=";")
    
    # Verificar que el CSV contiene las columnas esperadas
    if all(col in df.columns for col in ["context", "importance", "human", "response"]):
        st.write(f"Archivo CSV cargado con {len(df)} filas.")
        
        # Calcular el número de páginas
        total_pages = (len(df) - 1) // OBSERVACIONES_POR_PAGINA + 1
        
        # Mostrar solo las observaciones de la página actual
        start_idx = st.session_state.page * OBSERVACIONES_POR_PAGINA
        end_idx = start_idx + OBSERVACIONES_POR_PAGINA
        page_data = df.iloc[start_idx:end_idx]
        
        # Inicializar session_state para preguntas, respuestas y variaciones si no existen
        if "generated_questions" not in st.session_state:
            st.session_state["generated_questions"] = {}

        if "generated_answers" not in st.session_state:
            st.session_state["generated_answers"] = {}

        if "variations" not in st.session_state:
            st.session_state["variations"] = {}

        # Función para generar variaciones dinámicamente utilizando el LLM
        def generate_variations_llm(idx, question, response, importance):
            num_variations = 1 if importance == "Baja" else 3 if importance == "Mediana" else 5
            try:
                variaciones = invoke_call_chain(question, response, num_variations)
                # Desempaquetar la lista de variaciones
                st.session_state["variations"][idx] = variaciones
                st.success(f"Variaciones generadas para la fila {idx+1}")
            except Exception as e:
                st.error(f"Error al generar variaciones: {str(e)}")

        # Función para generar variaciones en batch para todas las filas visibles
        def generate_all_variations():
            batch_input = []
            
            # Preparar las entradas para el batch call
            for idx, row in page_data.iterrows():
                importance = st.session_state.get(f"importancia_{idx}", row["importance"])
                question = st.session_state.get(f"pregunta_{idx}", row["human"])
                response = st.session_state.get(f"respuesta_{idx}", row["response"])
                
                # Definir el número de variaciones basado en la importancia
                num_variations = 1 if importance == "Baja" else 3 if importance == "Mediana" else 5
                
                # Agregar la entrada al batch
                batch_input.append({"question": question, "response": response, "n": num_variations})
            
            try:
                # Llamar al método batch_call_chain para generar las variaciones en lote
                all_variations = batch_call_chain(batch_input)
                
                # Asignar las variaciones generadas a la sesión
                for idx, variaciones in enumerate(all_variations):
                    st.session_state["variations"][start_idx + idx] = variaciones
                
                st.success("Variaciones generadas para todas las filas visibles.")
            except Exception as e:
                st.error(f"Error al generar variaciones en batch: {str(e)}")

        def delete_variation(idx, variation_idx):
            if idx in st.session_state["variations"]:
                del st.session_state["variations"][idx][variation_idx]
                st.success(f"Variación {variation_idx + 1} de la fila {idx + 1} eliminada.")
        # Función para guardar los datos en el CSV
        def save_all_data():
            all_data = []
            for idx, row in page_data.iterrows():
                campos_llenos = (
                    st.session_state.get(f"context_{idx}", "").strip() and
                    st.session_state.get(f"pregunta_{idx}", "").strip() and
                    st.session_state.get(f"respuesta_{idx}", "").strip() and
                    all(st.session_state.get(f"variacion_{idx}_{j}", "").strip() for j in range(len(st.session_state["variations"].get(idx, []))))
                )

                if campos_llenos:
                    base_datos = {
                        'context': st.session_state[f"context_{idx}"],
                        'importance': st.session_state[f"importancia_{idx}"],
                        'response': st.session_state[f"respuesta_{idx}"]
                    }
                    original_datos = base_datos.copy()
                    original_datos['human'] = st.session_state[f"pregunta_{idx}"]
                    all_data.append(original_datos)

                    for j in range(len(st.session_state["variations"].get(idx, []))):
                        variacion_datos = base_datos.copy()
                        variacion_datos['human'] = st.session_state[f"variacion_{idx}_{j}"]
                        all_data.append(variacion_datos)

            df_nuevos_datos = pd.DataFrame(all_data)

            try:
                df = pd.read_csv(NOMBRE_ARCHIVO_CSV, sep=';')
            except FileNotFoundError:
                df = pd.DataFrame(columns=['context', 'importance', 'human', 'response'])

            df = pd.concat([df, df_nuevos_datos], ignore_index=True)
            csv = df.to_csv(index=False, sep=';')
            st.download_button(
            label="Descargar CSV",
            data=csv,
            file_name=f"datos_procesados_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime='text/csv',
            )

            st.success("Datos guardados en el archivo CSV.")


        # Crear vista en columnas para la página actual
        for i, (idx, row) in enumerate(page_data.iterrows()):
            col1, col2, col3, col4, col5, col6, col7 = st.columns([1, 1, 1, 1, 2, 2, 2])

            # Columna 1: Contexto
            with col1:
                st.text_area(f"Contexto {start_idx + i + 1}", row["context"], height=100, key=f"context_{idx}")

            # Columna 2: Pregunta
            with col2:
                pregunta = st.session_state["generated_questions"].get(idx, row["human"])
                st.text_area(f"Pregunta {start_idx + i + 1}", pregunta, height=100, key=f"pregunta_{idx}")

            # Columna 3: Respuesta
            with col3:
                respuesta = st.session_state["generated_answers"].get(idx, row["response"])
                st.text_area(f"Respuesta {start_idx + i + 1}", respuesta, height=100, key=f"respuesta_{idx}")

            # Columna 4: Importancia
            with col4:
                importancia = st.text_input(f"Importancia {start_idx + i + 1}", row["importance"], key=f"importancia_{idx}")

            # Columna 5: Botón para generar variaciones individualmente
            with col5:
                if st.button(f"Generar Variaciones {start_idx + i + 1}", key=f"generate_variations_{idx}"):
                    generate_variations_llm(idx, st.session_state[f"pregunta_{idx}"], st.session_state[f"respuesta_{idx}"], importancia)

            # Columna 6: Variaciones dinámicas (solo se actualiza esta parte)
            with col6:
                if idx in st.session_state["variations"]:
                    num_variations = len(st.session_state["variations"][idx])
                    for j in range(num_variations):
                        col_var, col_del = st.columns([4, 1])  # Crear dos columnas: una para el texto y otra para el botón
                        with col_var:
                            st.text_area(f"Variación {j + 1} Fila {start_idx + i + 1}", st.session_state["variations"][idx][j], height=50, key=f"variacion_{idx}_{j}")
                        with col_del:
                            if st.button(f"Eliminar", key=f"delete_variation_{idx}_{j}"):
                                delete_variation(idx, j)  # Llamar a la función para eliminar la variación
                else:
                    st.write("No hay variaciones generadas.")

        # Controles de paginación
        col_prev, col_next = st.columns([1, 1])

        with col_prev:
            if st.session_state.page > 0:
                if st.button("Página anterior"):
                    st.session_state.page -= 1

        with col_next:
            if st.session_state.page < total_pages - 1:
                if st.button("Página siguiente"):
                    st.session_state.page += 1

        # Botones globales para generar variaciones y guardar datos
        col_all_gen, col_all_save = st.columns([1, 1])

        with col_all_gen:
            if st.button("Generar todas las variaciones"):
                generate_all_variations()

        with col_all_save:
            if st.button("Guardar todo en el CSV"):
                save_all_data()
        st.button("Actualizar estado", key=f"update_states_1")

else:
    st.write("Esperando que se suba un archivo CSV...")
