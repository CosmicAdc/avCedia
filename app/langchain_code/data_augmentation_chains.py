
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import PromptTemplate
from langchain.output_parsers import CommaSeparatedListOutputParser
from .model import llm

output_parser = CommaSeparatedListOutputParser()
format_instructions = "La respuesta debe ser una lista de valores separados por comas , ejemplo: `¿pregunta variada 1? , ¿pregunta variada 2? , ¿pregunta variada 3? , ¿pregunta variada 4?,¿pregunta variada 5? `" 

def show(input):
    print(input)
    return input


template_variation_generate_question=""" A continuación tienes una pregunta y una respuesta basada en un contexto específico de un documento.
Tu tarea es generar exactamente {n} versiones alternativas de esta pregunta manteniendo su significado y que se puedan responder con la misma respuesta, 
pero utilizando diferentes estructuras gramaticales o palabras , pero la pregunta nunca debe ser vaga, sino debe especificar el tema del que se trata, las preguntas no deben tener comas aunque sea necesario, ni siquiera las necesarias , a excepción de las separaciones de las preguntas:
Pregunta: {question}
Respuesta: {response}
Genera exactamente {n} variaciones de la pregunta original que conserven el mismo significado:
\n{format_instructions}
"""

prompt_variation_question = ChatPromptTemplate.from_template(template_variation_generate_question)


prompt_variation_question = PromptTemplate(
    template=template_variation_generate_question,
    input_variables=["question","response","n"],
    partial_variables={"format_instructions": format_instructions},
)


chain_question_variation = (
    prompt_variation_question
    | llm
    | output_parser
)

def invoke_call_chain(question,response,n):
    return chain_question_variation.invoke({"question": question,"response":response,"n":n})


def batch_call_chain(input):
    return chain_question_variation.batch(input)


# q="¿Cuál fue el problema principal que se presentó al intentar mejorar el rendimiento de los sistemas de un solo procesador a principios de la década de 2000?"
# a="El problema principal que se presentó al intentar mejorar el rendimiento de los sistemas de un solo procesador a principios de la década de 2000 fue el aumento exponencial del calor disipado por los propios procesadores debido al incremento en el número de transistores."
# n=3
# print(invoke_call_chain (q,a,n))
