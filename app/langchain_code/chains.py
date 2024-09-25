
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from .model import llm


def show(input):
    print(input)
    return input

template_generate_question_base=""" Utilizando el siguiente contexto, formula una pregunta que se pueda responder de manera clara 
y precisa con la información proporcionada, pero no menciones el contexto.  Asegúrate de que la pregunta esté alineada con los detalles del contexto y 
sea relevante para el tema tratado, en caso que el contexto no tenga información relevante para hacer una buena pregunta responde exactamente: "NO INFORMATION" se estricto con esto.
No hagas respondas con preguntas que mencionen el contexto proporcionado o frases similares
Contexto: {context}"""

prompt_question_base = ChatPromptTemplate.from_template(template_generate_question_base)

chain_question = (
    {"context": RunnablePassthrough()}
    | prompt_question_base
    | llm
    | StrOutputParser()
)

def invoke_call_chain(doc):
    return chain_question.invoke({"context": doc})


def batch_call_chain(docs):
    return chain_question.batch(docs)




template_generate_answer_base="""Utilizando el siguiente contexto, responde la pregunta proporcionada utilizando 
la información que se encuentra en el contexto. No agregues información basada en suposiciones , pero brinda y da toda la información posible acerca del tema .
Si el contexto no contiene suficiente información para responder la pregunta, responde exactamente: "NO INFORMATION". Sé estricto con esto.
No menciones en tu respuesta basado en el contexto proporcionado o frases similares en tu respuesta.
Contexto: {context}
Pregunta: {question}"""

prompt_answer_base = ChatPromptTemplate.from_template(template_generate_answer_base)

chain_response = (
    prompt_answer_base
    | llm
    | StrOutputParser()
)

def invoke_call_chain_answer(context,question):
    return chain_response.invoke({"context": context,"question":question})


def batch_call_chain_answer(questions):
    return chain_response.batch(questions)