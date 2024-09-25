
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from .model import llm

template_generate_question_base=""" Utilizando el siguiente contexto, formula una pregunta que se pueda responder de manera clara 
y precisa con la información proporcionada. Asegúrate de que la pregunta esté alineada con los detalles del contexto y 
sea relevante para el tema tratado, en caso que el contexto no tenga información relevante para hacer una buena pregunta responde exactamente: "NO INFORMATION" se estricto con esto.
Contexto: {context}"""

prompt_question_base = ChatPromptTemplate.from_template(template_generate_question_base)

chain = (
    {"context": RunnablePassthrough()}
    | prompt_question_base
    | llm
    | StrOutputParser()
)

def invoke_call_chain(doc):
    return chain.invoke({"context": doc})


def batch_call_chain(docs):
    return chain.batch(docs)


