# import getpass
# import os
# from langchain_cohere import ChatCohere
# from langchain_cohere import CohereEmbeddings
from langchain_ollama.llms import OllamaLLM
from langchain_ollama import OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI


# os.environ["COHERE_API_KEY"] = "DbKjSsGbvVOESIv8wLZRlPW4rX0k8EQTkqy5xlbS"

# llm = ChatCohere(model="command-r-plus")
# embeddings = CohereEmbeddings(model="embed-english-light-v3.0")


#llm = ChatOpenAI(model="gpt-4o-mini",max_tokens=1024,temperature=0.2)


#llm = ChatGroq(temperature=0.15, groq_api_key="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", model_name="llama3-groq-70b-8192-tool-use-preview")

#llm = OllamaLLM(model="qwen2.5", temperature=0.25, num_predict=1024)
#llm = OllamaLLM(model="llama3.1", temperature=0.3, num_predict=2048)

embeddings = OllamaEmbeddings(model="nomic-embed-text")