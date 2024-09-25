import getpass
import os
from langchain_cohere import ChatCohere
from langchain_cohere import CohereEmbeddings
from langchain_ollama.llms import OllamaLLM
from langchain_ollama import OllamaEmbeddings

# os.environ["COHERE_API_KEY"] = "DbKjSsGbvVOESIv8wLZRlPW4rX0k8EQTkqy5xlbS"

# llm = ChatCohere(model="command-r-plus")
# embeddings = CohereEmbeddings(model="embed-english-light-v3.0")

llm = OllamaLLM(model="qwen2.5", temperature=0.4, num_predict=640)
embeddings = OllamaEmbeddings(model="nomic-embed-text")