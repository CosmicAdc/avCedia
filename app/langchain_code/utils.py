from langchain_unstructured import UnstructuredLoader
from .bases import text_splitter


def load_document(path:str):
    return UnstructuredLoader(path)

def split_document(doc):
    return text_splitter.create_documents([doc])





