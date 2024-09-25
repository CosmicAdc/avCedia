from langchain_unstructured import UnstructuredLoader
from .bases import text_splitter


def load_document(path:str):
    return UnstructuredLoader(file_path)

def split_document(doc):
    return text_splitter.create_documents([doc])


def load_and_split(path:str):
    loader=load_document(path)
    doc=loader.load()
    print(doc)
    return "hola"
    return split_document(doc)




