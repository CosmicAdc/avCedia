
from .model import embeddings 
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_text_splitters import CharacterTextSplitter


#text_splitter=SemanticChunker(embeddings,breakpoint_threshold_type="standard_deviation")
# text_splitter = RecursiveCharacterTextSplitter(
#     chunk_size=1500,
#     chunk_overlap=100,
#     separators=[
#         "\n\n",
#         "\n",
#     ],
# )

text_splitter = CharacterTextSplitter(
    separator="\n\n",
    chunk_size=4000, # 1000 , 1500
    chunk_overlap=25,
    length_function=len,
    is_separator_regex=False,
)