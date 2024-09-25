
from langchain_experimental.text_splitter import SemanticChunker
from .model import embeddings 

text_splitter=SemanticChunker(embeddings, breakpoint_threshold_type="percentile")
