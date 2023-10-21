import pinecone
from app.controllers.chatbot_controller import pineconeCredential
from llama_index.vector_stores import PineconeVectorStore
from llama_index.storage.storage_context import StorageContext
from llama_index import Document
import pandas as pd
import openai
from llama_index.node_parser import SimpleNodeParser
from llama_index import GPTVectorStoreIndex, StorageContext, ServiceContext
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.text_splitter import get_default_text_splitter
import configparser

config = configparser.ConfigParser()
config.read('config.ini')

class updateStore:
    def __init__(self) -> None:
        pineCredential = pineconeCredential(api= config.get('VectorDB', 'api_key'),env= config.get('VectorDB', 'env'))
        pinecone.init(api_key = pineCredential.api, environment = pineCredential.env)
        self.openai_api_key = config.get('Agent', 'api_key')
        openai.api_key = self.openai_api_key
        self.docs = []

        df = pd.read_csv("./data/data.csv", encoding="ISO-8859-1") # The data updated is stored in the path mentioned
        for i, row in df.iterrows():
            self.docs.append(Document(
                text = row['bodytext'],
                doc_id = row['id'],
                extra_info = {'title': row['categories']}
            ))

        parser = SimpleNodeParser(text_splitter=get_default_text_splitter())

        self.nodes = parser.get_nodes_from_documents(self.docs)
                
    def update(self, data):
        '''
        To update latest data into pinecone vector database 

        Workflow:
        1. To check if the vector index is existing.
        2. If no, create a new index for for the vector database. If yes, use the current index as the vector database.
        3. Setup the vector store
        4. Use OpenAI embedding model 'text-embedding-ada-002' to vectorize the articles
        5. Succesfully update vector store.
        '''
        msg = data.get('input', None)
        index_name = config.get('VectorDB', 'index')
        if index_name not in pinecone.list_indexes():
            pinecone.create_index(
                index_name,
                dimension = 1536,
                metric = 'cosine'
            )
        pinecone_index = pinecone.Index(index_name)
        
        vector_store = PineconeVectorStore(
            pinecone_index = pinecone_index,
            add_sparse_vector = True,
        )
        storage_context = StorageContext.from_defaults(vector_store = vector_store)

        embed_model = OpenAIEmbedding(model = 'text-embedding-ada-002', embed_batch_size = 100)
        service_context = ServiceContext.from_defaults(embed_model = embed_model)
        index = GPTVectorStoreIndex.from_documents(
            self.docs, 
            storage_context = storage_context,
            service_context = service_context
        )

        return("Succesfully update vector store!")
