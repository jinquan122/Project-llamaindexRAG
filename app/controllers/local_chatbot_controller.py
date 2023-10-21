from dataclasses import dataclass
import pinecone
from llama_index.vector_stores import PineconeVectorStore
from llama_index import VectorStoreIndex
from app.chatbot.local_chatbot import Llama2_7b
import configparser

config = configparser.ConfigParser()
config.read('config.ini')

@dataclass
class pineconeCredential:
    api: str
    env: str

class localchatbotController:
    def __init__(self):
        pineCredential = pineconeCredential(api= config.get('VectorDB', 'api_key'),env= config.get('VectorDB', 'env'))
        pinecone.init(api_key = pineCredential.api, environment = pineCredential.env)
        llama2_7b = Llama2_7b()

        self.vector_store = PineconeVectorStore(pinecone_index=pinecone.Index(config.get('VectorDB', 'index')))
        self.index = VectorStoreIndex.from_vector_store(vector_store=self.vector_store)
        self.agent = llama2_7b.llamaindexAgent(index=self.index)

    def chat(self, input):
        msg = input.get('input', None)
        def generate():
            response = self.agent.chat(msg)
            return str(response)
        return generate()

