from dataclasses import dataclass
import pinecone
from llama_index.vector_stores import PineconeVectorStore
from llama_index import VectorStoreIndex
from app.chatbot.openai_chatbot import openai_gpt
from flask import Response
import openai
import configparser

config = configparser.ConfigParser()
config.read('config.ini')

@dataclass
class pineconeCredential:
    api: str
    env: str

class chatbotController:
    def __init__(self):
        pineCredential = pineconeCredential(api= config.get('VectorDB', 'api_key'),env= config.get('VectorDB', 'env'))
        self.openai_api_key = config.get('Agent', 'api_key')

        openai.api_key = self.openai_api_key
        pinecone.init(api_key = pineCredential.api, environment = pineCredential.env)
        gpt = openai_gpt()

        self.vector_store = PineconeVectorStore(pinecone_index=pinecone.Index(config.get('VectorDB', 'index')))
        self.index = VectorStoreIndex.from_vector_store(vector_store=self.vector_store)
        self.agent = gpt.llamaindexAgent(index=self.index)

    def chat(self, input):
        msg = input.get('input', None)
        def generate():
            response = self.agent.chat(msg)
            return str(response)
        return generate()

