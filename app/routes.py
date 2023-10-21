from flask import Blueprint, request
from app.controllers.chatbot_controller import chatbotController
from app.controllers.vectorstore_controller import updateStore
from app.controllers.local_chatbot_controller import localchatbotController

chatAgent = chatbotController()
localChatAgent = localchatbotController()
updateVectorStore = updateStore()

api_blueprint = Blueprint("api", __name__)

@api_blueprint.route("/chatagent", methods=['POST'])
def chatAgentHandler():
  return chatAgent.chat(request.json)

@api_blueprint.route("/localchatagent", methods=['POST'])
def localchatAgentHandler():
  return localChatAgent.chat(request.json)

@api_blueprint.route("/updatestore", methods=['POST'])
def updateStoreHandler():
  return updateVectorStore.update(request.json)