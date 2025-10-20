import chainlit as cl
from dotenv import load_dotenv
import os

from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import Qdrant
from qdrant_client import QdrantClient
from langchain_openai import ChatOpenAI

load_dotenv()
URL = os.getenv("URL")
QDRANT_HOST = os.getenv("QDRANT_HOST")
QDRANT_PORT = os.getenv("QDRANT_PORT")
QDRANT_COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MODEL = os.getenv("MODEL")
CHAT_MODEL = os.getenv("CHAT_MODEL")
if not URL or not QDRANT_HOST or not QDRANT_PORT or not QDRANT_COLLECTION_NAME or not OPENAI_API_KEY or not MODEL or not CHAT_MODEL:
    raise Exception("Environment variables are not set")

vector_db_client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
embeddings = OpenAIEmbeddings(model=MODEL, api_key=OPENAI_API_KEY)
vector_db = Qdrant(client=vector_db_client, collection_name=QDRANT_COLLECTION_NAME, embeddings=embeddings)
llm = ChatOpenAI(model=CHAT_MODEL, openai_api_key=OPENAI_API_KEY)
history = []

@cl.on_chat_start
async def start():
    await cl.Message(
        content="Szia! Én az Eötvös József Gimnázium weboldalának chatbotja vagyok. Miben segíthetek?",
    ).send()
    history.append({"role": "assistant", "content": "Szia! Én az Eötvös József Gimnázium weboldalának chatbotja vagyok. Miben segíthetek?"})

@cl.on_message
async def main(message: cl.Message):
    question = message.content
    history.append({"role": "user", "content": question})
    query_results = vector_db.similarity_search(question)
    for result in query_results:
        print(result.page_content)
        history.append({"role": "assistant", "content": result.page_content})
    response = llm.invoke(history)
    history.append({"role": "assistant", "content": response.content})
    await cl.Message(content=response.content).send()