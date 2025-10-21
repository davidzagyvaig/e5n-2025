from bs4 import BeautifulSoup
import chainlit as cl
from dotenv import load_dotenv
import os

from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import Qdrant
from qdrant_client import QdrantClient, models
from langchain_openai import ChatOpenAI
from fastembed.sparse.bm25 import Bm25
from fastembed.late_interaction import LateInteractionTextEmbedding
import requests

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

client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
embeddings = OpenAIEmbeddings(model=MODEL, api_key=OPENAI_API_KEY)
vector_db = Qdrant(client=client, collection_name=QDRANT_COLLECTION_NAME, embeddings=embeddings)
llm = ChatOpenAI(model=CHAT_MODEL, openai_api_key=OPENAI_API_KEY)

dense_embedding_model = OpenAIEmbeddings(model=MODEL, api_key=OPENAI_API_KEY)
bm25_embedding_model = Bm25("Qdrant/bm25")
late_interaction_embedding_model = LateInteractionTextEmbedding("colbert-ir/colbertv2.0")

SYSTEM_PROMPT = {
    "role": "system",
    "content": "You are a helpful assistant that can answer questions about the Eötvös József Gimnázium weboldalának tartalma."
}
history: list[dict[str, str]] = []

@cl.on_chat_start
async def start():
    await cl.Message(
        content="Szia! Én az Eötvös József Gimnázium weboldalának chatbotja vagyok. Miben segíthetek?",
    ).send()
    history.clear()
    history.extend([SYSTEM_PROMPT, {"role": "assistant", "content": "Szia! Én az Eötvös József Gimnázium weboldalának chatbotja vagyok. Miben segíthetek?"}])

@cl.on_message
async def main(message: cl.Message):
    question = message.content
    history.append({"role": "user", "content": question})
    query_results = retrieve_text(question)
    for result in query_results:
        history.append({"role": "assistant", "content": result.payload["text"]})
        page_content = BeautifulSoup(requests.get(result.payload["url"], verify=False).text, "html.parser").get_text(separator="\n", strip=True)
        history.append({"role": "assistant", "content": page_content})
    await cl.Message(content="Fetched results from:\n" + "\n".join([result.payload["url"] for result in query_results])).send()
    response = llm.invoke(history)
    history.append({"role": "assistant", "content": response.content})
    await cl.Message(content=response.content).send()

def retrieve_text(question: str):
    dense_query = dense_embedding_model.embed_query(question)
    sparse_query = next(bm25_embedding_model.query_embed(question))
    late_query = next(late_interaction_embedding_model.query_embed(question))
    results = client.query_points(
        collection_name=QDRANT_COLLECTION_NAME,
        prefetch=[
            models.Prefetch(
                prefetch=[
                    models.Prefetch(
                        query=models.SparseVector(**sparse_query.as_object()),
                        using="bm25",
                        limit=100,
                    ),
                ],
                query=dense_query,
                using="text-dense",
                limit=25,
            ),
        ],
        query=late_query,
        using="colbertv2.0",
        with_payload=True,
        limit=5,
    )
    return results.points