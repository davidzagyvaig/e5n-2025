import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
import os
import argparse
import urllib3
urllib3.disable_warnings()

from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient, models
from langchain_qdrant import Qdrant
from langchain_openai import OpenAIEmbeddings

load_dotenv()
URL = os.getenv("URL")
QDRANT_HOST = os.getenv("QDRANT_HOST")
QDRANT_PORT = os.getenv("QDRANT_PORT")
QDRANT_COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MODEL = os.getenv("MODEL")
if not URL or not QDRANT_HOST or not QDRANT_PORT or not QDRANT_COLLECTION_NAME or not OPENAI_API_KEY or not MODEL:
    raise Exception("Environment variables are not set")

vector_db_client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
try:
    vector_db_client.delete_collection(collection_name=QDRANT_COLLECTION_NAME)
    print(f"Collection '{QDRANT_COLLECTION_NAME}' deleted.")
    vector_db_client.recreate_collection(
        collection_name=QDRANT_COLLECTION_NAME,
        vectors_config=models.VectorParams(size=1536, distance=models.Distance.COSINE),
    )
    print("Collection created successfully.")
except Exception:
    print(f"Collection '{QDRANT_COLLECTION_NAME}' does not exist, creating it.")
    vector_db_client.recreate_collection(
        collection_name=QDRANT_COLLECTION_NAME,
        vectors_config=models.VectorParams(size=1536, distance=models.Distance.COSINE),
    )
    print("Collection created successfully.")
embeddings = OpenAIEmbeddings(model=MODEL, api_key=OPENAI_API_KEY)
vector_db = Qdrant(client=vector_db_client, collection_name=QDRANT_COLLECTION_NAME, embeddings=embeddings)

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, required=False, default=0)
    args = parser.parse_args()
    if not os.path.exists("links.txt"):
        print("Links file not found, starting BFS")
        bfs(URL)
    print("Links obtained")
    embed_text(args.start)
    

def bfs(url: str) -> None:
    visited = set()
    queue = [url]
    while queue:
        current_url = queue.pop()
        if current_url in visited:
            continue
        visited.add(current_url)
        print(f"Visiting {current_url}")
        try:
            response = requests.get(current_url, verify=False)
            soup = BeautifulSoup(response.text, "html.parser")
            for link in soup.find_all("a"):
                if link.get("href") not in visited and link.get("href").startswith("https://www.ejg") and "wp-content" not in link.get("href"):
                    queue.insert(0, link.get("href"))
        except:
            print(f"Failed to visit {current_url}")
    print(f"Visited {len(visited)} pages")
    with open("links.txt", "w") as f:
        for link in visited:
            f.write(link + "\n")

def embed_text(start: int) -> None:
    with open("links.txt", "r") as f:
        links = f.readlines()
    if not links:
        raise Exception("Links file is empty")
    for i, link in enumerate(links[start:]):
        try:
            print(f"Processing link {i+1}/{len(links[start:])}")
            text = extract_text(link)
            if text is None:
                print(f"Failed to extract text from {link.strip()}")
                continue
            chunks = chunk_text(text)
            vector_db.add_texts(texts=chunks, metadatas=[{"url": link.strip()}] * len(chunks))
        except Exception as e:
            print(f"Failed to process link")
    print("Text embedded")

def extract_text(url: str) -> str | None:
    response = requests.get(url, verify=False)
    if response.status_code != 200:
        return None
    soup = BeautifulSoup(response.text, "html.parser")
    text = soup.find("td", id="main-table-mid").get_text(separator="\n", strip=True).split("Naptár 2025-2026")[0].strip()
    return text

def chunk_text(text: str) -> list[str]:
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_text(text)
    return chunks

if __name__ == "__main__":
    main()