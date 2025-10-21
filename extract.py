import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
import os
import argparse
import uuid
import urllib3
urllib3.disable_warnings()

from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient, models
from langchain_openai import OpenAIEmbeddings
from fastembed.sparse.bm25 import Bm25
from fastembed.late_interaction import LateInteractionTextEmbedding

load_dotenv()
URL = os.getenv("URL")
QDRANT_HOST = os.getenv("QDRANT_HOST")
QDRANT_PORT = os.getenv("QDRANT_PORT")
QDRANT_COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MODEL = os.getenv("MODEL")
if not URL or not QDRANT_HOST or not QDRANT_PORT or not QDRANT_COLLECTION_NAME or not OPENAI_API_KEY or not MODEL:
    raise Exception("Environment variables are not set")

dense_embedding_model = OpenAIEmbeddings(model=MODEL, api_key=OPENAI_API_KEY)
bm25_embedding_model = Bm25("Qdrant/bm25")
late_interaction_embedding_model = LateInteractionTextEmbedding("colbert-ir/colbertv2.0")

client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
client.recreate_collection(
    collection_name=QDRANT_COLLECTION_NAME,
    vectors_config={
        "text-dense": models.VectorParams(
            size=1536,  # openai vector size
            distance=models.Distance.COSINE,
        ),
        "colbertv2.0": models.VectorParams(
            size=128,
            distance=models.Distance.COSINE,
            multivector_config=models.MultiVectorConfig(
                comparator=models.MultiVectorComparator.MAX_SIM,
            )
        )
    },
    sparse_vectors_config={
        "bm25": models.SparseVectorParams(
            modifier=models.Modifier.IDF,
        )
    },
)

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, required=False, default=0)
    parser.add_argument("--search", type=str, required=False, default="basic")
    args = parser.parse_args()
    if not os.path.exists("links.txt"):
        print("Links file not found, starting search")
        if args.search == "bfs":
            bfs(URL)
        elif args.search == "basic":
            basic_search(URL)
        else:
            raise ValueError(f"Invalid search mode: {args.search}")
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

def basic_search(url: str) -> None:
    links = set()
    response = requests.get(url, verify=False)
    soup = BeautifulSoup(response.text, "html.parser")
    for link in soup.find_all("a"):
        if link.get("href").startswith("https://www.ejg") and "wp-content" not in link.get("href"):
            links.add(link.get("href"))
    print(f"Found {len(links)} links")
    with open("links.txt", "w") as f:
        for link in links:
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
            for chunk in chunks:
                upload_points(chunk, link.strip())
        except Exception as e:
            print(f"Failed to process link {link.strip()}: {e}")
    print("Text embedded")

def extract_text(url: str) -> str | None:
    response = requests.get(url, verify=False)
    if response.status_code != 200:
        return None
    soup = BeautifulSoup(response.text, "html.parser")
    text = soup.find("td", id="main-table-mid").get_text(separator="\n", strip=True).split("Naptár 2025-2026")[0].strip()
    if len(text) < 300:
        return None
    return text

def chunk_text(text: str) -> list[str]:
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_text(text)
    return chunks

def upload_points(text: str, url: str) -> None:
    id = str(uuid.uuid4())
    dense_embedding = dense_embedding_model.embed_query(text)
    bm25_embedding = list(bm25_embedding_model.embed(text))
    late_interaction_embedding = list(late_interaction_embedding_model.embed(text))
    client.upload_points(
        collection_name=QDRANT_COLLECTION_NAME,
        points=[
            models.PointStruct(
                id=id,
                vector={
                    "text-dense": dense_embedding,
                    "bm25": bm25_embedding[0].as_object(),
                    "colbertv2.0": late_interaction_embedding[0].tolist(),
                },
                payload={
                    "url": url,
                    "text": text,
                }
            )
        ]
    )

if __name__ == "__main__":
    main()