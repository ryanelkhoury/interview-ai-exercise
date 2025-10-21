"""Document loader for the RAG example."""

import json
from typing import Any
import requests
import chromadb
import uuid
import tqdm

from langchain_text_splitters import RecursiveCharacterTextSplitter

from ai_exercise.constants import SETTINGS
from ai_exercise.loading.chunk_json import chunk_data
from ai_exercise.models import Document
from ai_exercise.llm.completions import get_completion
from ai_exercise.constants import SETTINGS, openai_client


def get_contextual_retrieval(specification: dict, chunk: str) -> str:

    title = specification.get("info", {}).get("title", "Unknown API")
    version = specification.get("info", {}).get("version", "N/A")
    description = specification.get("info", {}).get("description", "")

    servers = [s.get("url") for s in specification.get("servers", [])]
    tags = [t.get("name") for t in specification.get("tags", [])]
    schemas = list(specification.get("components", {}).get("schemas", {}).keys())
    paths = list(specification.get("paths", {}).keys())

    # Build a compact, human-readable summary of the spec
    full_doc = f"""
    API: {title} (v{version})
    Description: {description}
    Servers: {', '.join(servers)}
    Tags: {', '.join(tags[:10])}
    Schemas: {', '.join(schemas[:15])}...
    Paths: {', '.join(paths[:15])}...
    """

    pre_append = f"""
    <document>
    {full_doc}
    </document>
    Here is the chunk we want to situate within the whole document 
    <chunk>
    {chunk}
    </chunk>
    Please give a short succinct context to situate this chunk within the overall document for the purposes of 
    improving search retrieval of the chunk. Answer only with the succinct context and nothing else. 
    """

    try:
        return get_completion(
            client=openai_client,
            messages=[{'role': 'user', 'content': pre_append}],
            model='gpt-4o-mini',
        ).strip()
    except Exception as e:
        print(f"⚠️ Contextual Retrieval Failed: {e}")

    return ""

def chunk_openapi_spec(specification: dict) -> list[dict]:
    """
    Simple chunking: capture everything, no data loss.
    """
    chunks = []
    spec_title = specification.get("info", {}).get("title", "unknown")
    
    # Chunk each endpoint
    for path, path_item in specification.get("paths", {}).items():
        for method, operation in path_item.items():
            if method not in ["get", "post", "put", "patch", "delete", "options", "head", "trace"]:
                continue

            text = f"""
                Service: {spec_title}
                Version: {specification.get("info", {}).get("version", "")}
                Base URL: {specification.get("servers", [{}])[0].get("url", "")}
                Endpoint: {method.upper()} {path}

                {json.dumps(operation, indent=2, ensure_ascii=False)}
                """

            contextual_retrieval = get_contextual_retrieval(specification, text)  
            
            chunks.append({
                "source_spec": spec_title,
                "method": method.upper(),
                "path": path,
                "text_for_embedding": contextual_retrieval + text,
                "full_spec": specification,
            })
    
    # Add ONE chunk for components (schemas, security, etc.)
        # Chunk components - ONE CHUNK PER SCHEMA
    if "components" in specification:
        components = specification["components"]
        
        # Security schemes (usually small, can be one chunk)
        if "securitySchemes" in components:
            security_text = f"""
            Service: {spec_title}
            Type: Security Schemes

            {json.dumps(components["securitySchemes"], indent=2, ensure_ascii=False)}
            """

            contextual_retrieval = get_contextual_retrieval(specification, security_text)      
            
            chunks.append({
                "source_spec": spec_title,
                "method": "SECURITY",
                "path": "/components/securitySchemes",
                "text_for_embedding": contextual_retrieval + security_text
            })
        
        # Each schema gets its own chunk
        if "schemas" in components:
            for schema_name, schema_def in components["schemas"].items():
                schema_text = f"""
                Service: {spec_title}
                Type: Data Schema
                Schema Name: {schema_name}

                {json.dumps(schema_def, indent=2, ensure_ascii=False)}
                """

                contextual_retrieval = get_contextual_retrieval(specification, schema_text)      

                chunks.append({
                    "source_spec": spec_title,
                    "method": "SCHEMA",
                    "path": f"/components/schemas/{schema_name}",
                    "text_for_embedding": contextual_retrieval + schema_text
                })
    
    return chunks


# --- 🧠 Function to build clean embedding text
def build_embedding_text(chunk: dict) -> str:
    """
    Build clean, context-rich text for embedding from a chunk.
    This is what actually gets embedded in the vector database.
    """
    # Already built! Just return it
    return chunk["text_for_embedding"]


def get_json_data() -> list[dict[str, Any]]:
    """
    Fetch and return all OpenAPI JSON specs from StackOne's OAS endpoints.
    
    Returns:
        List of parsed OpenAPI specs (each as a dict).
    """
    urls = [
        "https://api.eu1.stackone.com/oas/stackone.json",
        "https://api.eu1.stackone.com/oas/hris.json",
        "https://api.eu1.stackone.com/oas/ats.json",
        "https://api.eu1.stackone.com/oas/lms.json",
        "https://api.eu1.stackone.com/oas/iam.json",
        "https://api.eu1.stackone.com/oas/crm.json",
        "https://api.eu1.stackone.com/oas/marketing.json",
    ]

    specs = []
    for url in urls:
        try:
            print(f"🌐 Fetching {url} ...")
            response = requests.get(url, timeout=20)
            response.raise_for_status()
            specs.append(response.json())
            print(f"✅ Loaded spec from {url}")
        except Exception as e:
            print(f"⚠️ Failed to load {url}: {e}")

    print(f"\n✅ Successfully loaded {len(specs)} OpenAPI specs total.")
    return specs


def document_json_array(data: list[dict[str, Any]], source: str) -> list[Document]:
    """Converts an array of JSON chunks into a list of Document objects."""
    return [
        Document(page_content=json.dumps(item), metadata={"source": source})
        for item in data
    ]


def build_docs(data: dict[str, Any]) -> list[Document]:
    """Chunk (badly) and convert the JSON data into a list of Document objects."""
    docs = []
    for attribute in ["paths", "webhooks", "components"]:
        chunks = chunk_data(data, attribute)
        docs.extend(document_json_array(chunks, attribute))
    return docs


def split_docs(docs_array: list[Document]) -> list[Document]:
    """Some may still be too long, so we split them."""
    splitter = RecursiveCharacterTextSplitter(
        separators=["}],", "},", "}", "]", " ", ""], chunk_size=SETTINGS.chunk_size
    )
    return splitter.split_documents(docs_array)


def add_documents(collection: chromadb.Collection, docs: list[Document], batch_size: int = 100) -> None:
    """
    Add documents to the collection in batches to avoid hitting token limits.

    Args:
        collection: ChromaDB collection to add documents to
        docs: List of documents to add
        batch_size: Number of documents to process in each batch (default: 100)
    """
    total_docs = len(docs)
    print(f"📦 Adding {total_docs} documents in batches of {batch_size}...")

    for i in range(0, total_docs, batch_size):
        batch = docs[i:i + batch_size]
        batch_num = (i // batch_size) + 1
        total_batches = (total_docs + batch_size - 1) // batch_size

        print(f"  Processing batch {batch_num}/{total_batches} ({len(batch)} documents)...")

        collection.add(
            documents=[doc.page_content for doc in batch],
            metadatas=[doc.metadata for doc in batch],
            ids=[str(uuid.uuid4()) for _ in batch],
        )

    print(f"✅ Successfully added all {total_docs} documents to collection.")
