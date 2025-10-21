"""Retrieve relevant chunks from a vector store."""

import chromadb


def get_relevant_chunks(
    collection: chromadb.Collection, query: str, k: int
) -> list[dict]:
    """
    Retrieve k most relevant chunks for the query with their similarity scores.

    Returns:
        List of dicts containing document, metadata, and distance score for each result
    """
    results = collection.query(query_texts=[query], n_results=k)

    # Combine documents, metadatas, and distances into structured results
    chunks = []
    for i in range(len(results["documents"][0])):
        chunks.append({
            "document": results["documents"][0][i],
            "metadata": results["metadatas"][0][i] if results.get("metadatas") else {},
            "distance": results["distances"][0][i] if results.get("distances") else None,
            "id": results["ids"][0][i] if results.get("ids") else None,
        })

    return chunks
