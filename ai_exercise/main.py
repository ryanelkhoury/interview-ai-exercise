"""FastAPI app creation, main API routes."""

from fastapi import FastAPI

from ai_exercise.constants import SETTINGS, chroma_client, openai_client
from ai_exercise.llm.completions import get_completion
from ai_exercise.llm.embeddings import openai_ef
import tqdm
from ai_exercise.loading.document_loader import (
    add_documents,
    chunk_openapi_spec,
    get_json_data,
)
from ai_exercise.models import (
    ChatMessage,
    ChatOutput,
    ChatQuery,
    HealthRouteOutput,
    Document,
    LoadDocumentsOutput,
    # EvaluationOutput,  # Commented out until evaluator module is created
)
from ai_exercise.retrieval.retrieval import get_relevant_chunks
from ai_exercise.retrieval.vector_store import create_collection
# from ai_exercise.evaluation.evaluator import RetrievalEvaluator  # Module doesn't exist yet

app = FastAPI()

collection = create_collection(chroma_client, openai_ef, SETTINGS.collection_name)


@app.get("/health")
def health_check_route() -> HealthRouteOutput:
    """Health check route to check that the API is up."""
    return HealthRouteOutput(status="ok")

@app.get("/load")
async def load_docs_route() -> LoadDocumentsOutput:
    """
    Fetch all OpenAPI specs, chunk them, build embedding-ready docs,
    and load them into the vector store.
    """
    global collection  # Declare global at the start of the function

    print("🚀 Starting OpenAPI ingestion...")

    # 0️⃣ Clear existing collection data
    try:
        existing_count = collection.count()
        if existing_count > 0:
            print(f"🗑️  Deleting {existing_count} existing documents from collection...")
            # Delete the collection and recreate it
            chroma_client.delete_collection(name=SETTINGS.collection_name)
            # Recreate the collection with the same settings
            collection = create_collection(chroma_client, openai_ef, SETTINGS.collection_name)
            print("✅ Collection cleared and recreated")
        else:
            print("ℹ️  Collection is already empty")
    except Exception as e:
        print(f"⚠️  Error clearing collection: {e}")

    # 1️⃣ Fetch all OpenAPI specs
    specs = get_json_data()
    all_chunks = []

    # 2️⃣ Chunk each spec
    for spec in tqdm.tqdm(specs):
        chunks = chunk_openapi_spec(spec)
        all_chunks.extend(chunks)

    print(f"\n✅ Created {len(all_chunks)} chunks total.\n")

    # 3️⃣ Build embedding-ready text for each chunk
    documents = []
    for i, chunk in enumerate(all_chunks):
        # Build metadata from chunk fields
        doc_metadata = {
            "source_spec": chunk["source_spec"],
            "method": chunk["method"],
            "path": chunk["path"],
            "chunk_id": f"chunk_{i+1}"
        }

        # Create Document
        documents.append(
            Document(
                page_content=chunk["text_for_embedding"].strip(),
                metadata=doc_metadata
            )
        )

    print(f"✅ Ready {len(documents)} documents for embedding.\n")
    
    # 5️⃣ Optionally split long documents into smaller parts
    # documents = split_docs(documents)

    # for i, doc in enumerate(documents):
    #     text_length = len(doc.page_content)
    #     if text_length > 1000:
    #         print(f"📄 Document {i+1}: {text_length} characters")

    # 6️⃣ Add documents to vector store
    add_documents(collection, documents)

    print(f"🧠 Number of documents in collection: {collection.count()}")

    # 7️⃣ Return status
    return LoadDocumentsOutput(status="ok")


@app.post("/chat")
def chat_route(chat_query: ChatQuery) -> ChatOutput:
    """
    RAG chat route with query rewriting.
    Steps:
    1. Rewrite user query using LLM.
    2. Retrieve relevant docs from vector store.
    3. Build prompt and get final completion.
    """

    print(f"💬 Incoming user query: {chat_query.query}")
    messages = '\n\n'.join([f"{message.role}: {message.content}" for message in chat_query.chat_history])

    # --- 1️⃣ Rewrite the query for better retrieval
    rewrite_prompt = f"""
        You are a *rewrite engine*, not a conversational assistant.
        \n\nINPUT\n=====\nConversation History\n--------------------
        \n{messages}\n\n
        TASK\n====\n
        Rewrite the **Latest User Message** so that it can be understood entirely on its own.
        \n1. Replace every pronoun **and every vague noun phrase** with a fully explicit description drawn from the conversation.
        \n2. Do **not** add explanations, salutations, quotation marks, or mention the rewrite process.
        \n3. Output **only** the rewritten sentence or question—nothing else.

        Original question:
        {chat_query.query}

        Rewritten version (clear, specific, technical, matching API terminology):
    """

    rewritten_query = chat_query.query

    try:
        rewritten_query = get_completion(
            client=openai_client,
            messages=[{'role': 'user', 'content': rewrite_prompt}],
            model=SETTINGS.openai_model,
        ).strip()
    except Exception as e:
        print(f"⚠️ Query rewriting failed: {e}")
        # rewritten_query = chat_query.query

    # Get relevant chunks from the collection using both rewritten and original queries
    print(f"\n🔍 Searching with rewritten query: {rewritten_query}")
    hits_rewritten = get_relevant_chunks(collection=collection, query=rewritten_query, k=SETTINGS.k_neighbors)

    # print(f"🔍 Searching with original query: {chat_query.query}")
    # hits_original = get_relevant_chunks(collection=collection, query=chat_query.query, k=SETTINGS.k_neighbors)

    # Combine results from both queries
    all_hits = hits_rewritten

    # Filter by distance threshold and deduplicate
    seen = set()
    relevant_chunks = []
    filtered_count = 0
    context_texts = []

    for hit in all_hits:
        chunk_text = hit.get("document", "")
        distance = hit.get("distance", float('inf'))

        print(f"  Distance: {distance:.4f} | Text preview: {chunk_text[:100]}...")

        # Only include if below threshold and not already seen
        # Lower distance = more similar (0=identical, higher=less similar)
        if distance <= SETTINGS.distance_threshold and chunk_text not in seen:
            seen.add(chunk_text)
            context_texts.append(hit.get('document', ''))
            relevant_chunks.append(hit)
            # print(f"  ✓ Similarity: {similarity_pct:.1f}% | {method.upper()} {path}")
        elif distance > SETTINGS.distance_threshold:
            filtered_count += 1

    print(f"📚 Retrieved {len(relevant_chunks)} relevant chunks ({filtered_count} filtered out)")

    # Debug: Print the actual chunks retrieved
    for i, chunk in enumerate(relevant_chunks[:3]):  # Show first 3
        distance = chunk.get('distance', 'N/A')
        text_preview = chunk.get('document', '')[:150]
        print(f"  Chunk {i+1} (distance={distance:.4f}): {text_preview}...")

    context = "\n\n".join(context_texts)
    if not context:
        context = "No relevant documentation found."
        print("⚠️  WARNING: No relevant context found after filtering!")

    # --- 5️⃣ Build messages array for chat completion
    system_message = f"""You are a highly knowledgeable API assistant.
Use the following API documentation context to answer the question precisely.

If not specified, clarify which service the user is talking about:
Applications, Contacts, Companies, Users, Courses, StackOne
Answer clearly, with direct reference to the relevant API endpoints.

---
Context:
{context}
---

If you don't find the answer, say: "I couldn't find that in the documentation."
"""

    messages = [{"role": "system", "content": system_message}]

    # Add chat history to messages
    if chat_query.chat_history:
        for msg in chat_query.chat_history:
            messages.append({"role": msg.role, "content": msg.content})

    # Add current user query
    messages.append({"role": "user", "content": chat_query.query})

    print("🧠 Constructed messages array with", len(messages), "messages\n")

    # --- 6️⃣ Get LLM answer
    answer = get_completion(
        client=openai_client,
        messages=messages,
        model=SETTINGS.openai_model,
    )

    # --- 7️⃣ Update chat history (only user queries and assistant answers, no context)
    updated_history = list(chat_query.chat_history) if chat_query.chat_history else []
    updated_history.append(ChatMessage(role="user", content=chat_query.query))
    updated_history.append(ChatMessage(role="assistant", content=answer))

    print(updated_history)
    # --- 8️⃣ Return structured output
    return ChatOutput(
        message=answer,
        chat_history=updated_history,
        contexts=context_texts,  # Include contexts for evaluation
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=80, reload=True)
