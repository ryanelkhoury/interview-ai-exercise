"""Streamlit app for RAG demo.

Start from project root with :
```bash
PYTHONPATH=. streamlit run demo/main.py
```
"""

import requests
import streamlit as st

from demo.ping import display_message_if_ping_fails

st.set_page_config(
    "RAG Example",
)

if "session" not in st.session_state:
    st.session_state.session = {}

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "How can I help you?"},
    ]


with st.sidebar:
    display_message_if_ping_fails()

    st.divider()

    st.subheader("Document Management")

    if st.button("Load Documents", type="primary", use_container_width=True):
        with st.spinner("Loading documents into vector store..."):
            try:
                response = requests.get("http://localhost/load")
                response.raise_for_status()
                result = response.json()
                if result.get("status") == "ok":
                    st.success("Documents loaded successfully!")
                else:
                    st.error("Failed to load documents")
            except Exception as e:
                st.error(f"Error loading documents: {e}")

    st.divider()

    st.subheader("Chat Management")

    if st.button("Clear Chat History", use_container_width=True):
        st.session_state.messages = [
            {"role": "assistant", "content": "How can I help you?"},
        ]
        st.rerun()

st.title("RAG Example 🤖")

if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "How can I help you?"},
    ]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input("Which path gives me the candidate list?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)
    msg = ""

    with st.spinner("Thinking..."):
        try:
            # Build chat history from session state (excluding the current prompt we just added)
            chat_history = [
                {"role": msg["role"], "content": msg["content"]}
                for msg in st.session_state.messages[:-1]  # Exclude initial assistant message and current user message
                if msg["role"] != "assistant" or msg["content"] != "How can I help you?"  # Skip initial greeting
            ]

            response = requests.post(
                "http://localhost/chat",
                json={
                    "query": prompt,
                    "chat_history": chat_history
                }
            )
            response.raise_for_status()
            result = response.json()
            msg = result["message"]

            # Update session state with the returned chat history
            if "chat_history" in result:
                # Reconstruct messages from the API's chat_history
                st.session_state.messages = [
                    {"role": "assistant", "content": "How can I help you?"},
                ]
                for history_msg in result["chat_history"]:
                    st.session_state.messages.append({
                        "role": history_msg["role"],
                        "content": history_msg["content"]
                    })
        except Exception as e:
            st.error(e)
            st.stop()

    st.empty()

    # Only add the assistant message if we haven't already updated from chat_history
    if "chat_history" not in result:
        st.session_state.messages.append({"role": "assistant", "content": msg})

    st.chat_message("assistant").write(msg)
