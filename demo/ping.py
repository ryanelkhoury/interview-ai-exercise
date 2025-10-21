import requests
import streamlit as st


def display_message_if_ping_fails() -> None:
    try:
        ping_backend()
    except requests.RequestException:
        st.error(
            "The demo couldn't access the backend. Did you start it? "
            "Run `make start-api"
        )


def ping_backend() -> None:
    """Ping backend to check that the endpoint is working."""
    response = requests.get("http://localhost/health")
    response.raise_for_status()
