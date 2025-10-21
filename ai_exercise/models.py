"""Types for the API."""

from dataclasses import dataclass

from pydantic import BaseModel


@dataclass
class Document:
    """A document to be added to the vector store."""

    page_content: str
    metadata: dict = None


class HealthRouteOutput(BaseModel):
    """Model for the health route output."""

    status: str


class LoadDocumentsOutput(BaseModel):
    """Model for the load documents route output."""

    status: str


class ChatMessage(BaseModel):
    """Model for a single chat message."""

    role: str  # "user" or "assistant"
    content: str


class ChatQuery(BaseModel):
    """Model for the chat input."""

    query: str
    chat_history: list[ChatMessage] | None = None


class ChatOutput(BaseModel):
    """Model for the chat route output."""

    message: str
    chat_history: list[ChatMessage]
    contexts: list[str] | None = None  # Optional field for evaluation purposes


class EvaluationMetrics(BaseModel):
    """Model for evaluation metrics."""

    avg_mrr: float
    avg_precision_1: float = 0.0
    avg_precision_3: float = 0.0
    avg_precision_5: float = 0.0
    avg_precision_10: float = 0.0
    avg_recall_1: float = 0.0
    avg_recall_3: float = 0.0
    avg_recall_5: float = 0.0
    avg_recall_10: float = 0.0
    avg_f1_1: float = 0.0
    avg_f1_3: float = 0.0
    avg_f1_5: float = 0.0
    avg_f1_10: float = 0.0
    avg_hit_rate_1: float = 0.0
    avg_hit_rate_3: float = 0.0
    avg_hit_rate_5: float = 0.0
    avg_hit_rate_10: float = 0.0
    avg_ndcg_1: float = 0.0
    avg_ndcg_3: float = 0.0
    avg_ndcg_5: float = 0.0
    avg_ndcg_10: float = 0.0
    total_queries: int
    total_relevant: int
    total_retrieved: int

    class Config:
        """Allow field names with @ symbols via aliases."""

        populate_by_name = True


class EvaluationOutput(BaseModel):
    """Model for the evaluation route output."""

    status: str
    summary: dict
    by_category: dict
    config: dict
    failed_queries: list[dict]
