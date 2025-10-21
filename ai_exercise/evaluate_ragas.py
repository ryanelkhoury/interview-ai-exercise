"""Simple RAG evaluation using Ragas framework.

This script demonstrates how to evaluate the RAG system using Ragas,
a framework specifically designed for evaluating RAG applications.
"""

import requests
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness,
)


def evaluate_rag_system() -> None:
    """Run Ragas evaluation on the RAG system."""
    # Define test queries with ground truth
    # These should be questions where you know the expected answer
    test_data = {
        "question": [
            # =====================
            # LMS (Learning Management System)
            # =====================
            "How can I list all LMS courses?",
            "How do I get a specific course by ID?",
            "How can I filter courses updated after a specific date?",
            "How do I list all assignments for a user?",
            "How do I create a new assignment for a user?",
            "How do I filter user assignments by status?",
            "What happens if I provide an invalid LMS account ID?",
            "How do I paginate through LMS courses?",

            # =====================
            # IAM (Identity & Access Management)
            # =====================
            "How can I list all IAM users?",
            "How do I get details of a specific IAM user?",
            "How do I update an IAM user?",
            "How can I filter IAM users updated after a certain date?",
            "How do I expand related roles or groups when listing IAM users?",
            "What happens if I try to get a user that does not exist?",
            "How do I handle too many IAM requests?",

            # =====================
            # CRM (Customer Relationship Management)
            # =====================
            "How do I list all CRM contacts?",
            "How do I create a new CRM contact?",
            "How do I retrieve a specific CRM contact?",
            "How do I update a CRM contact?",
            "How can I list all CRM accounts?",
            "How do I filter CRM contacts updated after a given date?",
            "How do I include custom fields when retrieving CRM contacts?",
            "What happens if a CRM request times out?",
            "How do I handle CRM rate limiting?",

            # =====================
            # Marketing (Email Templates)
            # =====================
            "How can I list all email templates?",
            "How do I create a new email template?",
            "How do I get a specific email template by ID?",
            "How can I update an existing email template?",
            "How can I filter email templates updated after a certain date?",
            "What happens if I provide an invalid template ID?",
            "How do I paginate through email templates?",
            "What should I do if my marketing API call times out?"
        ],
        "ground_truth": [
            # =====================
            # LMS
            # =====================
            "Use GET /unified/lms/courses to list all LMS courses.",
            "Use GET /unified/lms/courses/{id} to retrieve a specific course by ID.",
            "Use GET /unified/lms/courses with the 'filter[updated_after]' query parameter to filter courses updated after a specific date.",
            "Use GET /unified/lms/users/{id}/assignments to list all assignments for a user.",
            "Use POST /unified/lms/users/{id}/assignments to create a new assignment for a user.",
            "Use GET /unified/lms/users/{id}/assignments with 'filter[status]' to filter assignments by status.",
            "A 401 Unauthorized or 403 Forbidden error will be returned if the LMS account ID is invalid.",
            "Use the 'next' query parameter in GET /unified/lms/courses to paginate through results.",

            # =====================
            # IAM
            # =====================
            "Use GET /unified/iam/users to list all IAM users.",
            "Use GET /unified/iam/users/{id} to get details of a specific IAM user.",
            "Use PATCH /unified/iam/users/{id} to update an IAM user.",
            "Use GET /unified/iam/users with 'filter[updated_after]' to filter IAM users updated after a certain date.",
            "Use the 'expand' query parameter (e.g., expand=roles,groups) in GET /unified/iam/users to expand related entities.",
            "A 404 Not Found response will be returned if the requested user does not exist.",
            "If too many IAM requests are sent, a 429 Too Many Requests response is returned.",

            # =====================
            # CRM
            # =====================
            "Use GET /unified/crm/contacts to list all CRM contacts.",
            "Use POST /unified/crm/contacts to create a new CRM contact.",
            "Use GET /unified/crm/contacts/{id} to retrieve a specific CRM contact.",
            "Use PATCH /unified/crm/contacts/{id} to update a CRM contact.",
            "Use GET /unified/crm/accounts to list all CRM accounts.",
            "Use GET /unified/crm/contacts with 'filter[updated_after]' to filter contacts updated after a given date.",
            "Use the 'include' query parameter (e.g., include=custom_fields) in GET /unified/crm/contacts to include custom fields.",
            "A 408 Request Timeout response indicates that the CRM request took too long to process.",
            "If the request rate is too high, the API will return a 429 Too Many Requests response.",

            # =====================
            # Marketing
            # =====================
            "Use GET /unified/marketing/templates/email to list all email templates.",
            "Use POST /unified/marketing/templates/email to create a new email template.",
            "Use GET /unified/marketing/templates/email/{id} to retrieve a specific email template by ID.",
            "Use PATCH /unified/marketing/templates/email/{id} to update an existing email template.",
            "Use GET /unified/marketing/templates/email with 'filter[updated_after]' to filter email templates updated after a certain date.",
            "A 404 Not Found response will be returned if an invalid template ID is provided.",
            "Use the 'next' query parameter in GET /unified/marketing/templates/email to paginate through results.",
            "A 408 Request Timeout response indicates that the marketing API call took too long to process."
        ]
    }


    contexts_list = []
    answers_list = []

    # Base URL for the API - requires the FastAPI server to be running
    api_base_url = "http://localhost/chat"

    print(f"\n{'='*80}")
    print(f"Fetching responses from API for {len(test_data['question'])} questions...")
    print(f"{'='*80}\n")

    for i, question in enumerate(test_data["question"], 1):
        print(f"[{i}/{len(test_data['question'])}] Processing: {question[:60]}...")

        # Call the actual running API server via HTTP
        response = requests.post(
            api_base_url,
            json={"query": question, "chat_history": []},
            timeout=30
        )
        response.raise_for_status()
        result = response.json()

        # Extract answer and contexts from the response
        answers_list.append(result["message"])
        contexts_list.append(result.get("contexts", []))

    # Add to test data
    test_data["contexts"] = contexts_list
    test_data["answer"] = answers_list

    # Create Ragas dataset
    dataset = Dataset.from_dict(test_data)

    # Evaluate using Ragas metrics
    print("\n" + "=" * 80)
    print("Running Ragas Evaluation...")
    print("=" * 80 + "\n")

    result = evaluate(
        dataset,
        metrics=[
            context_precision,  # Are retrieved contexts relevant?
            context_recall,  # Did we retrieve all necessary context?
            faithfulness,  # Is the answer faithful to the context?
            answer_relevancy,  # Is the answer relevant to the question?
        ],
    )

    # Print the full result dictionary
    print("Full Results:")
    print(result)


if __name__ == "__main__":
    evaluate_rag_system()
