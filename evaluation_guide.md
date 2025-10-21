# Evaluation Guide

This project uses **Ragas** to evaluate the RAG system quality.

## Quick Start

### 1. Install and Load Documents

```bash
make install
make dev-api

# In another terminal
curl http://localhost/load
```

### 2. Run Evaluation

```bash
make evaluate
```

This runs test queries through your RAG system and evaluates the results.

## What Ragas Measures

Ragas evaluates four key aspects of your RAG system:

### 1. Context Precision (0.0 - 1.0)
**Question:** Are the retrieved documents actually relevant?

**Target:** > 0.7

**Low score means:** Too much noise in retrieval. Fix by:
- Tightening distance thresholds
- Improving chunking strategy
- Adding re-ranking

### 2. Context Recall (0.0 - 1.0)
**Question:** Did we retrieve all necessary information?

**Target:** > 0.8

**Low score means:** Missing important context. Fix by:
- Increasing k (number of retrieved documents)
- Using hybrid search (semantic + keyword)
- Improving chunking to keep related info together

### 3. Faithfulness (0.0 - 1.0)
**Question:** Is the answer faithful to the retrieved context (no hallucinations)?

**Target:** > 0.8

**Low score means:** LLM is hallucinating. Fix by:
- Improving prompts to emphasize using only provided context
- Lowering temperature
- Adding citation requirements

### 4. Answer Relevancy (0.0 - 1.0)
**Question:** Does the answer actually address the question?

**Target:** > 0.7

**Low score means:** Answers are off-topic. Fix by:
- Improving retrieval to find relevant docs
- Better prompt engineering
- Adding question analysis step

## Good Results Look Like

All metrics above **0.75**:

```
Context Precision:  0.85
Context Recall:     0.90
Faithfulness:       0.88
Answer Relevancy:   0.82
```

## Adding Test Cases

Edit `ai_exercise/evaluate_ragas.py`:

```python
test_data = {
    "question": [
        "How do I list all candidates?",
        "What authentication methods are supported?",
        # Add your questions here
    ],
    "ground_truth": [
        "Use GET /ats/candidates endpoint to list all candidates.",
        "The API supports API key authentication using the X-API-Key header.",
        # Add expected answers here
    ],
}
```

**Tips:**
- Use real user queries
- Include edge cases
- Cover different question types (how-to, what-is, troubleshooting)

## Troubleshooting

**"No module named 'ragas'"**
Run: `make install`

**"Collection not found"**
Load documents: `curl http://localhost/load`

**Slow evaluation**
Ragas makes LLM calls for each metric. Start with 3-5 test cases.

## Resources

- [Ragas Documentation](https://docs.ragas.io/)
- [Ragas Metrics](https://docs.ragas.io/en/latest/concepts/metrics/)
