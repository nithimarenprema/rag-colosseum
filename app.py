import json
import os
import re
import numpy as np

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

model = SentenceTransformer("all-MiniLM-L6-v2")

with open("kb.json", "r", encoding="utf-8") as f:
    data = json.load(f)

passages = []
for i, item in enumerate(data):
    if isinstance(item, str):
        passages.append({"id": i, "text": item})
    else:
        passages.append({
            "id": item.get("id", i),
            "text": item.get("text", item.get("content", item.get("passage", "")))
        })

passages = [p for p in passages if p["text"]]

texts = [p["text"] for p in passages]
embeddings = model.encode(
    texts, normalize_embeddings=True
)

class Query(BaseModel):
    question: str
    top_k: int = 3

@app.get("/")
def home():
    return {"status": "RAG service running"}

@app.post("/query")
def query(request: Query):
    question = request.question.strip()

    if not question:
        return {
            "answer": None,
            "sources": [],
            "scores": [],
            "confidence": 0.0
        }

    q_embedding = model.encode(
        [question], normalize_embeddings=True
    )[0]

    similarities = np.dot(embeddings, q_embedding)
    order = np.argsort(similarities)[::-1]

    k = max(1, min(request.top_k, 5))
    best = int(order[0])
    best_score = float(similarities[best])

    if best_score < 0.35:
        return {
            "answer": None,
            "sources": [],
            "scores": [],
            "confidence": 0.1
        }

    selected = [
        int(i) for i in order[:k]
        if similarities[i] >= max(0.35, best_score - 0.15)
    ]

    answer = "\n\n".join(
        passages[i]["text"] for i in selected
    )

    return {
        "answer": answer,
        "sources": [passages[i]["id"] for i in selected],
        "scores": [
            round(float(similarities[i]), 4) for i in selected
        ],
        "confidence": round(min(1.0, max(0.0, best_score)), 4)
    }
