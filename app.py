import json
import math
import os
import re
from collections import Counter

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field


# ============================================================
# 1. APPLICATION SETUP
# ============================================================

app = FastAPI(
    title="RAG Colosseum",
    description="Retrieval-Augmented Generation API",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# 2. LOAD KNOWLEDGE BASE
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
KB_PATH = os.path.join(BASE_DIR, "kb.json")


def load_knowledge_base():
    if not os.path.exists(KB_PATH):
        raise FileNotFoundError("kb.json was not found")

    with open(KB_PATH, "r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, list):
        raise ValueError("Knowledge base must be a JSON list")

    if len(data) != 50:
        raise ValueError(
            f"Expected exactly 50 passages, found {len(data)}"
        )

    passages = []

    for item in data:
        if not isinstance(item, str):
            raise ValueError("Every knowledge-base passage must be text")

        passages.append(item.strip())

    if any(not passage for passage in passages):
        raise ValueError("Knowledge base contains an empty passage")

    return passages


PASSAGES = load_knowledge_base()


# ============================================================
# 3. TEXT PROCESSING
# ============================================================

STOP_WORDS = {
    "a", "an", "the", "is", "are", "was", "were",
    "what", "which", "who", "when", "where", "why",
    "how", "does", "do", "did", "can", "could",
    "would", "should", "will", "to", "of", "in",
    "on", "at", "for", "from", "by", "with",
    "about", "into", "and", "or", "as", "be",
    "been", "being", "that", "this", "these",
    "those", "it", "its", "their", "there",
    "use", "used", "using", "tell", "me",
    "please", "explain", "describe"
}


def normalize(text):
    text = text.lower()
    text = text.replace("-", " ")
    text = re.sub(r"[^a-z0-9\s.]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def tokenize(text):
    normalized = normalize(text)

    words = re.findall(r"[a-z0-9]+", normalized)

    return [
        word for word in words
        if word not in STOP_WORDS
    ]


# ============================================================
# 4. TECHNICAL ENTITY DETECTION
# ============================================================

ENTITY_ALIASES = {
    "postgresql": ["postgresql", "postgres", "pgsql"],
    "mongodb": ["mongodb", "mongo"],
    "redis": ["redis"],
    "mysql": ["mysql"],
    "memcached": ["memcached", "memcache"],
    "cassandra": ["cassandra"],
    "raft": ["raft"],
    "etcd": ["etcd"],
    "cockroachdb": ["cockroachdb"],
    "amazon dynamo": ["amazon dynamo", "dynamo"],
    "kafka": ["kafka"],
    "zookeeper": ["zookeeper"],
    "kubernetes": ["kubernetes", "k8s"],
    "docker": ["docker"],
    "http3": ["http3", "http 3"],
    "quic": ["quic"],
    "tcp": ["tcp"],
    "udp": ["udp"],
    "tls": ["tls"],
    "oauth": ["oauth"],
    "jwt": ["jwt"],
    "sql": ["sql"],
    "python": ["python"],
    "java": ["java"],
    "machine learning": ["machine learning"],
    "neural network": ["neural network", "neural networks"],
    "vector clock": ["vector clock", "vector clocks"],
    "hybrid logical clock": [
        "hybrid logical clock",
        "hybrid logical clocks"
    ],
}


def find_entities(text):
    q = normalize(text)
    found = []

    for entity, aliases in ENTITY_ALIASES.items():
        for alias in aliases:
            pattern = r"\b" + re.escape(alias) + r"\b"

            if re.search(pattern, q):
                found.append(entity)
                break

    return found


# ============================================================
# 5. ATTRIBUTE DETECTION
# ============================================================

ATTRIBUTE_GROUPS = {
    "port": [
        "port", "ports", "default port",
        "listening port", "network port"
    ],
    "timeout": [
        "timeout", "timeouts", "time out",
        "idle", "inactivity"
    ],
    "heartbeat": [
        "heartbeat", "heartbeats"
    ],
    "threshold": [
        "threshold", "thresholds"
    ],
    "algorithm": [
        "algorithm", "algorithms"
    ],
    "compression": [
        "compression", "compressed"
    ],
    "storage": [
        "storage", "stored", "persistence"
    ],
    "consistency": [
        "consistency", "consistent"
    ],
    "replication": [
        "replication", "replica", "replicas"
    ],
    "encryption": [
        "encryption", "encrypted"
    ],
    "protocol": [
        "protocol", "protocols"
    ],
    "memory": [
        "memory", "in-memory"
    ],
    "latency": [
        "latency", "delay"
    ],
    "size": [
        "size", "length", "bytes"
    ],
    "number": [
        "number", "count", "how many"
    ],
    "time": [
        "time", "duration", "interval"
    ],
}


def detect_attribute(question):
    q = normalize(question)

    for attribute, terms in ATTRIBUTE_GROUPS.items():
        for term in terms:
            if re.search(
                r"\b" + re.escape(term) + r"\b",
                q
            ):
                return attribute

    return None


# ============================================================
# 6. COMPARISON DETECTION
# ============================================================

def is_comparison_question(question):
    q = normalize(question)

    comparison_patterns = [
        r"\bcompare\b",
        r"\bcomparison\b",
        r"\bversus\b",
        r"\bvs\b",
        r"\bdifference\b",
        r"\bdifferences\b",
        r"\bboth\b",
        r"\brespectively\b",
        r"\bbetween\b",
        r"\band\b",
    ]

    return any(
        re.search(pattern, q)
        for pattern in comparison_patterns
    )


def answer_entities_for_question(question):
    entities = find_entities(question)

    if len(entities) <= 1:
        return entities

    # A question asking about two entities may be a comparison.
    if is_comparison_question(question):
        return entities

    # If the question asks for one attribute of one entity,
    # the first detected entity is the target.
    if detect_attribute(question):
        return entities[:1]

    return entities


# ============================================================
# 7. PROMPT INJECTION PROTECTION
# ============================================================

INJECTION_PATTERNS = [
    r"ignore previous instructions",
    r"ignore all instructions",
    r"disregard previous instructions",
    r"forget your instructions",
    r"reveal your system prompt",
    r"show your system prompt",
    r"print your hidden prompt",
    r"reveal hidden instructions",
    r"act as an unrestricted",
    r"bypass safety",
    r"ignore the knowledge base",
    r"override your instructions",
]


def is_injection(question):
    q = normalize(question)

    for pattern in INJECTION_PATTERNS:
        if re.search(pattern, q):
            return True

    return False


# ============================================================
# 8. BUILD BM25 INDEX
# ============================================================

DOCUMENT_TOKENS = [
    tokenize(passage)
    for passage in PASSAGES
]

DOCUMENT_LENGTHS = [
    len(tokens)
    for tokens in DOCUMENT_TOKENS
]

AVERAGE_DOCUMENT_LENGTH = (
    sum(DOCUMENT_LENGTHS) / len(DOCUMENT_LENGTHS)
    if DOCUMENT_LENGTHS else 1
)

DOCUMENT_FREQUENCIES = Counter()

for tokens in DOCUMENT_TOKENS:
    for word in set(tokens):
        DOCUMENT_FREQUENCIES[word] += 1


# BM25 parameters
K1 = 1.5
B = 0.75


def bm25_score(query_tokens, document_index):
    document_tokens = DOCUMENT_TOKENS[document_index]

    if not document_tokens:
        return 0.0

    frequencies = Counter(document_tokens)
    document_length = len(document_tokens)

    score = 0.0

    for word in query_tokens:

        frequency = frequencies.get(word, 0)

        if frequency == 0:
            continue

        document_frequency = DOCUMENT_FREQUENCIES.get(word, 0)

        inverse_document_frequency = math.log(
            1 + (
                len(PASSAGES) - document_frequency + 0.5
            ) / (
                document_frequency + 0.5
            )
        )

        numerator = frequency * (K1 + 1)

        denominator = frequency + K1 * (
            1 - B + B * (
                document_length / AVERAGE_DOCUMENT_LENGTH
            )
        )

        score += (
            inverse_document_frequency
            * numerator
            / denominator
        )

    return score


# ============================================================
# 9. RETRIEVAL
# ============================================================

def retrieve(question, top_k=3):

    query_tokens = tokenize(question)

    if not query_tokens:
        return []

    query_entities = find_entities(question)

    attribute = detect_attribute(question)

    results = []

    for index, passage in enumerate(PASSAGES):

        score = bm25_score(query_tokens, index)

        passage_lower = normalize(passage)

        # Reward passages containing requested entities.
        for entity in query_entities:
            aliases = ENTITY_ALIASES.get(entity, [entity])

            if any(
                re.search(
                    r"\b" + re.escape(alias) + r"\b",
                    passage_lower
                )
                for alias in aliases
            ):
                score += 1.5

        # Reward passages containing relevant attribute terms.
        if attribute:
            attribute_terms = ATTRIBUTE_GROUPS.get(
                attribute, []
            )

            if any(
                term in passage_lower
                for term in attribute_terms
            ):
                score += 0.5

        if score > 0:
            results.append({
                "id": index,
                "text": passage,
                "score": score
            })

    results.sort(
        key=lambda item: item["score"],
        reverse=True
    )

    return results[:max(1, min(top_k, 10))]


# ============================================================
# 10. SENTENCE EXTRACTION
# ============================================================

def split_sentences(text):
    sentences = re.split(
        r"(?<=[.!?])\s+",
        text.strip()
    )

    return [
        sentence.strip()
        for sentence in sentences
        if sentence.strip()
    ]


def sentence_matches_entity(sentence, entity):
    sentence_normalized = normalize(sentence)

    aliases = ENTITY_ALIASES.get(entity, [entity])

    return any(
        re.search(
            r"\b" + re.escape(alias) + r"\b",
            sentence_normalized
        )
        for alias in aliases
    )


def sentence_attribute_match(sentence, attribute):
    if not attribute:
        return True

    sentence_normalized = normalize(sentence)

    terms = ATTRIBUTE_GROUPS.get(attribute, [])

    return any(
        term in sentence_normalized
        for term in terms
    )


# ============================================================
# 11. SELECT ANSWER SENTENCES
# ============================================================

def select_sentences(question, selected):

    target_entities = answer_entities_for_question(question)

    attribute = detect_attribute(question)

    query_tokens = set(tokenize(question))

    chosen = []

    # First try to find sentences about the requested entity.
    if target_entities:

        for entity in target_entities:

            best_sentence = None
            best_score = 0.0

            for item in selected:

                for sentence in split_sentences(item["text"]):

                    if not sentence_matches_entity(
                        sentence, entity
                    ):
                        continue

                    if not sentence_attribute_match(
                        sentence, attribute
                    ):
                        continue

                    sentence_tokens = set(tokenize(sentence))

                    overlap = len(
                        query_tokens.intersection(sentence_tokens)
                    )

                    score = (
                        overlap
                        + item["score"] * 0.15
                    )

                    if score > best_score:
                        best_score = score
                        best_sentence = sentence

            if best_sentence:
                chosen.append(best_sentence)

        return list(dict.fromkeys(chosen))

    # For questions without recognized entities,
    # choose relevant sentences using word overlap.
    for item in selected:

        for sentence in split_sentences(item["text"]):

            sentence_tokens = set(tokenize(sentence))

            overlap = len(
                query_tokens.intersection(sentence_tokens)
            )

            if overlap >= 2:
                chosen.append(sentence)

    return list(dict.fromkeys(chosen))


# ============================================================
# 12. BUILD GROUNDED ANSWER
# ============================================================

def build_answer(question, selected):

    if not selected:
        return None

    sentences = select_sentences(question, selected)

    if not sentences:
        return None

    answer = " ".join(sentences)

    return answer


# ============================================================
# 13. RELEVANCE AND CONFIDENCE
# ============================================================

def calculate_confidence(question, answer, selected):

    if not answer or not selected:
        return 0.0

    query_tokens = set(tokenize(question))
    answer_tokens = set(tokenize(answer))

    if not query_tokens or not answer_tokens:
        return 0.0

    overlap = len(
        query_tokens.intersection(answer_tokens)
    )

    coverage = overlap / len(query_tokens)

    entities = answer_entities_for_question(question)

    entity_coverage = 1.0

    if entities:
        matched = sum(
            1 for entity in entities
            if any(
                sentence_matches_entity(sentence, entity)
                for sentence in split_sentences(answer)
            )
        )

        entity_coverage = matched / len(entities)

    confidence = (
        0.45 * min(coverage, 1.0)
        + 0.35 * entity_coverage
        + 0.20
    )

    return round(
        max(0.0, min(confidence, 1.0)),
        4
    )


# ============================================================
# 14. ABSTENTION
# ============================================================

def should_abstain(question, selected, answer):

    if is_injection(question):
        return True

    if not selected or not answer:
        return True

    query_tokens = set(tokenize(question))
    answer_tokens = set(tokenize(answer))

    if not query_tokens or not answer_tokens:
        return True

    overlap = query_tokens.intersection(answer_tokens)

    # Reject answers with almost no relation to the question.
    if len(overlap) < 1:
        return True

    entities = answer_entities_for_question(question)

    # If an entity was requested, it must appear in the answer.
    for entity in entities:
        if not any(
            sentence_matches_entity(sentence, entity)
            for sentence in split_sentences(answer)
        ):
            return True

    # Reject weak retrieval matches.
    if selected[0]["score"] < 0.5:
        return True

    return False


# ============================================================
# 15. API REQUEST MODEL
# ============================================================

class QueryRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=1,
        max_length=2000
    )

    top_k: int = Field(
        default=3,
        ge=1,
        le=10
    )


# ============================================================
# 16. HEALTH ENDPOINTS
# ============================================================

@app.get("/")
def root():

    return {
        "status": "RAG Colosseum running",
        "passages": len(PASSAGES)
    }


@app.get("/health")
def health():

    return {
        "status": "healthy",
        "passages": len(PASSAGES)
    }


# ============================================================
# 17. MAIN QUERY ENDPOINT
# ============================================================

@app.post("/query")
def query_endpoint(request: QueryRequest):

    question = request.question.strip()

    # Reject empty questions.
    if not question:
        return {
            "answer": None,
            "sources": [],
            "scores": [],
            "confidence": 0.0
        }

    # Reject prompt injection attempts.
    if is_injection(question):
        return {
            "answer": None,
            "sources": [],
            "scores": [],
            "confidence": 0.0
        }

    # Retrieve relevant passages.
    selected = retrieve(
        question,
        request.top_k
    )

    # Construct answer from retrieved passages.
    answer = build_answer(
        question,
        selected
    )

    # Abstain when evidence is insufficient.
    if should_abstain(
        question,
        selected,
        answer
    ):
        return {
            "answer": None,
            "sources": [],
            "scores": [],
            "confidence": 0.0
        }

    # Calculate confidence.
    confidence = calculate_confidence(
        question,
        answer,
        selected
    )

    # Return source IDs and retrieval scores.
    sources = [
        item["id"]
        for item in selected
    ]

    scores = [
        round(float(item["score"]), 4)
        for item in selected
    ]

    return {
        "answer": answer,
        "sources": sources,
        "scores": scores,
        "confidence": confidence
    }


# ============================================================
# 18. STARTUP INFORMATION
# ============================================================

@app.on_event("startup")
def startup_event():

    print("===================================")
    print("RAG Colosseum started successfully")
    print("Knowledge base passages:", len(PASSAGES))
    print("Retrieval: BM25")
    print("API endpoint: /query")
    print("===================================")
