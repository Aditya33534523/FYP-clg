import os
import re
import numpy as np
import pandas as pd
import faiss
import requests
from sentence_transformers import SentenceTransformer

OLLAMA_API_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "qwen2.5:3b"

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_PATH = os.path.join(BASE_DIR, "dataset", "master_dataset.csv")

_engine_instance = None

# ==============================
# OUT-OF-DOMAIN KEYWORDS
# ==============================
IN_DOMAIN_KEYWORDS = [
    # AI / ML / DL general
    "ai", "artificial intelligence", "machine learning", "deep learning",
    "neural network", "model", "algorithm", "dataset", "training", "testing",
    "overfitting", "underfitting", "bias", "variance", "feature", "label",
    "classification", "regression", "clustering", "prediction", "accuracy",
    "loss", "gradient", "optimizer", "epoch", "batch", "layer", "weight",
    "activation", "function", "parameter", "hyperparameter",
    # Specific algorithms
    "linear regression", "logistic regression", "decision tree", "random forest",
    "svm", "support vector", "knn", "k-nearest", "naive bayes", "bayes",
    "adaboost", "boosting", "bagging", "ensemble", "xgboost",
    "k-means", "dbscan", "pca", "dimensionality reduction",
    # Deep learning
    "ann", "cnn", "rnn", "lstm", "gru", "transformer", "attention",
    "backpropagation", "feedforward", "convolutional", "recurrent",
    "autoencoder", "gan", "generative", "bert", "gpt",
    "dropout", "batch normalization", "pooling", "embedding",
    # Concepts
    "supervised", "unsupervised", "reinforcement", "semi-supervised",
    "cross validation", "confusion matrix", "precision", "recall", "f1",
    "roc", "auc", "learning rate", "momentum", "regularization",
    "lasso", "ridge", "softmax", "sigmoid", "relu", "perceptron",
    "knowledge representation", "search algorithm", "intelligent agent",
    "expert system", "fuzzy logic", "genetic algorithm",
]

OUT_OF_DOMAIN_PATTERNS = [
    # Math / arithmetic
    r'\d+\s*[\+\-\*\/\%\^]\s*\d+',   # e.g. 3+3, 10*5
    r'\bsolve\b.*\bequation\b',
    r'\bintegral\b|\bderivative\b|\bcalculus\b',
    # General knowledge
    r'\bwho is\b|\bwho was\b|\bwho are\b',
    r'\bwhat is the capital\b|\bcountry\b|\bcity\b',
    r'\bweather\b|\btemperature\b|\bclimate\b',
    r'\bcook\b|\brecipe\b|\bfood\b|\bingredient\b',
    r'\bmovie\b|\bfilm\b|\bactor\b|\bsong\b|\bmusic\b',
    r'\bsport\b|\bcricket\b|\bfootball\b|\bnba\b|\bipl\b',
    r'\bhistory of\b(?!.*(?:ai|machine learning|neural|deep learning))',
    r'\bpolitics\b|\belection\b|\bpresident\b|\bprime minister\b',
    r'\bstock\b|\bshare price\b|\bcrypto\b|\bbitcoin\b',
    r'\bjoke\b|\bfunny\b|\bmeme\b',
    r'\btranslate\b|\blanguage\b(?!.*(?:python|programming|code))',
]

POLITE_REFUSALS = {
    "math":    "I appreciate the question, but I'm specialized in **Artificial Intelligence, Machine Learning, and Deep Learning** only. For math calculations, please use a calculator or a general-purpose assistant. 😊",
    "general": "That's outside my area of expertise! I'm an AI tutor focused exclusively on **Artificial Intelligence, Machine Learning, and Deep Learning**. Please ask me something within those domains and I'll be happy to help! 🎓",
    "unclear": "I couldn't find a confident answer for that in my academic dataset. Could you rephrase your question? I cover topics in **AI, Machine Learning, and Deep Learning**. 🤔",
}


def is_out_of_domain(question: str) -> tuple[bool, str]:
    """
    Returns (True, reason) if question is out of domain, else (False, "").
    Uses whole-word matching to prevent substring false positives (e.g. "ken" matching "knn").
    """
    q = question.lower().strip()

    # Check for arithmetic patterns first
    if re.search(r'\d+\s*[\+\-\*\/\%\^]\s*\d+', q):
        return True, "math"

    # Check explicit out-of-domain patterns
    for pattern in OUT_OF_DOMAIN_PATTERNS:
        if re.search(pattern, q):
            return True, "general"

    # Check if ANY in-domain keyword exists using WHOLE WORD matching
    # This prevents "ken" from matching "knn", "can" from matching "cnn" etc.
    for keyword in IN_DOMAIN_KEYWORDS:
        # Escape special chars in keyword, then wrap in word boundaries
        escaped = re.escape(keyword)
        # For short abbreviations (<=4 chars) use exact word boundary
        # For longer phrases use simple substring (phrases are specific enough)
        if len(keyword) <= 4:
            pattern = r'\b' + escaped + r'\b'
        else:
            pattern = escaped
        if re.search(pattern, q):
            return False, ""

    # If no in-domain keyword found AND question is short/vague
    word_count = len(q.split())
    if word_count <= 5:
        return True, "general"

    # Longer question without keywords — give benefit of doubt, let RAG try
    return False, ""


class RAGEngine:

    def __init__(self):
        if not os.path.exists(CSV_PATH):
            raise Exception(f"master_dataset.csv not found at {CSV_PATH}")
        self.df = pd.read_csv(CSV_PATH)
        if self.df.empty:
            raise Exception("Dataset is empty.")
        self.texts = self.df["content"].fillna("").tolist()
        self.embed_model = SentenceTransformer("all-MiniLM-L6-v2")
        self.embeddings = self.embed_model.encode(self.texts, show_progress_bar=True)
        dimension = self.embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(np.array(self.embeddings).astype("float32"))
        print(f"✅ RAG initialized with {len(self.texts)} chunks.")

    def clean_text(self, text):
        return re.sub(r"\s+", " ", text).strip()

    def answer_question(self, question, marks=1):
        question = self.clean_text(question)

        # ── STEP 1: Out-of-domain check ──
        out, reason = is_out_of_domain(question)
        if out:
            return POLITE_REFUSALS[reason], 0.0, "Out of Domain"

        # ── STEP 2: RAG retrieval ──
        q_emb = self.embed_model.encode([question])
        D, I = self.index.search(np.array(q_emb).astype("float32"), k=3)
        best_distance = D[0][0]
        confidence = float(round(1 / (1 + best_distance), 4))

        # ── STEP 3: Confidence threshold — too far from any chunk ──
        if best_distance > 1.8:
            return POLITE_REFUSALS["unclear"], 0.0, "Out of Domain"

        retrieved_chunks = [self.texts[i] for i in I[0]]
        context = " ".join(retrieved_chunks)
        subject = self.df.iloc[I[0][0]]["subject"]

        if marks == 1:
            word_limit = "40-60 words"
        elif marks == 2:
            word_limit = "100-130 words"
        else:
            word_limit = "300-350 words"

        extra = ""
        if marks >= 5:
            extra = """## Key Points
List the most important takeaways as bullet points.

## Applications
Real-world uses or examples that ground the concept.

## Example
A concrete, illustrative example."""

        prompt = f"""You are a world-class academic tutor specializing in Artificial Intelligence, Machine Learning, and Deep Learning.

RULES:
- Answer using ONLY the provided context. Do not use outside knowledge.
- Keep your answer within {word_limit}.
- Use **bold** for key terms on first mention.
- Be direct — lead with the answer, then explain.

Context:
{context}

Question:
{question}

Answer:

## Definition
A concise, precise definition of the concept.

## Explanation
A clear walkthrough of how it works and why it matters.

{extra}
"""
        try:
            response = requests.post(
                OLLAMA_API_URL,
                json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False},
                timeout=60
            )
            response.raise_for_status()
            data = response.json()
            answer = data.get("response", "").strip()
            if not answer:
                return POLITE_REFUSALS["unclear"], 0.0, subject
        except Exception as e:
            print(f"Ollama Error: {e}")
            # Fallback to raw context if Ollama is down
            answer = f"**Based on academic dataset:**\n\n{context[:600]}..."

        return answer, confidence, subject


def get_rag_engine():
    global _engine_instance
    if _engine_instance is None:
        _engine_instance = RAGEngine()
    return _engine_instance