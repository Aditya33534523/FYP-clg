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
# FOLLOW-UP / CONVERSATIONAL PATTERNS
# ==============================
FOLLOW_UP_PATTERNS = [
    r'\bexplain\s*(more|again|better|further|in detail)\b',
    r'\btell\s*(me)?\s*(more|again)\b',
    r'\belaborate\b',
    r'\bwhat\s*do\s*you\s*mean\b',
    r'\bcan\s*you\s*(explain|clarify|elaborate)\b',
    r'\bnot\s*(clear|satisfy|satisfied|enough|understood|helpful)\b',
    r"\bdidn'?t\s*(satisfy|understand|get|help)\b",
    r'\bi\s*don\'?t\s*(understand|get)\b',
    r'\bgive\s*(me)?\s*(more|another|a better|an? example)\b',
    r'\bmore\s*(detail|info|information|explanation)\b',
    r'\bwhat\s*about\b',
    r'\bhow\s*(exactly|specifically)\b',
    r'\bsimplify\b',
    r'\bin\s*simple\s*(terms|words|language)\b',
]

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
    "unclear": "I couldn't find a confident answer for that in my academic dataset. Could you rephrase your question or be more specific? I cover topics in **AI, Machine Learning, and Deep Learning**. 🤔",
    "followup": "I'd love to help you understand better! Could you **rephrase your question** with the specific topic? For example:\n\n• \"Explain CNN in more detail\"\n• \"What is backpropagation? Explain simply\"\n• \"Give me an example of overfitting\"\n\nThis helps me find the best answer from my academic dataset! 📚",
}


def is_follow_up(question: str) -> bool:
    """Detect conversational follow-ups like 'explain more', 'didn't satisfy', etc."""
    q = question.lower().strip()
    for pattern in FOLLOW_UP_PATTERNS:
        if re.search(pattern, q):
            return True
    return False


def is_out_of_domain(question: str) -> tuple[bool, str]:
    """
    Returns (True, reason) if question is out of domain, else (False, "").
    Uses whole-word matching to prevent substring false positives (e.g. "ken" matching "knn").
    """
    q = question.lower().strip()

    # ── Check for follow-up/feedback first — never reject these ──
    if is_follow_up(q):
        # Check if a domain keyword is also present => let RAG handle
        for keyword in IN_DOMAIN_KEYWORDS:
            escaped = re.escape(keyword)
            if len(keyword) <= 4:
                pattern = r'\b' + escaped + r'\b'
            else:
                pattern = escaped
            if re.search(pattern, q):
                return False, ""
        # Pure follow-up with no topic => ask user to rephrase
        return True, "followup"

    # Check for arithmetic patterns first
    if re.search(r'\d+\s*[\+\-\*\/\%\^]\s*\d+', q):
        return True, "math"

    # Check explicit out-of-domain patterns
    for pattern in OUT_OF_DOMAIN_PATTERNS:
        if re.search(pattern, q):
            return True, "general"

    # Check if ANY in-domain keyword exists using WHOLE WORD matching
    for keyword in IN_DOMAIN_KEYWORDS:
        escaped = re.escape(keyword)
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

    def clean_context(self, text):
        """Clean OCR/PDF extraction artifacts from retrieved context chunks."""
        t = text
        # Fix common OCR broken words from PDF extraction
        t = re.sub(r'\b(\w+)\s(eld|tion|ing|ment|ness|able|ible|ful|less|ous|ive|ity|ence|ance|ure|ory|ist|ism)\b',
                   r'\1\2', t)
        # Fix 'sub eld' -> 'subfield', 'de ned' -> 'defined', etc.
        t = t.replace('sub eld', 'subfield')
        t = t.replace('de ned', 'defined')
        t = t.replace('de ne', 'define')
        t = t.replace('brie y', 'briefly')
        t = t.replace(' rst ', ' first ')
        t = t.replace(' nite ', ' finite ')
        t = t.replace('classi cation', 'classification')
        t = t.replace('classi er', 'classifier')
        t = t.replace('speci c', 'specific')
        t = t.replace('signi cant', 'significant')
        t = t.replace('arti cial', 'artificial')
        t = t.replace('di erent', 'different')
        t = t.replace('e ective', 'effective')
        t = t.replace('ef cient', 'efficient')
        # Remove stray mathematical notation artifacts
        t = re.sub(r'\{\(xi,\s*yi\)\}N\s*i=', '', t)
        t = re.sub(r'\bx\(\d+\)', '', t)
        t = re.sub(r'x\(j\)', 'feature j', t)
        t = re.sub(r'\bi=\s*,', '', t)
        t = re.sub(r'j\s*=\s*,\s*\.\s*\.\s*\.\s*,\s*D', '', t)
        # Remove reference numbers/citations like [1], (1), etc.
        t = re.sub(r'\[\d+\]', '', t)
        # Collapse multiple spaces
        t = re.sub(r'\s{2,}', ' ', t).strip()
        return t

    def clean_answer(self, answer):
        """Post-process LLM answer to ensure clean, student-friendly output."""
        a = answer.strip()
        # Remove "Answer:" prefix if model echoed it
        a = re.sub(r'^\s*Answer:\s*', '', a, flags=re.IGNORECASE)
        # Remove "Based on the context" filler phrases
        a = re.sub(r'^\s*(Based on the (provided |given )?context,?\s*)', '', a, flags=re.IGNORECASE)
        a = re.sub(r'^\s*(According to the (provided |given )?context,?\s*)', '', a, flags=re.IGNORECASE)
        # Remove trailing incomplete sentences (ends with comma or no period)
        lines = a.rstrip().split('\n')
        if lines and not lines[-1].rstrip().endswith(('.', '!', '?', ':', '*')):
            # Only trim if the last line looks truncated (< 20 chars)
            if len(lines[-1].strip()) < 20 and len(lines) > 1:
                lines = lines[:-1]
        a = '\n'.join(lines)
        # Ensure starts with uppercase
        if a and a[0].islower():
            a = a[0].upper() + a[1:]
        return a.strip()

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
        context = self.clean_context(" ".join(retrieved_chunks))
        subject = self.df.iloc[I[0][0]]["subject"]

        prompt = self._build_prompt(question, context, marks)
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
            answer = self.clean_answer(answer)
        except Exception as e:
            print(f"Ollama Error: {e}")
            answer = ("⚠️ The AI engine is temporarily unavailable. "
                      "Please make sure **Ollama** is running and try again.\n\n"
                      f"Your question was about **{subject}** — I found relevant material "
                      "in my dataset but couldn't generate a response right now.")

        return answer, confidence, subject

    def _build_prompt(self, question, context, marks):
        """Build a student-friendly prompt based on answer length."""
        if marks == 1:
            return f"""You are a friendly AI tutor helping students learn AI, ML, and Deep Learning.

RULES:
- Use ONLY the context below to answer. Do NOT add outside knowledge.
- Keep your answer concise: 40-60 words.
- Use **bold** for key technical terms.
- Write in clear, simple language a student can understand.
- Start directly with the answer — no filler phrases like "Based on the context".

Context:
{context}

Student's Question: {question}

Your Answer:"""
        elif marks == 2:
            return f"""You are a friendly AI tutor helping students learn AI, ML, and Deep Learning.

RULES:
- Use ONLY the context below to answer. Do NOT add outside knowledge.
- Write 100-130 words.
- Use **bold** for key technical terms on first mention.
- Write clearly so a student can understand and use this in an exam.
- Start with the core answer, then explain with an example if possible.
- Do NOT start with "Based on the context" or similar filler.

Context:
{context}

Student's Question: {question}

Your Answer:"""
        else:
            return f"""You are a friendly AI tutor helping students learn AI, ML, and Deep Learning.

RULES:
- Use ONLY the context below to answer. Do NOT add outside knowledge.
- Write a detailed 300-350 word answer.
- Use **bold** for key technical terms on first mention.
- Write clearly — a student should be able to read this and write it in an exam.
- Use proper markdown formatting with headings.
- Do NOT start with "Based on the context" or similar filler.

Context:
{context}

Student's Question: {question}

Provide your answer with these sections:

## Definition
A clear, one-sentence definition.

## Explanation
How it works and why it matters, explained step-by-step.

## Key Points
- Important takeaways as bullet points

## Example
A concrete, easy-to-understand example.

Your Answer:"""

    def answer_question_stream(self, question, marks=1):
        """
        Generator that yields tokens one-by-one from Ollama streaming API.
        First yields a metadata dict, then yields string tokens.
        """
        question = self.clean_text(question)

        # Out-of-domain check (instant)
        out, reason = is_out_of_domain(question)
        if out:
            yield {"type": "meta", "confidence": 0.0, "subject": "Out of Domain"}
            yield {"type": "done", "full_answer": POLITE_REFUSALS[reason]}
            return

        # RAG retrieval
        q_emb = self.embed_model.encode([question])
        D, I = self.index.search(np.array(q_emb).astype("float32"), k=3)
        best_distance = D[0][0]
        confidence = float(round(1 / (1 + best_distance), 4))

        if best_distance > 1.8:
            yield {"type": "meta", "confidence": 0.0, "subject": "Out of Domain"}
            yield {"type": "done", "full_answer": POLITE_REFUSALS["unclear"]}
            return

        retrieved_chunks = [self.texts[i] for i in I[0]]
        context = self.clean_context(" ".join(retrieved_chunks))
        subject = self.df.iloc[I[0][0]]["subject"]

        prompt = self._build_prompt(question, context, marks)

        # Yield metadata first
        yield {"type": "meta", "confidence": confidence, "subject": subject}

        # Stream from Ollama
        full_answer = ""
        try:
            response = requests.post(
                OLLAMA_API_URL,
                json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": True},
                stream=True,
                timeout=120
            )
            response.raise_for_status()
            import json
            for line in response.iter_lines():
                if line:
                    chunk = json.loads(line)
                    token = chunk.get("response", "")
                    if token:
                        full_answer += token
                        yield {"type": "token", "token": token}
                    if chunk.get("done", False):
                        break
        except Exception as e:
            print(f"Ollama Stream Error: {e}")
            if not full_answer:
                full_answer = ("⚠️ The AI engine is temporarily unavailable. "
                              "Please make sure **Ollama** is running and try again.\n\n"
                              f"Your question was about **{subject}** — I found relevant material "
                              "in my dataset but couldn't generate a response right now.")
                yield {"type": "token", "token": full_answer}

        # Post-process the final answer
        full_answer = self.clean_answer(full_answer)
        yield {"type": "done", "full_answer": full_answer}


def get_rag_engine():
    global _engine_instance
    if _engine_instance is None:
        _engine_instance = RAGEngine()
    return _engine_instance