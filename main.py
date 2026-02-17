import os
import re
import glob
import textwrap
import requests
import numpy as np
import pandas as pd
from tqdm import tqdm

import spacy
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity, cosine_distances

# ----------------------------
# CONFIG
# ----------------------------

DATA_DIR = "data"
FILE_PATTERN = "*.txt"
MIN_PATIENT_CHARS = 50
N_CLUSTERS = 4
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"  # 384-dim embeddings[web:34][web:75]

# load spaCy model for dependency parsing[web:95][web:98]
nlp = spacy.load("en_core_web_sm")

# ----------------------------
# EMOTION-RESPONSE CONFIG
# ----------------------------

EMOTION_WORDS = [
    "anxious", "anxiety", "worried", "worry", "afraid", "scared",
    "fear", "panicked", "panic", "sad", "depressed", "down",
    "ashamed", "shame", "guilty", "guilt", "angry",
    "overwhelmed", "stressed", "stress", "nervous", "frustrated",
    "upset", "embarrassed", "annoyed", "irritated", "numb"
]

EMOTION_MARKERS_PATTERN = re.compile(
    r"\b(feel|feeling|felt|emotion|emotional|mood|anxious|worried|sad|depressed|angry|scared|afraid|overwhelmed|stressed)\b",
    re.IGNORECASE
)

FIRST_PERSON = re.compile(
    r"\b(i|i'm|im|i am|i’ve|ive|myself|me)\b",
    re.IGNORECASE
)

# Verbs/phrases indicating responses to emotions (avoidance, control, acceptance, etc.)[web:48][web:57][web:60]
RESPONSE_VERBS = [
    "avoid", "avoided", "avoiding",
    "push away", "pushed away", "pushing away",
    "suppress", "suppressed", "suppressing",
    "hold in", "holding in", "bottle up", "bottling up",
    "ignore", "ignoring", "block out", "blocking out",
    "distract myself", "distracted myself",
    "deal with", "dealt with", "handle", "handled",
    "cope with", "coping with",
    "control", "controlled", "controlling",
    "manage", "managed",
    "let myself feel", "let myself really feel",
    "allow myself to feel", "allow myself to",
    "sit with", "stay with", "be present with",
]

# Belief statements about having/allowing/expressing emotions[web:48][web:57][web:60]
BELIEF_PHRASES = [
    "i don't let myself feel",
    "i do not let myself feel",
    "i shouldn't feel", "i should not feel",
    "i can't stand feeling", "i cannot stand feeling",
    "i can't handle feeling", "i cannot handle feeling",
    "my emotions take over", "my feelings take over",
    "i lose control when i feel",
    "i lose control of my emotions",
    "it's a sign of weakness if i have",
    "it's wrong to feel",
]

# Patterns we explicitly want to exclude (common false positives)
EXCLUSION_PATTERNS = [
    r"\bi just don't feel like it\b",
    r"\bi don't feel like it\b",
    r"\bi don't really feel that\b",
    r"\bi don't really feel that way\b",
    r"\bi don't feel that way\b",
    r"\bi don't feel that\b",
    r"\bi don't feel\.\s*$",
    r"\bi don't feel\s*$",
]

MAX_EMOTION_SNIPPETS_PER_CONVO = 30

# ----------------------------
# 1. LOAD & FILTER TRANSCRIPTS
# ----------------------------

def load_transcripts_conversation_level(
    data_dir: str,
    file_pattern: str = "*.txt",
    min_patient_chars: int = 50
):
    all_files = glob.glob(os.path.join(data_dir, file_pattern))
    records = []

    patient_pattern = re.compile(
        r"^\s*(?:<p>\s*)?(?:PATIENT|CLIENT)\s*:\s*(.*)",
        re.IGNORECASE
    )

    counselor_pattern = re.compile(
        r"^\s*(?:<p>\s*)?(?:COUNSELOR|THERAPIST)\s*:\s*(.*)",
        re.IGNORECASE
    )

    for fp in tqdm(all_files, desc="Loading transcripts (conversation-level)"):
        with open(fp, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()

        has_patient = False
        has_counselor = False
        patient_utterances = []

        for line in lines:
            m_patient = patient_pattern.match(line)
            m_counselor = counselor_pattern.match(line)

            if m_patient:
                has_patient = True
                text = m_patient.group(1).strip()
                if text:
                    patient_utterances.append(text)
            elif m_counselor:
                has_counselor = True

        if not (has_patient and has_counselor):
            continue

        full_patient_text = " ".join(patient_utterances).strip()
        if len(full_patient_text) < min_patient_chars:
            continue

        records.append({
            "conversation_id": os.path.basename(fp),
            "patient_text": full_patient_text
        })

    df = pd.DataFrame(records)
    df = df.head(100)
    return df

# ----------------------------
# 2. BASIC UTILITIES
# ----------------------------

def split_into_sentences_for_spans(doc):
    # use spaCy sentence boundaries[web:95][web:98]
    return list(doc.sents)

def is_emotion_token(tok):
    lower = tok.lemma_.lower()
    if lower in ["feel", "felt", "feeling", "emotion", "emotional", "mood"]:
        return True
    if lower in EMOTION_WORDS:
        return True
    return False

def is_response_verb(tok):
    lower = tok.lemma_.lower()
    for phrase in RESPONSE_VERBS:
        main = phrase.split()[0]
        if lower == main:
            return True
    return False

def is_excluded_emotion_response_text(text: str) -> bool:
    s = text.lower()
    for pat in EXCLUSION_PATTERNS:
        if re.search(pat, s):
            return True
    return False

# ----------------------------
# 3. DEPENDENCY-BASED EMOTION-RESPONSE SPANS
# ----------------------------

def extract_emotion_response_spans_from_doc(doc):
    """
    Return spans where there is an explicit syntactic link between
    a response verb and an emotion phrase, using spaCy dependencies.
    """
    spans = []

    # 1) Within-sentence links
    for sent in doc.sents:
        # first-person requirement at the span level
        if not FIRST_PERSON.search(sent.text.lower()):
            continue

        emo_tokens = [tok for tok in sent if is_emotion_token(tok)]
        resp_tokens = [tok for tok in sent if is_response_verb(tok)]

        if not emo_tokens or not resp_tokens:
            continue

        # Check explicit dependency links: response ↔ emotion
        keep = False
        for e in emo_tokens:
            for r in resp_tokens:
                # require a direct syntactic relation, not just same sentence
                if (
                    e.head == r
                    or r.head == e
                    or any(child == e for child in r.children)
                    or any(child == r for child in e.children)
                ):
                    print("EMO:", e.text, "| lemma:", e.lemma_, "| dep:", e.dep_, "| head:", e.head.text)
                    print("RESP:", r.text, "| lemma:", r.lemma_, "| dep:", r.dep_, "| head:", r.head.text)
                    keep = True
                    break
            if keep:
                break

        if keep:
            spans.append(sent.text.strip())

    # 2) Cross-sentence emotion→response pairs (optional)
    sents = list(doc.sents)
    for i in range(len(sents) - 1):
        s1, s2 = sents[i], sents[i+1]
        txt1 = s1.text.strip()
        txt2 = s2.text.strip()
        span_txt = (txt1 + " " + txt2).strip().lower()

        if not FIRST_PERSON.search(span_txt):
            continue

        has_emo = any(is_emotion_token(t) for t in s1)
        has_resp = any(is_response_verb(t) for t in s2)

        if has_emo and has_resp:
            spans.append((txt1 + " " + txt2).strip())

    # de-duplicate
    unique = []
    seen = set()
    for sp in spans:
        key = sp.strip()
        if key not in seen:
            seen.add(key)
            unique.append(sp)
    return unique

# ----------------------------
# 4. GLOBAL + LOCAL (EMOTION-RESPONSE) EMBEDDINGS
# ----------------------------

def build_conversation_embeddings(df: pd.DataFrame, model: SentenceTransformer):
    """
    For each conversation:
      - global_emb: embedding of full patient_text.
      - emotion_emb: mean embedding of spans where dependency parsing indicates
        a response verb is related to an emotion phrase (within or across adjacent sentences).
      - final_emb: concatenation [global_emb ; emotion_emb] (or global+zeros if no spans).
    """
    global_embs = []
    emotion_embs = []
    final_embs = []
    snippet_rows = []

    for _, row in tqdm(df.iterrows(), total=len(df),
                       desc="Building conversation embeddings (dep-based)"):
        conv_id = row["conversation_id"]
        text = row["patient_text"]

        # 1) global conversation embedding
        global_emb = model.encode([text], batch_size=1, show_progress_bar=False)[0]
        global_embs.append(global_emb)

        # 2) parse with spaCy and extract emotion-response spans
        doc = nlp(text)
        spans = extract_emotion_response_spans_from_doc(doc)

        if spans:
            spans = spans[:MAX_EMOTION_SNIPPETS_PER_CONVO]
            span_embs = model.encode(spans, batch_size=32, show_progress_bar=False)
            emotion_emb = span_embs.mean(axis=0)
            emotion_embs.append(emotion_emb)

            for sp in spans:
                snippet_rows.append({
                    "conversation_id": conv_id,
                    "emotion_response_span": sp,
                })

            final_emb = np.concatenate([global_emb, emotion_emb], axis=0)
        else:
            dim = global_emb.shape[0]
            emotion_emb = np.zeros(dim, dtype=np.float32)
            emotion_embs.append(emotion_emb)
            final_emb = np.concatenate([global_emb, emotion_emb], axis=0)

        final_embs.append(final_emb)

    global_embs = np.vstack(global_embs)
    emotion_embs = np.vstack(emotion_embs)
    final_embs = np.vstack(final_embs)

    if snippet_rows:
        out_df = pd.DataFrame(snippet_rows)
        out_df.to_csv("conversation_emotion_response_spans_dep.csv", index=False)
        print("Saved emotion-response spans to conversation_emotion_response_spans_dep.csv")
    else:
        print("No emotion-response spans found; nothing written.")

    return global_embs, emotion_embs, final_embs

# ----------------------------
# 5. K-MEANS CLUSTERING & EVALUATION
# ----------------------------

def cluster_conversations(
    embeddings: np.ndarray,
    n_clusters: int
):
    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=42,
        n_init=10
    )
    cluster_labels = kmeans.fit_predict(embeddings)
    return kmeans, cluster_labels


def evaluate_k_range(embeddings, k_range):
    scores = {}
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(embeddings)
        score = silhouette_score(embeddings, labels)
        scores[k] = score
        print(f"K={k}, silhouette={score:.3f}")
    return scores

def get_cluster_representatives(
    df: pd.DataFrame,
    embeddings: np.ndarray,
    label_col: str,
    n_examples: int = 5
):
    reps = {}
    for c in sorted(df[label_col].unique()):
        cluster_idx = np.where(df[label_col].values == c)[0]
        cluster_embs = embeddings[cluster_idx]

        centroid = cluster_embs.mean(axis=0, keepdims=True)
        dists = cosine_distances(cluster_embs, centroid).flatten()
        order = np.argsort(dists)

        examples = []
        for rank in order[:n_examples]:
            global_idx = cluster_idx[rank]
            conv_id = df.iloc[global_idx]["conversation_id"]
            examples.append((global_idx, conv_id, float(dists[rank])))

        reps[c] = examples

    return reps

# ----------------------------
# 6. LLM SUMMARIZATION
# ----------------------------

PERPLEXITY_API_KEY = os.getenv("PERPLEXITY_API_KEY")
PERPLEXITY_API_URL = "https://api.perplexity.ai/chat/completions"
PERPLEXITY_MODEL = "sonar"  # example; confirm in docs[web:34][web:75]

def summarize_cluster_with_llm(cluster_spans, max_chars: int = 800):
    if not cluster_spans:
        return "No clear pattern of responses to emotions detected in this cluster."

    joined = " ".join(cluster_spans)
    joined = joined[:max_chars]

    prompt = textwrap.dedent(f"""
        You are summarizing how patients respond to and think about their emotions
        in the context of psychotherapy conversations.

        Spans:
        {joined}

        In one sentence, describe the dominant relationship to emotions in this cluster.
    """).strip()

    if not PERPLEXITY_API_KEY:
        return "(LLM summary placeholder) Cluster of conversations with a shared way of responding to emotions."

    headers = {
        "Authorization": f"Bearer {PERPLEXITY_API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": PERPLEXITY_MODEL,
        "messages": [
            {"role": "system", "content": "You are a concise scientific summarization assistant."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.2,
        "max_tokens": 80,
    }

    try:
        resp = requests.post(PERPLEXITY_API_URL, headers=headers, json=payload, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print("Perplexity error:", e)
        return "(LLM error fallback) Cluster of conversations with a shared way of responding to emotions."

# ----------------------------
# 7. CLUSTER INSPECTION
# ----------------------------

def inspect_clusters_dep(df: pd.DataFrame, label_col: str, n_examples: int = 3):
    print(f"\n=== {label_col} overview ===")
    print(df[label_col].value_counts().sort_index())

    snippet_path = "conversation_emotion_response_spans_dep.csv"
    if not os.path.exists(snippet_path):
        print("No snippet file found; skipping example spans.")
        return

    snippets_df = pd.read_csv(snippet_path)

    for c in sorted(df[label_col].unique()):
        print(f"\n--- Cluster {c} ({label_col}) ---")
        cluster_conv_ids = df.loc[df[label_col] == c, "conversation_id"].unique()
        print(f"Cluster {c} has {len(cluster_conv_ids)} conversations.")

        # Get spans for conversations in this cluster
        cluster_spans = snippets_df[snippets_df["conversation_id"].isin(cluster_conv_ids)]

        if cluster_spans.empty:
            print("No emotion-response spans found for this cluster.")
            continue

        # Sample spans from multiple conversations, not just the first file
        sampled_spans = []
        seen_conv_ids = set()

        # iterate over rows, but try to collect spans from as many different
        # conversation_ids as possible up to some cap (e.g. 60)
        for _, row in cluster_spans.iterrows():
            conv_id = row["conversation_id"]
            span = row["emotion_response_span"]
            sampled_spans.append(span)
            seen_conv_ids.add(conv_id)
            if len(sampled_spans) >= 60:
                break

        # LLM summary over a diverse sample
        cluster_desc = summarize_cluster_with_llm(sampled_spans)

        print(f"Cluster {c} description (LLM): {cluster_desc}")
        print(f"Cluster {c} example conversations in sample: {list(seen_conv_ids)[:10]}")

        # Print a few example spans from *different* conversations
        print("\nTop example emotion-response spans (from multiple conversations):")
        printed = 0
        used_conv_ids = set()
        for _, row in cluster_spans.iterrows():
            conv_id = row["conversation_id"]
            span = row["emotion_response_span"]
            if conv_id in used_conv_ids:
                continue
            print(f"[{conv_id}] {span}")
            used_conv_ids.add(conv_id)
            printed += 1
            if printed >= n_examples:
                break

# ----------------------------
# 8. MAIN PIPELINE
# ----------------------------

def main():
    df = load_transcripts_conversation_level(
        DATA_DIR,
        FILE_PATTERN,
        MIN_PATIENT_CHARS
    )
    print(f"Kept {len(df)} conversations with clear PATIENT/COUNSELOR tags.")

    if len(df) == 0:
        print("No conversations found, exiting.")
        return

    model = SentenceTransformer(MODEL_NAME)

    print("\nBuilding global + dependency-based emotion-response embeddings...")
    global_embs, emotion_embs, final_embs = build_conversation_embeddings(df, model)

    print("\nClustering on combined (global + emotion-response) embeddings...")
    kmeans_final, labels_final = cluster_conversations(final_embs, N_CLUSTERS)
    df["emotion_dep_cluster"] = labels_final

    df.to_csv("conversation_level_emotion_dep_clusters.csv", index=False)
    print("Saved cluster assignments to conversation_level_emotion_dep_clusters.csv")

    reps_final = get_cluster_representatives(df, final_embs, "emotion_dep_cluster", n_examples=3)
    for c, exs in reps_final.items():
        print(f"\n=== Combined-embedding Cluster {c} representatives ===")
        for idx, conv_id, dist in exs:
            print(f"  - conv {conv_id} (idx={idx}, dist={dist:.4f})")

    inspect_clusters_dep(df, "emotion_dep_cluster", n_examples=5)


if __name__ == "__main__":
    main()
