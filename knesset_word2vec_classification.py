import sys
import json
import re

import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import classification_report


HEB_TOKEN_RE = re.compile(r"[א-ת]+(?:[\"״׳']+[א-ת]+)*")

def tokenize_sentence(sentence: str):
    return HEB_TOKEN_RE.findall(sentence)
# ==========================================
# CONFIGURATION
# ==========================================
SPEAKER_1 = "ראובן ריבלין"  # top1 speaker from homework3
SPEAKER_2 = "א' בורג"  # top2 speaker from homework3


def normalize_speaker(s: str) -> str:
    """
    Normalizes speaker names to handle variations (e.g., 'R. Rivlin' -> 'Reuven Rivlin').
    """
    if not isinstance(s, str):
        return ""

    s = s.strip()

    # --- SPEAKER 1 VARIATIONS ---
    if s in ["רובי ריבלין", "ר' ריבלין", "ראובן ריבלין>", "ראובן ריבלין"]:
        return SPEAKER_1

    # --- SPEAKER 2 VARIATIONS ---
    if s in ["אברהם בורג", "א' בורג", "אברום בורג"]:
        return SPEAKER_2

    # If it's neither, return the original string
    return s


def get_sentence_embedding(sentence_tokens, model):
    """
    Calculates the sentence embedding by averaging the word vectors.
    """
    valid_vectors = []
    for word in sentence_tokens:
        if word in model.wv.key_to_index:
            valid_vectors.append(model.wv[word])

    if not valid_vectors:
        return np.zeros(model.vector_size)

    return np.mean(valid_vectors, axis=0)


def load_corpus(path):
    """
    Load the jsonl corpus robustly.
    """
    rows = []

    def pick_field(obj, candidates):
        for k in candidates:
            v = obj.get(k)
            if isinstance(v, str) and v.strip():
                return v.strip()
        return None

    print(f"Loading corpus from: {path}...")
    with open(path, encoding='utf8') as f:
        for line in f:
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue

            speaker = pick_field(obj, ['speaker', 'speaker_name', 'speakerName', 'member_name', 'name'])
            text = pick_field(obj, ['text', 'sentence', 'sentence_text', 'utterance'])

            # fallback logic
            if speaker is None:
                for k, v in obj.items():
                    if 'speaker' in str(k).lower() and isinstance(v, str) and v.strip():
                        speaker = v.strip()
                        break
            if text is None:
                for k, v in obj.items():
                    lk = str(k).lower()
                    if (('text' in lk) or ('sentence' in lk)) and isinstance(v, str) and v.strip():
                        text = v.strip()
                        break

            if speaker and text:
                rows.append({'speaker': speaker, 'text': text})

    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError("Corpus loaded as empty. Check JSON keys.")
    return df


def main():
    if len(sys.argv) < 3:
        print("Usage: python knesset_word2vec_classification.py <path/to/corpus> <path/to/model>")
        sys.exit(1)

    corpus_path = sys.argv[1]
    model_path = sys.argv[2]

    # 1. Load Model
    print(f"Loading Word2Vec model from: {model_path}...")
    try:
        model = Word2Vec.load(model_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)

    # 2. Load Data
    try:
        df = load_corpus(corpus_path)
    except ValueError as e:
        print(e)
        sys.exit(1)

    # 3. NORMALIZE SPEAKERS (New Step)
    print("Normalizing speaker names...")
    df['speaker'] = df['speaker'].apply(normalize_speaker)

    # 4. Filter for Specific Speakers
    print(f"Filtering for normalized speakers: '{SPEAKER_1}' and '{SPEAKER_2}'...")
    filtered_df = df[df['speaker'].isin([SPEAKER_1, SPEAKER_2])].copy()

    if filtered_df.empty:
        print(f"Error: No sentences found. Check if your variations in 'normalize_speaker' match the data.")
        sys.exit(1)

    print(f"Found {len(filtered_df)} sentences.")
    # 4.5 Balance the two speakers by downsampling the larger class
    counts = filtered_df['speaker'].value_counts()
    n = counts.min()

    filtered_df = (filtered_df.groupby('speaker', group_keys=False)
                   .sample(n=n, random_state=42)
                   .reset_index(drop=True))

    print("Before balance:", counts.to_dict())
    print("After balance:", filtered_df['speaker'].value_counts().to_dict())

    # 5. Feature Extraction
    print("Generating sentence embeddings...")
    filtered_df['tokens'] = filtered_df['text'].apply(tokenize_sentence)
    X = np.array([get_sentence_embedding(tokens, model) for tokens in filtered_df['tokens']])
    y = filtered_df['speaker'].values

    # 6. Classification
    knn = KNeighborsClassifier(n_neighbors=5)

    print("Running 5-Fold Cross Validation...")
    try:
        y_pred = cross_val_predict(knn, X, y, cv=5)
        print("\nClassification Report:")
        print("======================")
        print(classification_report(y, y_pred))
    except ValueError as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()