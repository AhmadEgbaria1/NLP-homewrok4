import sys
import os
import json
import re
import numpy as np
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity


# ---------- Tokenization ----------
# Extract Hebrew word tokens (ignores numbers/latin/punctuation). Keeps internal quotes/apostrophes.
HEB_TOKEN_RE = re.compile(r"[א-ת]+(?:[\"״׳']+[א-ת]+)*")

def tokenize_sentence(sentence: str):
    return HEB_TOKEN_RE.findall(sentence)


# ---------- Load corpus  ----------
def load_sentences(path):
    """
      Load a JSONL corpus robustly.
      Each line is expected to be a JSON object with a 'sentence_text' field.
      We skip empty/invalid lines and keep both tokenized and original sentences.
      """
    sentences = []
    original_sentences = []

    with open(path, encoding="utf8") as f:
        for line in f:
            line = line.strip()
            if not line or line in ("[", "]"):
                continue
            if line.endswith(","):
                line = line[:-1]

            try:
                obj = json.loads(line)
                sent = obj.get("sentence_text", "")
                tokens = tokenize_sentence(sent)
                if tokens:
                    sentences.append(tokens)
                    original_sentences.append(sent)
            except Exception:
                # Robustness: ignore malformed JSON lines
                continue

    return sentences, original_sentences


# ---------- Sentence embedding ----------
# Mean vector of all tokens that exist in the model vocabulary
def sentence_embedding(tokens, model):
    vectors = [model.wv[w] for w in tokens if w in model.wv]
    if not vectors:
        return None
    return np.mean(vectors, axis=0)

# ---------- Word replacement helper ----------
def pick_replacement(model, target, topn=3, positive=None, negative=None):
    """
        Pick a replacement word for 'target' using Word2Vec most_similar.
        If 'positive' is not given, we default to [target].
        Returns None if target is OOV or if most_similar fails.
        """
    wv = model.wv
    if target not in wv:
        return None

    pos = positive[:] if positive else []
    neg = negative[:] if negative else []


    if not pos:
        pos = [target]

    try:
        candidates = wv.most_similar(positive=pos, negative=neg, topn=topn)
    except Exception:
        return None


    for cand, _score in candidates:
        if cand != target:
            return cand
    return None


def replace_word_in_sentence(sentence, old, new):
    """
       Replace a single exact word occurrence (first match only).
       Note: \b may be imperfect for Hebrew in edge cases, but works for many clean tokens.
       """
    pattern = rf'\b{re.escape(old)}\b'
    return re.sub(pattern, new, sentence, count=1)

# ---------- Part D ----------
def do_section_d(model, out_dir):
    """
        Replace the words marked 'in red' in the given 5 sentences.
        The list RED_WORDS_PER_SENT must match the exact words required by the assignment.
        """
    SENTS = [
        "בשעות הקרובות נפתח את הדיון בסעיף הבא שעל סדר היום",
        "אבקש מהנוכחים להתיישב כדי שנוכל להתחיל.",
        "אנו מודים לצוות המקצועי על עבודתו.",
        "הנושא יועבר להמשך טיפול בוועדת המשנה.",
        "ההצעה הובאה להצבעה ואושרה."
    ]

    # the red words
    RED_WORDS_PER_SENT = [
        ["בשעות", "בסעיף"],
        ["אבקש", "להתיישב"],
        ["לצוות", "עבודתו"],
        ["הנושא", "טיפול"],
        ["הובאה", "ואושרה"]
    ]



    CONTROL = {

        (0, "בשעות"): {
            "positive": [ "השעות","זמן"],
            "negative": ["טיסות"]
        },
        (0, "בסעיף"): {
            "positive": ["בנושא", "בעניין"],
            "negative": ["ג", "סעיף"]
        },



        (1, "להתיישב"): {
            "positive": ["לשבת", "להתכנס"],
            "negative": ["פצצות", "מלחמה"]
        },


        (2, "לצוות"): {
            "positive": [ "לעובדים"],
            "negative": ["למזכירות"]
        },
        (2, "עבודתו"): {
            "positive": ["פעילותו",],
            "negative": ["מהמועד"]
        },




        (4, "הובאה"): {
            "positive": ["הוגשה", "הוצגה"],
            "negative": ["שתדון"]
        },
        (4, "ואושרה"): {
            "positive": ["התקבלה", "אושרה"],
            "negative": ["אסיה", "ציונות"]
        }
    }

    out_path = os.path.join(out_dir, "red_words_sentences.txt")
    with open(out_path, "w", encoding="utf8") as f:
        for i, sent in enumerate(SENTS, start=1):
            new_sent = sent
            replaced_pairs = []

            for red in RED_WORDS_PER_SENT[i-1]:
                cfg = CONTROL.get((i-1, red), {})
                repl = pick_replacement(
                    model,
                    red,
                    topn=cfg.get("topn", 3),
                    positive=cfg.get("positive"),
                    negative=cfg.get("negative")
                )


                if not repl:
                    continue

                new_sent = replace_word_in_sentence(new_sent, red, repl)
                replaced_pairs.append((red, repl))

            f.write(f"{i}: {sent}: {new_sent}\n")
            if replaced_pairs:
                pairs_str = ",".join([f"({a}:{b})" for a, b in replaced_pairs])
            else:
                pairs_str = ""
            f.write(f"replaced words: {pairs_str}\n")

   # print("Wrote:", out_path)
'''
def print_pair_similarities(model: Word2Vec):
        pairs = [
            ("בעד", "נגד"),
            ("נמוך", "גבוה"),
            ("אסור", "מותר"),
        ]

        for w1, w2 in pairs:
            if w1 not in model.wv or w2 not in model.wv:
                missing = [w for w in (w1, w2) if w not in model.wv]
                print(f"{w1}-{w2}: missing from vocab -> {missing}")
                continue

            sim = model.wv.similarity(w1, w2)  # cosine similarity
            print(f"similarity({w1}, {w2}) = {sim:.4f}")
'''
# ---------- MAIN ----------
def main():
    if len(sys.argv) != 3:
        #print("Usage: python vec2word_knesset.py <corpus.jsonl> <output_dir>")
        sys.exit(1)

    corpus_path = sys.argv[1]
    out_dir = sys.argv[2]
    os.makedirs(out_dir, exist_ok=True)

    # Part A: load + train Word2Vec
    tokenized_sentences, original_sentences = load_sentences(corpus_path)
   # print("Number of sentences:", len(tokenized_sentences))

    # Word2Vec hyperparameters:
    # vector_size: embedding dimension
    # window: context window size
    # min_count: ignore words with total frequency < min_count
    try:
        model = Word2Vec(
            sentences=tokenized_sentences,
            vector_size=77,
            window=8,
            min_count=2,
            workers=4,
            epochs=10
        )
    except Exception as e:
       # print("Error during Word2Vec training:", e)
        sys.exit(1)

    try:
        model.save(os.path.join(out_dir, "knesset_word2vec.model"))
    except Exception as e:
        #print("Error saving Word2Vec model:", e)
        sys.exit(1)

    # Part B.1: for each given word, find the top-5 most similar words
    # Similarity is computed using cosine similarity between embedding vectors (via model.wv.similarity).
    words = ["יום", "אישה", "דרך", "ארוך", "תוכנית", "אוהב", "אסור", "איתן", "זכות"]
    with open(os.path.join(out_dir, "knesset_similar_words.txt"), "w", encoding="utf8") as f:
        for w in words:
            if w not in model.wv:
                f.write(f"{w}: word not in vocabulary\n")
                continue

            sims = []
            for other in model.wv.index_to_key:
                if other != w:
                    sims.append((other, model.wv.similarity(w, other)))

            sims.sort(key=lambda x: x[1], reverse=True)
            top5 = sims[:5]
            f.write(f"{w}: " + ", ".join(f"({x},{s:.3f})" for x, s in top5) + "\n")

    # Part B.2 + B.3: sentence similarity using mean word vectors (excluding OOV tokens)
    # We keep only sentences with at least 4 *valid* tokens (tokens that exist in model vocabulary).
    sent_vectors = []
    valid_sentences = []

    for toks, sent in zip(tokenized_sentences, original_sentences):
        valid_tokens = [w for w in toks if w in model.wv]
        if len(valid_tokens) >= 4:
            vec = sentence_embedding(toks, model)
            if vec is not None:
                sent_vectors.append(vec)
                valid_sentences.append(sent)

    sent_vectors = np.array(sent_vectors)
    chosen_idx = list(range(min(10, len(sent_vectors))))

    with open(os.path.join(out_dir, "knesset_similar_sentences.txt"), "w", encoding="utf8") as f:
        for i in chosen_idx:
            scores = cosine_similarity(sent_vectors[i].reshape(1, -1), sent_vectors)[0]
            scores[i] = -1  # exclude self-match
            best = int(np.argmax(scores))
            f.write(f"{valid_sentences[i]}: most similar sentence: {valid_sentences[best]}\n")

    do_section_d(model, out_dir)
    #print_pair_similarities(model)



if __name__ == "__main__":
    main()

