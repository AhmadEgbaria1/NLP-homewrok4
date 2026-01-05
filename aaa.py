import sys
import os
import json
import re
import numpy as np
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity


# ---------- Tokenization ----------
HEB_TOKEN_RE = re.compile(r"[א-ת]+(?:[\"״׳']+[א-ת]+)*")

def tokenize_sentence(sentence: str):
    return HEB_TOKEN_RE.findall(sentence)


# ---------- Load corpus (robust JSONL) ----------
def load_sentences(path):
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
                continue

    return sentences, original_sentences


# ---------- Sentence embedding ----------
def sentence_embedding(tokens, model):
    vectors = [model.wv[w] for w in tokens if w in model.wv]
    if not vectors:
        return None
    return np.mean(vectors, axis=0)


def pick_replacement(model, target, topn=3, positive=None, negative=None):
    """
    מחזיר מילה חלופית למילת יעד.
    - אם target לא ב-vocab: מחזיר None
    - משתמש ב most_similar עם אפשרות ל positive/negative
    """
    wv = model.wv
    if target not in wv:
        return None

    pos = positive[:] if positive else []
    neg = negative[:] if negative else []

    # אם לא הועברו positive בכלל, נתחיל מהמילה עצמה
    if not pos:
        pos = [target]

    try:
        candidates = wv.most_similar(positive=pos, negative=neg, topn=topn)
    except Exception:
        return None

    # בחר את הראשון שהוא לא אותה מילה
    for cand, _score in candidates:
        if cand != target:
            return cand
    return None


def replace_word_in_sentence(sentence, old, new):
    # מחליף מילה שלמה, גם אם יש סימן פיסוק אחריה
    pattern = rf'\b{re.escape(old)}\b'
    return re.sub(pattern, new, sentence, count=1)


def do_section_d(model, out_dir):
    # 5 המשפטים מהתרגיל
    SENTS = [
        "בשעות הקרובות נפתח את הדיון בסעיף הבא שעל סדר היום",
        "אבקש מהנוכחים להתיישב כדי שנוכל להתחיל.",
        "אנו מודים לצוות המקצועי על עבודתו.",
        "הנושא יועבר להמשך טיפול בוועדת המשנה.",
        "ההצעה הובאה להצבעה ואושרה."
    ]

    # כאן אתה קובע מה “באדום”.
    # כרגע שמתי ניחוש של מילה אחת בכל משפט — תשנה לפי מה שמסומן אצלך.
    RED_WORDS_PER_SENT = [
        ["בשעות", "בסעיף"],
        ["אבקש", "להתיישב"],
        ["לצוות", "עבודתו"],
        ["הנושא", "טיפול"],
        ["הובאה", "ואושרה"]
    ]

    # אם אתה רוצה לכוון עם positive/negative, אפשר להגדיר פה לכל מילה אדומה:
    # המפתח הוא (index_sentence, red_word)
    # הערכים: dict עם positive/negative/topn
    CONTROL = {
        # משפט 1: בשעות הקרובות ... בסעיף הבא
        (0, "בשעות"): {
            "positive": [ "השעות","זמן"],
            "negative": ["טיסות"]
        },
        (0, "בסעיף"): {
            "positive": ["בנושא", "בעניין"],
            "negative": ["ג", "סעיף"]
        },

        # משפט 2: אבקש ... להתיישב

        (1, "להתיישב"): {
            "positive": ["לשבת", "להתכנס"],
            "negative": ["פצצות", "מלחמה"]
        },

        # משפט 3: לצוות המקצועי ... עבודתו
        (2, "לצוות"): {
            "positive": [ "לעובדים"],
            "negative": ["למזכירות"]
        },
        (2, "עבודתו"): {
            "positive": ["פעילותו",],
            "negative": ["מהמועד"]
        },



        # משפט 5: הובאה ... ואושרה
        (4, "הובאה"): {
            "positive": ["הוגשה", "הוצגה"],
            "negative": ["שתדון"]
        },
        (4, "ואושרה"): {
            "positive": ["התקבלה", "אושרה"],
            "negative": ["אסיה", "ציונות"]
        }
    }

    out_path = os.path.join(out_dir, "sentences_words_red.txt")
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

                # אם לא מצא (לא במילון וכו’) – מדלגים
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

    print("Wrote:", out_path)

# ---------- MAIN ----------
def main():
    if len(sys.argv) != 3:
        print("Usage: python vec2word_knesset.py <corpus.jsonl> <output_dir>")
        sys.exit(1)

    corpus_path = sys.argv[1]
    out_dir = sys.argv[2]
    os.makedirs(out_dir, exist_ok=True)

    # ----- Part A: Load + Train -----
    tokenized_sentences, original_sentences = load_sentences(corpus_path)
    print("Number of sentences:", len(tokenized_sentences))

    model = Word2Vec(
        sentences=tokenized_sentences,
        vector_size=77,
        window=8,
        min_count=2,
        workers=4,
        epochs=10
    )

    model_path = os.path.join(out_dir, "knesset_word2vec.model")
    model.save(model_path)

    # ----- Part B.1: Similar words -----
    words = ["יום", "אישה", "דרך", "ארוך", "תוכנית", "אוהב", "אסור", "איתן", "זכות"]
    with open(os.path.join(out_dir, "words_similar_knesset.txt"), "w", encoding="utf8") as f:
        for w in words:
            if w not in model.wv:
                continue
            sims = model.wv.most_similar(w, topn=5)
            line = f"{w}: " + ", ".join([f"({x},{s:.3f})" for x, s in sims])
            f.write(line + "\n")

    # ----- Part B.2 + B.3: Sentence similarity -----
    sent_vectors = []
    valid_sentences = []

    for toks, sent in zip(tokenized_sentences, original_sentences):
        vec = sentence_embedding(toks, model)
        if vec is not None and len(toks) >= 4:
            sent_vectors.append(vec)
            valid_sentences.append(sent)

    sent_vectors = np.array(sent_vectors)
    # בחר 10 משפטים ראשונים "תקינים"
    chosen_idx = list(range(min(10, len(sent_vectors))))

    with open(os.path.join(out_dir, "sentences_similar_knesset.txt"), "w", encoding="utf8") as f:
        for i in chosen_idx:
            # דמיון של וקטור אחד מול כל הוקטורים (לא NxN!)
            scores = cosine_similarity(sent_vectors[i].reshape(1, -1), sent_vectors)[0]
            scores[i] = -1  # כדי לא לבחור את עצמו
            best = int(np.argmax(scores))
            f.write(f"{valid_sentences[i]}: most similar sentence: {valid_sentences[best]}\n")
    #part d
    do_section_d(model, out_dir)


if __name__ == "__main__":
    main()
