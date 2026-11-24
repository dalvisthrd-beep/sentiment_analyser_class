"""Streamlit Web Application: Cell Phone Review Sentiment Analyser.

Deployed for Y4 Data Science - NLP Lab 2.
Identifies customer review sentiment (Positive, Neutral, Negative)
using NLTK VADER, Baseline RoBERTa, and Sarcasm-Aware Intent Detection.
"""

from __future__ import annotations

import importlib.machinery
import json
from pathlib import Path
import re
import sys
import types

# Windows workaround for torchvision C++ extension
if "torchvision" not in sys.modules:
    _tv = types.ModuleType("torchvision")
    _tv.__spec__ = importlib.machinery.ModuleSpec("torchvision", loader=None)
    _tv.io = types.ModuleType("torchvision.io")
    _tv.io.ImageReadMode = object()
    _tv.io.decode_image = lambda *args, **kwargs: None
    _tv.transforms = types.ModuleType("torchvision.transforms")

    class _InterpolationMode:
        NEAREST_EXACT = BOX = BILINEAR = HAMMING = BICUBIC = LANCZOS = 0

    _tv.transforms.InterpolationMode = _InterpolationMode
    _tv.transforms.functional = types.ModuleType(
        "torchvision.transforms.functional"
    )
    _tv.transforms.functional.pil_to_tensor = lambda image: image
    sys.modules["torchvision"] = _tv
    sys.modules["torchvision.io"] = _tv.io
    sys.modules["torchvision.transforms"] = _tv.transforms
    sys.modules["torchvision.transforms.functional"] = _tv.transforms.functional

import nltk
import pandas as pd
import streamlit as st
import torch
import torch.nn.functional as F

from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from sarcasm_detector import predict_sarcasm, resolve_sentiment

st.set_page_config(
    page_title="Cell Phone Review Sentiment Analyser",
    page_icon="📱",
    layout="wide",
)

MODEL_DIR = Path(__file__).resolve().parent / "trained_models"
SENTIMENT_MODEL_DIR = MODEL_DIR / "roberta_sentiment"
VADER_THRESHOLDS_PATH = MODEL_DIR / "vader_thresholds.json"
TRAINING_METRICS_PATH = MODEL_DIR / "training_metrics.json"

# Download resources silently
for pkg in ("vader_lexicon", "punkt", "punkt_tab", "stopwords", "wordnet"):
    nltk.download(pkg, quiet=True)


@st.cache_resource(show_spinner="Loading VADER Lexicon...")
def load_vader():
    return SentimentIntensityAnalyzer()


@st.cache_resource(show_spinner="Loading Twitter-RoBERTa Transformer...")
def load_roberta():
    model_source = (
        str(SENTIMENT_MODEL_DIR)
        if SENTIMENT_MODEL_DIR.exists()
        else "cardiffnlp/twitter-roberta-base-sentiment-latest"
    )
    tok = AutoTokenizer.from_pretrained(model_source)
    mod = AutoModelForSequenceClassification.from_pretrained(model_source)
    mod.eval()
    return tok, mod


@st.cache_data(show_spinner=False)
def load_vader_thresholds() -> dict:
    if not VADER_THRESHOLDS_PATH.exists():
        return {"negative": -0.05, "positive": 0.05}
    with VADER_THRESHOLDS_PATH.open(encoding="utf-8") as source:
        thresholds = json.load(source)
    if (
        not isinstance(thresholds, dict)
        or not isinstance(thresholds.get("negative"), (int, float))
        or not isinstance(thresholds.get("positive"), (int, float))
        or thresholds["negative"] >= thresholds["positive"]
    ):
        raise ValueError(f"Invalid VADER thresholds in {VADER_THRESHOLDS_PATH}")
    return thresholds


def clean_text(text: str) -> str:
    text = str(text or "").lower()
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"[^a-z0-9'\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def preprocess_nltk(text: str) -> dict:
    cleaned = clean_text(text)
    tokens = word_tokenize(cleaned)
    stops = set(stopwords.words("english"))
    lemma = WordNetLemmatizer()
    filtered = [t for t in tokens if t not in stops and len(t) > 1]
    lemmas = [lemma.lemmatize(t) for t in filtered]
    return {
        "cleaned": cleaned,
        "tokens": tokens,
        "filtered": filtered,
        "lemmas": lemmas,
        "processed_text": " ".join(lemmas),
    }


def score_vader_app(sia, text: str) -> dict:
    scores = sia.polarity_scores(text)
    comp = scores["compound"]
    thresholds = load_vader_thresholds()
    label = (
        "positive"
        if comp >= thresholds["positive"]
        else ("negative" if comp <= thresholds["negative"] else "neutral")
    )
    return {
        "label": label,
        "compound": comp,
        "thresholds": thresholds,
        "scores": {
            "negative": scores["neg"],
            "neutral": scores["neu"],
            "positive": scores["pos"],
        },
    }


def score_roberta_app(tokenizer, model, text: str) -> dict:
    encoded = tokenizer(
        text, truncation=True, max_length=192, return_tensors="pt"
    )
    device = next(model.parameters()).device
    encoded = {key: value.to(device) for key, value in encoded.items()}
    with torch.inference_mode():
        probs = F.softmax(model(**encoded).logits, dim=-1).squeeze(0).cpu().tolist()
    id2label = model.config.id2label
    labels = []
    for index in range(len(probs)):
        raw_label = id2label.get(index, id2label.get(str(index), ""))
        normalized = str(raw_label).lower()
        label = next(
            (sentiment for sentiment in ("negative", "neutral", "positive")
             if sentiment in normalized),
            ("negative", "neutral", "positive")[index],
        )
        labels.append(label)
    scores = {labels[i]: float(probs[i]) for i in range(len(probs))}
    label = max(scores, key=scores.get)
    return {"label": label, "scores": scores}


# Application Header
st.title("📱 Cell Phone Review Sentiment Analyser")
st.markdown(
    "Accurately analyze review sentiment (**Positive**, **Neutral**, or"
    " **Negative**) using **NLTK VADER**, **CardiffNLP Twitter-RoBERTa**, "
    "and **Sarcasm-Aware Intent Detection** to uncover genuine customer intent"
    " behind sarcastic comments."
)

sia = load_vader()
tok, mod = load_roberta()

tab1, tab2, tab3 = st.tabs([
    "🔍 Live Sentiment Analyser",
    "💡 Sarcasm & Sentiment Impact",
    "📊 Benchmarks & Performance",
])

EXAMPLES = {
    "Positive Review (With Playful Sarcasm)": (
        "Terrible product! Now my wife keeps stealing my phone charger because"
        " it charges in 10 minutes flat. Worst purchase ever, now I have to"
        " buy another one for myself!"
    ),
    "Positive Review (With Mock Warning)": (
        "Do NOT buy this wireless charger unless you enjoy having 100% battery"
        " before you even finish your morning coffee."
    ),
    "Negative Review (With Ironic Praise)": (
        "Oh brilliant! The phone case broke on day one. Truly a masterpiece of"
        " engineering."
    ),
    "Negative Review (With False Compliment)": (
        "Great design! If you love having your phone heat up like an electric"
        " toaster after 5 minutes of charging."
    ),
    "Standard Genuine Positive": (
        "This case fits my phone perfectly, feels super premium, and provides"
        " great grip."
    ),
    "Standard Genuine Negative": (
        "Very disappointed. The cable frayed and stopped working after 3 days"
        " of gentle use."
    ),
    "Standard Genuine Neutral": (
        "It is a standard black USB-C cable. Arrived on time and performs"
        " averagely."
    ),
}

with tab1:
    col_in, col_opt = st.columns([3, 1])
    with col_opt:
        sample_sel = st.selectbox(
            "Load Sample Comment:", ["(Type custom)"] + list(EXAMPLES.keys())
        )
    default_val = EXAMPLES[sample_sel] if sample_sel in EXAMPLES else ""
    user_review = st.text_area(
        "Enter Comment / Review:",
        value=default_val,
        height=120,
        placeholder="Type or paste a cell phone review here...",
    )

    if st.button("Analyze Sentiment", type="primary"):
        if not user_review.strip():
            st.warning("Please enter a comment to analyze.")
        else:
            proc = preprocess_nltk(user_review)
            v_res = score_vader_app(sia, user_review)
            r_res = score_roberta_app(tok, mod, user_review)
            sarcasm_res = predict_sarcasm(user_review)
            raw_sentiment = r_res["label"]
            final_res = resolve_sentiment(raw_sentiment, sarcasm_res)
            final_sentiment = final_res["label"]
            sarcasm_adjusted = final_res["adjusted"]
            final_sentiment_display = final_sentiment.upper()
            confidence_note = (
                "(adjusted using explicit sarcasm-polarity cues)"
                if sarcasm_adjusted
                else f"({r_res['scores'][final_sentiment]*100:.1f}% confidence)"
            )

            # Main Sentiment Outcome Banner
            if final_sentiment_display == "POSITIVE":
                st.success(
                    f"###  Final Sentiment Verdict: **POSITIVE** "
                    f"{confidence_note}"
                )
            elif final_sentiment_display == "NEGATIVE":
                st.error(
                    f"### ❌ Final Sentiment Verdict: **NEGATIVE** "
                    f"{confidence_note}"
                )
            else:
                st.info(
                    f"### ⚖️ Final Sentiment Verdict: **NEUTRAL** "
                    f"{confidence_note}"
                )

            # Sarcasm Context Box
            if sarcasm_res["is_sarcastic"]:
                if sarcasm_res["sarcasm_type"] == "positive_sarcasm":
                    st.info(
                        "🎭 **Positive sarcasm cues detected** (mock complaint)."
                        f" The final sentiment is **{final_sentiment}**; raw"
                        f" RoBERTa predicted **{raw_sentiment}**."
                    )
                elif sarcasm_res["sarcasm_type"] == "negative_sarcasm":
                    st.warning(
                        "🎭 **Negative sarcasm cues detected** (ironic praise)."
                        f" The final sentiment is **{final_sentiment}**; raw"
                        f" RoBERTa predicted **{raw_sentiment}**."
                    )
                else:
                    st.warning(
                        "🎭 The sarcasm detector flagged this comment. The"
                        " sarcasm corpus provides sarcasm labels only, so it"
                        " cannot determine whether the intended sentiment is"
                        " positive or negative by itself."
                    )
                if sarcasm_res["confidence"] is not None:
                    st.caption(
                        (
                            "Sarcasm polarity is determined by the explicit"
                            " cue rules; the trained detector's probability"
                            f" was {sarcasm_res['confidence'] * 100:.1f}%."
                            if sarcasm_res["method"]
                            == "trained detector and polarity cues"
                            else
                            f"Sarcasm probability: "
                            f"{sarcasm_res['confidence'] * 100:.1f}%."
                        )
                    )
            else:
                st.caption(
                    f"No sarcasm detected by {sarcasm_res['method']}."
                )

            st.divider()

            # 3-Model Comparison Grid
            c1, c2, c3 = st.columns(3)

            with c1:
                st.subheader("1. NLTK VADER")
                st.metric("Sentiment", v_res["label"].capitalize())
                st.metric(
                    "Compound Score",
                    f"{v_res['compound']:.3f}",
                    help=(
                        f"Positive >= {v_res['thresholds']['positive']:.2f}; "
                        f"negative <= {v_res['thresholds']['negative']:.2f}; "
                        "otherwise neutral."
                    ),
                )
                st.bar_chart(pd.DataFrame({"Score": v_res["scores"]}))
                st.caption(
                    "Rule & Lexicon Model: Evaluates raw positive/negative"
                    " dictionary words."
                )

            with c2:
                st.subheader("2. Baseline RoBERTa")
                st.metric("Sentiment", r_res["label"].capitalize())
                st.metric(
                    "Confidence",
                    f"{r_res['scores'][r_res['label']]*100:.1f}%",
                )
                st.bar_chart(pd.DataFrame({"Probability": r_res["scores"]}))
                st.caption(
                    "Standard Transformer: Highly accurate on literal text, but"
                    " literal on sarcasm."
                )

            with c3:
                st.subheader("3. RoBERTa Sentiment + Sarcasm Signal")
                st.metric(
                    "Sentiment",
                    final_sentiment.capitalize(),
                    delta="Sarcasm-adjusted" if sarcasm_adjusted else None,
                )
                st.metric(
                    "Sarcasm",
                    (
                        f"{sarcasm_res['confidence'] * 100:.1f}%"
                        if sarcasm_res["confidence"] is not None
                        else ("Detected" if sarcasm_res["is_sarcastic"] else "Not detected")
                    ),
                )
                st.bar_chart(pd.DataFrame({"Raw RoBERTa probability": r_res["scores"]}))
                st.caption(
                    "The final label applies polarity-specific sarcasm cues when"
                    " present; the chart shows the raw RoBERTa probabilities."
                )

            st.divider()
            st.subheader("NLTK Preprocessing Pipeline Details")
            t1, t2 = st.columns(2)
            with t1:
                st.write("**Filtered Tokens (No Stopwords):**")
                st.code(
                    " ".join(proc["filtered"]) if proc["filtered"] else "(none)"
                )
            with t2:
                st.write("**Lemmatized Tokens (WordNet):**")
                st.code(
                    " ".join(proc["lemmas"]) if proc["lemmas"] else "(none)"
                )

with tab2:
    st.subheader(
        "💡 How Sarcasm Detection Empowers Accurate Sentiment Analysis"
    )
    st.write(
        "In customer feedback, sarcasm flips the literal polarity of words."
        " Without sarcasm detection, sentiment analysers misclassify comments:"
    )

    col_pos, col_neg = st.columns(2)

    with col_pos:
        st.markdown("### 🟢 Identifying Positive Sentiment in Sarcasm")
        st.write(
            "Customers frequently write **mock complaints in 5-star"
            " reviews**:"
        )
        st.code(
            "Terrible case! Now my wife won't stop stealing it.\nWorst"
            " purchase ever, now I have to buy another one!\nDo NOT buy this"
            " charger unless you want 100% battery in 15 minutes."
        )
        st.markdown("""
        * **Surface Words**: *"Terrible"*, *"worst"*, *"stealing"* (Negative).
        * **True Sentiment**: **POSITIVE** (The product exceeded expectations).
        * **How Our Model Resolves It**: Sarcasm detection detects the mock complaint structure and correctly classifies the sentiment as **POSITIVE**.
        """)

    with col_neg:
        st.markdown("### 🔴 Identifying Negative Sentiment in Sarcasm")
        st.write(
            "Frustrated customers use **ironic praise in 1-star reviews**:"
        )
        st.code(
            "Oh brilliant! The phone case broke on day one. Truly a"
            " masterpiece.\nGreat design! If you love having your phone heat up"
            " like an electric toaster."
        )
        st.markdown("""
        * **Surface Words**: *"brilliant"*, *"masterpiece"*, *"great design"* (Positive).
        * **True Sentiment**: **NEGATIVE** (Product failed completely).
        * **How Our Model Resolves It**: Detects hyperbolic praise paired with defect nouns, properly assigning **NEGATIVE** sentiment.
        """)

    st.divider()
    img_path = Path(__file__).resolve().parent / "sarcasm_comparison.png"
    if img_path.exists():
        st.image(
            str(img_path),
            caption=(
                "Sentiment Classification Accuracy on Sarcastic vs Literal"
                " Comments"
            ),
            use_container_width=True,
        )

with tab3:
    st.subheader("Measured Model Performance")
    if TRAINING_METRICS_PATH.exists():
        with TRAINING_METRICS_PATH.open(encoding="utf-8") as source:
            metrics = json.load(source)
        vader_metrics = metrics["vader"]
        rows = [
            {
                "Task": "3-class sentiment",
                "Model": "VADER (default thresholds)",
                "Accuracy": vader_metrics["baseline_accuracy"],
                "Macro-F1": vader_metrics["baseline_macro_f1"],
            },
            {
                "Task": "3-class sentiment",
                "Model": "VADER (calibrated thresholds)",
                "Accuracy": vader_metrics["accuracy"],
                "Macro-F1": vader_metrics["macro_f1"],
            },
            {
                "Task": "3-class sentiment",
                "Model": "Fine-tuned RoBERTa",
                "Accuracy": metrics["roberta"]["accuracy"],
                "Macro-F1": metrics["roberta"]["macro_f1"],
            },
            {
                "Task": "Binary sarcasm detection",
                "Model": "Hashing + SGD",
                "Accuracy": metrics["sarcasm_detector"]["accuracy"],
                "Macro-F1": metrics["sarcasm_detector"]["macro_f1"],
            },
        ]
        st.dataframe(
            pd.DataFrame(rows).style.format(
                {"Accuracy": "{:.1%}", "Macro-F1": "{:.3f}"}
            ),
            hide_index=True,
            use_container_width=True,
        )
        st.caption(
            "Sentiment labels are derived from Amazon star ratings: 1-2 negative,"
            " 3 neutral, 4-5 positive. Sarcasm detection is a separate binary"
            " task and does not determine sentiment polarity."
        )
    else:
        st.info(
            "No training metrics found. Run "
            "`python train_sentiment_models.py` from the NLP folder to train"
            " RoBERTa, calibrate VADER, and fit the sarcasm detector."
        )
