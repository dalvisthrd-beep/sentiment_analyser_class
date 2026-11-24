# 📱 Cell Phone Review Sentiment Analyser

A Multi-Model **Sentiment Analysis** System for consumer reviews from the **Amazon Cell Phones & Accessories Dataset**. Built with **NLTK VADER**, **CardiffNLP Twitter-RoBERTa**, and **Sarcasm-Aware Intent Detection**.

---

## 🌟 Overview & Capabilities

* **Core Task: Sentiment Classification**:
  Classifies comments and reviews into **Positive**, **Neutral**, or **Negative**.
* **Sarcasm-Aware Intent Detection**:
  Standard sentiment models rely on surface word valences and misclassify non-literal text. This system identifies whether a comment contains sarcasm to accurately resolve the true underlying sentiment:
  * **Positive Sarcasm (Mock Complaints)**: Detects playful negative exaggeration (*"Terrible! My wife stole it immediately"*) and correctly classifies the sentiment as **Positive**.
  * **Negative Sarcasm (Ironic Praise)**: Detects hyperbolic compliments (*"Brilliant! The screen shattered on day one"*) and correctly inverts the sentiment to **Negative**.
* **Models Compared**:
  1. **NLTK VADER**: Rule- and lexicon-based sentiment scorer.
  2. **Twitter-RoBERTa**: Pretrained transformer (`cardiffnlp/twitter-roberta-base-sentiment-latest`).
  3. **Sarcasm-Aware RoBERTa**: Transformer representations fused with contrastive sarcasm cues for pragmatic sentiment detection.

---

## 🏋️ Train and evaluate the models

From the `NLP` folder, run:

```bash
python train_sentiment_models.py
```

The script uses Amazon review ratings for three-class sentiment labels
(1-2 stars: negative, 3: neutral, 4-5: positive), tunes VADER's decision
thresholds on a validation split, and fine-tunes the CardiffNLP RoBERTa model. By default, the pretrained encoder is frozen and its
classification head is adapted to the review data, which makes the run
practical on CPU-only machines. Pass `--full-finetune` to update the entire
encoder when a suitable GPU is available.
It also trains a separate binary sarcasm detector on
`train-balanced-sarcasm.csv`. The sarcasm corpus does not contain sentiment
polarity labels, so it is not used as positive/neutral/negative training data.
The existing hand-labelled sarcasm examples are included as additional
sentiment fine-tuning examples.

The default run samples up to 1,000 Amazon reviews per sentiment class and
trains for one epoch. Use `--max-per-class 0` to use every Amazon review or
adjust `--epochs` and `--sarcasm-epochs` as needed. Training metrics and model
artifacts are written to `trained_models/`; the app reports held-out metrics
from those files instead of displaying hard-coded accuracy claims.

The binary sarcasm detector identifies likely sarcasm but cannot determine
whether the intended sentiment is positive or negative. The fine-tuned
three-class RoBERTa model predicts sentiment and also receives the curated
sarcasm examples during training. When explicit mock-complaint or ironic-praise
polarity cues match, the app applies that polarity to the final label and
continues to show the raw RoBERTa probabilities separately.

---

## 🚀 Running Locally

1. **Clone the repository**:
   ```bash
   git clone https://github.com/dalvisthrd-beep/sentiment_analyser_class.git
   cd sentiment_analyser_class
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Launch the Streamlit app**:
   ```bash
   streamlit run app.py
   ```

