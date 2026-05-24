# Sentiment Analysis of Amazon Musical Instrument Reviews

A natural language processing project that classifies Amazon musical-instrument reviews into **positive**, **neutral**, and **negative** sentiment. The pipeline cleans raw review text, handles severe class imbalance with SMOTE, vectorizes with TF-IDF, and compares two classifiers — **K-Nearest Neighbors** and **Logistic Regression** — tuned with grid search.

> **Result:** Logistic Regression reaches **95% accuracy**, substantially outperforming K-NN (**65.38%**).

---

## 📊 Dataset
the **Amazon Musical Instruments Reviews** dataset from Kaggle, containing **10,261 reviews** collected between **2004 and 2014**.

| Column | Description |
| --- | --- |
| `reviewerID` | ID of the reviewer |
| `asin` | ID of the product |
| `reviewerName` | Name of the reviewer |
| `helpful` | Helpfulness rating, e.g. `2/3` |
| `reviewText` | Full text of the review |
| `overall` | Product rating (1–5) |
| `summary` | Short summary of the review |
| `unixReviewTime` | Review time (Unix) |
| `reviewTime` | Review time (raw) |

The target label is derived from the `overall` rating:

| Rating | Sentiment | Encoded |
| --- | --- | --- |
| 4–5 | positive | 2 |
| 3 | neutral | 1 |
| 1–2 | negative | 0 |

---

## 🔧 Workflow

### 1. Data preparation
- Merge `reviewText` + `summary` into a single `reviews` field.
- Drop identifier/metadata columns and rows with missing text.
- Map the 1–5 rating to a 3-class sentiment label.

### 2. Text preprocessing
- Remove numbers, punctuation, extra whitespace, and non-English characters.
- **Custom stop-word list** instead of NLTK's default — negation words such as *not*, *don't*, and *wouldn't* are deliberately **kept**, since they are critical signals for negative sentiment.
- Lemmatization with NLTK's `WordNetLemmatizer`.

### 3. Handling class imbalance
The raw class distribution is heavily skewed:

| Sentiment | Count |
| --- | --- |
| positive | 9,015 |
| neutral | 772 |
| negative | 467 |

**SMOTE** (Synthetic Minority Oversampling Technique) is applied to balance all three classes to 9,015 samples each.

### 4. Feature extraction
- **TF-IDF** vectorization with bigrams (`ngram_range=(2, 2)`) and `max_features=5000`.

### 5. Modeling
A 75/25 train-test split, with hyperparameters tuned via 5-fold `GridSearchCV`:

- **KNN** — searched over `n_neighbors`, `weights`, and distance `metric`. Best: `n_neighbors=3, weights='distance', metric='euclidean'`.
- **Logistic Regression** — searched over the regularization strength `C` (50 values on a log scale). Best: `C=10000`.

---

## 📈 Results

| Model | Accuracy | Precision | Recall |
| --- | --- | --- | --- |
| **Logistic Regression** | **94.40%** | **94.80%** | **94.66%** |
| K-NN | 66.76% | 77.44% | 66.76% |

**Logistic Regression outperformed K-NN in all metrics**, indicating it is more accurate and efficient at classifying customer reviews. K-NN struggles badly on the positive class (recall ≈ 5%), while Logistic Regression maintains strong, balanced performance across all three sentiment classes.

### Sample predictions
```text
"Worth the price. This is a nice one."                          → positive
"The quality of this item is terrible. I'm very disappointed."  → negative
"It's an okay product. Nothing special."                        → neutral
```

---
## 💡 Key takeaways

- Preserving negation words in the stop-word list matters for sentiment tasks because these words (e.g., *not, never, doesn’t*) carry important meaning that can completely change the sentiment of a sentence, so removing them would distort the true emotional polarity.
- SMOTE is essential here — without it, models would simply predict the dominant positive class.
- A well-regularized linear model (Logistic Regression) outperforms KNN on high-dimensional sparse TF-IDF features, where distance-based methods degrade.

---

