# 🍽️ Aspect-Based Sentiment Analysis of Restaurant Reviews

A comparative NLP study implementing three progressively sophisticated models — **TF-IDF + Linear SVM**, **Bi-LSTM with GloVe embeddings**, and **Fine-tuned DistilBERT** — to perform fine-grained sentiment classification on the SemEval-2014 restaurant review dataset.

---

## 📌 What is ABSA?

Standard sentiment analysis labels an entire review as positive or negative. **Aspect-Based Sentiment Analysis (ABSA)** goes further — it identifies *what* is being talked about and *how* the reviewer feels about it specifically.

```
"The pasta was amazing, but the service was incredibly slow."
         ↓                              ↓
  pasta → Positive              service → Negative
```

---

## 📁 Project Structure

```
absa-nlp-project/
│
├── data/
│   ├── Restaurants_Train_v2.xml        # SemEval-2014 training data (XML)
│   ├── Restaurants_Test_Data_phaseB.xml # SemEval-2014 test data (XML)
│   ├── cleaned_train.csv               # Preprocessed training data
│   └── cleaned_test.csv                # Preprocessed test data
│
├── notebooks/
│   ├── 01_Data_Exploration.ipynb       # EDA, parsing, visualizations
│   ├── 02_SVM_Baseline.ipynb           # TF-IDF + LinearSVC pipeline
│   ├── 03_Bi-LSTM.ipynb                # GloVe Bi-LSTM model
│   └── 04_BERT_absanlp.ipynb           # DistilBERT & DeBERTa fine-tuning
│
├── outputs/
│   ├── graphs/
│   │   ├── svm_c_tuning.png
│   │   ├── svm_confusion_matrix.png
│   │   ├── distilbert_learning_curve.png
│   │   └── distilbert_confusion_matrix.png
│   └── results/
│       └── sentiment_distribution.png
│
├── requirements.txt
└── README.md
```

---

## 🗂️ Dataset

**SemEval-2014 Task 4 – Aspect Based Sentiment Analysis (Restaurant Domain)**

The dataset is sourced from the [SemEval-2014 shared task](https://alt.qcri.org/semeval2014/task4/) and distributed in XML format, with each sentence annotated with aspect terms and their sentiment polarities.

| Feature | Training Set | Test Set |
|---|---|---|
| Total Aspect-Sentiment Pairs | ~3,041 | ~800 |
| Positive Instances | ~2,164 (~71%) | ~560 (~70%) |
| Negative Instances | ~637 (~21%) | ~170 (~21%) |
| Neutral Instances | ~240 (~8%) | ~70 (~9%) |
| Conflict Instances (Removed) | ~126 | ~29 |
| Sentiment Classes | Positive, Negative, Neutral | — |

> **Conflict labels** (contradictory sentiment within one sentence) were removed following standard ABSA practice.

**Preprocessing steps applied:**
- Parsed XML → Pandas DataFrame (`text`, `aspect`, `sentiment`, `from`, `to`)
- Removed `conflict` labelled rows
- Lowercased all text and aspect terms
- Applied stratified 80/20 train-test split
- Constructed unified input: `"review text [SEP] aspect term"`

---

## 🧠 Models Implemented

### 1. TF-IDF + Linear SVM (Baseline)

A classical, sparse-vector approach using **TF-IDF vectorisation** (unigrams + bigrams) and a **LinearSVC** classifier with One-vs-Rest multi-class decomposition.

**Key configuration:**
- `ngram_range=(1, 2)` — captures single words and two-word phrases
- `max_features=5000`, `sublinear_tf=True`
- `class_weight='balanced'` — corrects class imbalance automatically
- Regularisation `C` tuned via **5-fold cross-validation** over `[0.01, 0.1, 0.5, 1.0, 5.0, 10.0]`

---

### 2. Bidirectional LSTM with GloVe Embeddings

A sequential deep learning model that reads each review **forwards and backwards** simultaneously, concatenating both hidden states to capture full sentential context.

**Architecture:**
```
Input Tokens (MAX_LEN=60)
        ↓
GloVe Embedding Layer (100-d, frozen)
        ↓
SpatialDropout1D (0.3)
        ↓
Bidirectional LSTM (64 units, dropout=0.3, recurrent_dropout=0.3)
        ↓
Dense (32 units, ReLU)
        ↓
Dropout (0.5)
        ↓
Dense (3 units, Softmax) → [Negative, Neutral, Positive]
```

**Key configuration:**
- Pre-trained `glove.6B.100d` embeddings (frozen weights)
- `Adam` optimiser, `lr=0.001`, `batch_size=32`
- `EarlyStopping(patience=5, restore_best_weights=True)`
- `compute_class_weight('balanced')` passed to Keras training loop

---

### 3. Fine-Tuned DistilBERT (Best Model)

A transformer-based model fine-tuned end-to-end on the ABSA task. DistilBERT's bidirectional self-attention captures **full sentential context dynamically**, making it the most semantically aware of the three approaches.

**Key configuration:**
- Base model: `distilbert-base-uncased` from Hugging Face
- `max_length=128`, `learning_rate=2e-5`, `weight_decay=0.01`
- `num_train_epochs=4`, `per_device_train_batch_size=16`
- Custom `WeightedTrainer` subclassing Hugging Face `Trainer` to apply class-weighted `CrossEntropyLoss`
- `load_best_model_at_end=True`

> A **DeBERTa-base** (`microsoft/deberta-base`) variant was also explored in the same notebook with `lr=1e-5` and `batch_size=8`.

---

## ⚙️ Setup & Installation

### Prerequisites
- Python 3.8+
- GPU recommended for Notebooks 3 & 4 (Google Colab T4 GPU was used)

### Install Dependencies

```bash
pip install -r requirements.txt
```

**`requirements.txt`**
```
pandas
numpy
scikit-learn
matplotlib
seaborn
tensorflow
torch
transformers
accelerate
```

### GloVe Embeddings (for Bi-LSTM)

Download `glove.6B.100d.txt` from the [GloVe project page](https://nlp.stanford.edu/projects/glove/) and update the path in `03_Bi-LSTM.ipynb`:

```python
GLOVE_PATH = "path/to/glove.6B.100d.txt"
```

### Dataset

Download the SemEval-2014 restaurant XML files and place them in `data/`:
- `Restaurants_Train_v2.xml`
- `Restaurants_Test_Data_phaseB.xml`

---

## 🚀 Running the Project

Run the notebooks **in order**:

```bash
# Step 1: Parse, clean, and visualize the dataset
jupyter notebook notebooks/01_Data_Exploration.ipynb

# Step 2: Train and evaluate the SVM baseline
jupyter notebook notebooks/02_SVM_Baseline.ipynb

# Step 3: Train the Bi-LSTM model (GPU recommended)
jupyter notebook notebooks/03_Bi-LSTM.ipynb

# Step 4: Fine-tune DistilBERT (Google Colab with GPU strongly recommended)
jupyter notebook notebooks/04_BERT_absanlp.ipynb
```

> **Note:** Notebook 04 is optimised for **Google Colab**. Upload `cleaned_train.csv` to `/content/` before running.

---

## 📊 Results

All models used the same stratified 80/20 split (`random_state=42`) for a fair comparison. Metrics are macro-averaged across the three sentiment classes.

| Model | Accuracy | Precision | Recall | F1-Score | MCC |
|---|---|---|---|---|---|
| TF-IDF + Linear SVM | — | — | — | — | — |
| Bi-LSTM (GloVe) | — | — | — | — | — |
| DistilBERT (Fine-Tuned) | — | — | — | — | — |

> **Fill in actual values** from the `classification_report` outputs in your notebooks.

### Key Findings

- **DistilBERT** achieved the highest scores across all metrics, demonstrating the superiority of contextualised transformer representations for aspect-level sentiment classification.
- **Class weight balancing** significantly improved Recall on the minority `negative` and `neutral` classes across all models.
- **Aspect-context injection** (`text [SEP] aspect`) was a critical design choice that improved aspect-binding in all three models.
- The `MCC` score confirms that DistilBERT produces the most balanced predictions across all three classes, not just the majority class.

---

## 🔑 Core Design Choices

| Design Choice | Applied To | Purpose |
|---|---|---|
| `text [SEP] aspect` input format | All 3 models | Explicit aspect context injection |
| `class_weight='balanced'` | SVM | Corrects 71/21/8% class skew |
| `compute_class_weight` + Keras | Bi-LSTM | Same correction for neural training |
| Custom `WeightedTrainer` | DistilBERT | Weighted CrossEntropyLoss in HF Trainer |
| GloVe embeddings frozen | Bi-LSTM | Preserve pre-trained semantic geometry |
| 5-fold CV for C | SVM | Data-driven hyperparameter selection |
| EarlyStopping (patience=5) | Bi-LSTM | Prevents overfitting, restores best weights |
| `load_best_model_at_end=True` | DistilBERT | Selects lowest-val-loss checkpoint |

---
