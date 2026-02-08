# Text Classification with Multiple Embeddings - Group 6

## 📋 Project Overview

Comparative analysis of text classification performance using different model architectures and word embedding techniques. This project implements and evaluates multiple embedding-model combinations for spam detection.

**Course:** Machine Learning Techniques I  
**Date:** February 2026  
**Dataset:** Spam Detection (SMS Spam Collection)

## 👥 Team Members

| Name                | Model               | Embeddings              | Contact           |
| ------------------- | ------------------- | ----------------------- | ----------------- |
| Mitali Bela         | Logistic Regression | TF-IDF, Skip-gram, CBOW | m.bela@alustudent.com |
| Charlotte Kariza    | RNN                 | TF-IDF, Skip-gram, CBOW | c.kariza@alustudent.com |
| Ntezirizaza Erneste | LSTM                | TF-IDF, Skip-gram, CBOW | e.nteziriza@alustudent.com |
| Orpheus Manga       | GRU                 | TF-IDF, Skip-gram, CBOW | o.manga@alustudent.com |

## 🎯 Objectives

1. Implement and evaluate 4 different model architectures
2. Compare performance across multiple word embedding techniques:
   - TF-IDF (Term Frequency-Inverse Document Frequency)
   - Skip-gram (Word2Vec)
   - CBOW (Word2Vec)
3. Produce comprehensive comparative analysis with academic rigor
4. Document findings in research-style report with proper citations

## 📁 Project Structure

```
Group6-Text-Classification-Assignment/
│
├── README.md                              # This file
├── requirements.txt                        # Python dependencies
├── .gitignore                             # Git ignore rules
│
├── data/
│   ├── raw/                               # Original dataset (don't modify)
│   │   └── spam.csv
│   ├── processed/                         # Preprocessed data
│   └── README.md                          # Dataset documentation
│
├── notebooks/
│   ├── 01_data_exploration.ipynb          # SHARED - EDA (4+ visualizations)
│   ├── 02_member1_logistic_regression.ipynb
│   ├── 03_member2_rnn.ipynb
│   ├── 04_member3_lstm.ipynb
│   └── 05_member4_gru.ipynb
│
├── src/
│   ├── __init__.py
│   ├── preprocessing.py                   # SHARED - Text cleaning
│   ├── embeddings.py                      # SHARED - All embedding methods
│   ├── utils.py                           # SHARED - Helper functions
│   └── evaluation.py                      # SHARED - Metrics & visualizations
│
├── results/
│   ├── figures/                           # All plots/visualizations
│   ├── tables/                            # CSV result tables
│   └── comparison_results.csv             # Combined team results
│
└── docs/
    ├── BSE Group Assignments _ Task Sheet_Machine Learning Techniques I_C1_#_Group 6#].xlsx
```

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- pip or conda package manager
- Git

### Installation

1. **Clone the repository**

```bash
git clone https://github.com/ernesteNtezirizaza/Group6-Text-Classification-Assignment/
cd Group6-Text-Classification-Assignment
```

2. **Create virtual environment (recommended)**

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

4. **Download required embedding models** (if using pre-trained)

```python
# Run in Python
import nltk
import gensim.downloader as api

nltk.download('punkt')
nltk.download('stopwords')

# Download GloVe (optional)
# glove_model = api.load('glove-wiki-gigaword-100')
```

## 💻 Usage

### Step 1: Data Exploration (TEAM TASK)

```bash
jupyter notebook notebooks/01_data_exploration.ipynb
```

**Requirements:**

- 4+ visualizations (class balance, text length distribution, word clouds, vocabulary analysis)
- Statistical analysis
- Preprocessing strategy definition

### Step 2: Individual Model Development

Each member works on their assigned notebook:

```bash
jupyter notebook notebooks/02_member1_logistic_regression.ipynb
# OR
jupyter notebook notebooks/03_member2_rnn.ipynb
# etc.
```

**Each notebook must:**

- Implement the assigned model
- Train with at least 3 different embeddings
- Perform hyperparameter tuning
- Generate evaluation metrics (accuracy, F1, confusion matrix)
- Save results to `results/tables/`

## 📊 Evaluation Metrics

All models will be evaluated using:

- **Accuracy**: Overall classification accuracy
- **Precision**: Per-class precision
- **Recall**: Per-class recall
- **F1-Score**: Harmonic mean of precision and recall
- **Confusion Matrix**: Visual representation of predictions
- **Training Time**: Computational efficiency

## 🔧 Shared Modules

### `src/preprocessing.py`

- Text cleaning (remove punctuation, lowercase, etc.)
- Tokenization
- Stop word removal
- Stemming/Lemmatization

### `src/embeddings.py`

- TF-IDF vectorization
- Word2Vec (Skip-gram & CBOW) training
- GloVe loading and processing
- FastText training
- Embedding adaptation for different models

### `src/utils.py`

- Data loading helpers
- Train/test split utilities
- Model saving/loading
- Logging utilities

## 📝 Contribution Guidelines

### Version Control Workflow

1. **Pull latest changes** before starting work

   ```bash
   git pull origin main
   ```

2. **Create feature branch** for your work

   ```bash
   git checkout -b member1-logistic-regression
   ```

3. **Commit regularly** with clear messages

   ```bash
   git add .
   git commit -m "Add TF-IDF implementation for Logistic Regression"
   ```

4. **Push to remote**

   ```bash
   git push origin member1-logistic-regression
   ```

5. **Create Pull Request** for review

### Code Style

- Follow PEP 8 for Python code
- Add docstrings to all functions
- Comment complex logic
- Use meaningful variable names

### Documentation Requirements

- Update `docs/BSE Group Assignments _ Task Sheet_Machine Learning Techniques I_C1_#_Group 6#].xlsx` after each work session
- Document all experiments in notebooks
- Add citations for techniques used
- Keep README updated

## 📚 Key References

### Word Embeddings

- Mikolov et al. (2013). "Efficient Estimation of Word Representations in Vector Space" (Word2Vec)
- Pennington et al. (2014). "GloVe: Global Vectors for Word Representation"
- Bojanowski et al. (2017). "Enriching Word Vectors with Subword Information" (FastText)

## 📋 Deliverables Checklist

- [ ] **GitHub Repository**
  - [ ] Clean, well-documented code
  - [ ] Meaningful README
  - [ ] All notebooks functional
  - [ ] Proper .gitignore

- [ ] **PDF Report** (Academic Format)
  - [ ] Introduction & problem statement
  - [ ] Literature review with citations
  - [ ] Methodology (dataset, preprocessing, models, embeddings)
  - [ ] Results (2+ comparison tables, visualizations)
  - [ ] Discussion (analysis, limitations, insights)
  - [ ] Conclusion & future work
  - [ ] References (APA/IEEE format)
  - [ ] Contribution tracker included
  - [ ] Link to GitHub repo

- [ ] **Experiments**
  - [ ] Each member: 1 model × 3+ embeddings
  - [ ] Hyperparameter tuning documented
  - [ ] All results in `results/` folder

## 🤝 Communication

- **Team Meetings:** Google meet
- **Communication Channel:** WhatsApp
- **Sharing Documents** 

## 📞 Contact

For questions or issues, contact:

- Team Lead: Ntezirizaza Erneste - e.nteziriza@alustudent.com
- Course Instructor: Samiratu - sntohsi@alueducation.com

---

**Repository:** [\[GitHub URL\]](https://github.com/ernesteNtezirizaza/Group6-Text-Classification-Assignment)  
