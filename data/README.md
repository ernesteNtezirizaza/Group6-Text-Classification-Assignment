# SMS Spam Collection Dataset

## Dataset Information

**Name:** SMS Spam Collection  
**Source:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection)  
**Type:** Text Classification (Binary)  
**Domain:** Spam Detection

## Description

The SMS Spam Collection is a set of SMS tagged messages that have been collected for SMS Spam research. It contains one set of SMS messages in English, tagged as either legitimate (ham) or spam.

## Dataset Characteristics

- **Total Messages:** ~5,574
- **Classes:** 
  - Ham (legitimate messages)
  - Spam (spam messages)
- **Format:** CSV file with columns:
  - `v1`: Label (ham/spam)
  - `v2`: Text message content

## Class Distribution

| Class | Count | Percentage |
|-------|-------|------------|
| Ham   | ~4,827| ~86.6%     |
| Spam  | ~747  | ~13.4%     |

**Note:** The dataset is imbalanced with more ham messages than spam.

## Citation

If you use this dataset, please cite:

```
Almeida, T.A., Gómez Hidalgo, J.M., Yamakami, A.
SMS Spam Collection v.1
2012.
Available at: http://www.dt.fee.unicamp.br/~tiago/smsspamcollection/
```

## Preprocessing Notes

### Recommended Preprocessing Steps:
1. Convert text to lowercase
2. Remove URLs and email addresses
3. Remove punctuation
4. Remove stopwords (optional - discuss with team)
5. Tokenization
6. Lemmatization or Stemming

### Embedding-Specific Preprocessing:
- **TF-IDF:** Can work with full cleaned text
- **Word2Vec/GloVe/FastText:** Requires tokenized text
- Consider keeping vs. removing stopwords based on embedding method

## Data Files

- `raw/spam.csv` - Original dataset (DO NOT MODIFY)
- `processed/` - Preprocessed versions will be saved here

## Usage in Project

```python
import pandas as pd
from src.preprocessing import load_spam_data

# Load data
df = load_spam_data('data/raw/spam.csv')

# Basic exploration
print(df.head())
print(df['v1'].value_counts())
```

## Additional Resources

- [Original Dataset Page](http://www.dt.fee.unicamp.br/~tiago/smsspamcollection/)
- [Related Paper](https://www.researchgate.net/publication/228391052_SMS_Spam_Collection_v1)
