"""
Word embedding utilities for text classification project.

This module provides implementations for:
- TF-IDF
- Word2Vec (Skip-gram and CBOW)
- GloVe
- FastText

All team members should use these functions for consistency.
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec, FastText
from gensim.models.keyedvectors import KeyedVectors
import gensim.downloader as api


class TFIDFEmbedding:
    """
    TF-IDF vectorization for text classification.
    
    Usage:
        tfidf = TFIDFEmbedding(max_features=5000)
        X_train = tfidf.fit_transform(train_texts)
        X_test = tfidf.transform(test_texts)
    """
    
    def __init__(
        self,
        max_features: int = 5000,
        min_df: int = 2,
        max_df: float = 0.95,
        ngram_range: Tuple[int, int] = (1, 2)
    ):
        """
        Initialize TF-IDF vectorizer.
        
        Args:
            max_features: Maximum number of features
            min_df: Minimum document frequency
            max_df: Maximum document frequency
            ngram_range: Range of n-grams to consider
        """
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            min_df=min_df,
            max_df=max_df,
            ngram_range=ngram_range
        )
        
    def fit_transform(self, texts: List[str]) -> np.ndarray:
        """Fit vectorizer and transform texts."""
        return self.vectorizer.fit_transform(texts).toarray()
    
    def transform(self, texts: List[str]) -> np.ndarray:
        """Transform texts using fitted vectorizer."""
        return self.vectorizer.transform(texts).toarray()
    
    def get_feature_names(self) -> List[str]:
        """Get feature names (vocabulary)."""
        return self.vectorizer.get_feature_names_out().tolist()
    
    def get_vocab_size(self) -> int:
        """Get vocabulary size."""
        return len(self.vectorizer.vocabulary_)


class Word2VecEmbedding:
    """
    Word2Vec embeddings (Skip-gram or CBOW).
    
    Usage:
        w2v = Word2VecEmbedding(sg=1, vector_size=100)  # Skip-gram
        w2v.train(tokenized_texts)
        X_train = w2v.transform(tokenized_texts)
    """
    
    def __init__(
        self,
        vector_size: int = 100,
        window: int = 5,
        min_count: int = 2,
        sg: int = 1,  # 1 for Skip-gram, 0 for CBOW
        workers: int = 4,
        epochs: int = 10
    ):
        """
        Initialize Word2Vec model.
        
        Args:
            vector_size: Dimensionality of word vectors
            window: Context window size
            min_count: Minimum word frequency
            sg: 1 for Skip-gram, 0 for CBOW
            workers: Number of worker threads
            epochs: Number of training epochs
        """
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.sg = sg
        self.workers = workers
        self.epochs = epochs
        self.model = None
        
    def train(self, tokenized_texts: List[List[str]]):
        """
        Train Word2Vec model on tokenized texts.
        
        Args:
            tokenized_texts: List of tokenized documents
        """
        self.model = Word2Vec(
            sentences=tokenized_texts,
            vector_size=self.vector_size,
            window=self.window,
            min_count=self.min_count,
            sg=self.sg,
            workers=self.workers,
            epochs=self.epochs
        )
        
    def get_word_vector(self, word: str) -> Optional[np.ndarray]:
        """Get vector for a single word."""
        if self.model and word in self.model.wv:
            return self.model.wv[word]
        return None
    
    def document_vector(self, tokens: List[str], method: str = 'mean') -> np.ndarray:
        """
        Convert document to vector by aggregating word vectors.
        
        Args:
            tokens: List of word tokens
            method: 'mean', 'max', or 'sum'
            
        Returns:
            Document vector
        """
        vectors = [self.get_word_vector(word) for word in tokens]
        vectors = [v for v in vectors if v is not None]
        
        if not vectors:
            return np.zeros(self.vector_size)
        
        if method == 'mean':
            return np.mean(vectors, axis=0)
        elif method == 'max':
            return np.max(vectors, axis=0)
        elif method == 'sum':
            return np.sum(vectors, axis=0)
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def transform(self, tokenized_texts: List[List[str]], method: str = 'mean') -> np.ndarray:
        """
        Transform tokenized texts to document vectors.
        
        Args:
            tokenized_texts: List of tokenized documents
            method: Aggregation method ('mean', 'max', 'sum')
            
        Returns:
            Array of document vectors
        """
        return np.array([
            self.document_vector(tokens, method) 
            for tokens in tokenized_texts
        ])
    
    def save(self, filepath: str):
        """Save trained model."""
        if self.model:
            self.model.save(filepath)
    
    def load(self, filepath: str):
        """Load trained model."""
        self.model = Word2Vec.load(filepath)


class GloVeEmbedding:
    """
    GloVe embeddings (using pre-trained or custom trained).
    
    Usage:
        glove = GloVeEmbedding()
        glove.load_pretrained('glove-wiki-gigaword-100')
        X_train = glove.transform(tokenized_texts)
    """
    
    def __init__(self, vector_size: int = 100):
        """
        Initialize GloVe embedding handler.
        
        Args:
            vector_size: Dimensionality of vectors
        """
        self.vector_size = vector_size
        self.model = None
    
    def load_pretrained(self, model_name: str = 'glove-wiki-gigaword-100'):
        """
        Load pre-trained GloVe vectors from gensim.
        
        Args:
            model_name: Name of pre-trained model
                Options: 'glove-wiki-gigaword-50/100/200/300'
                         'glove-twitter-25/50/100/200'
        """
        print(f"Downloading {model_name}... This may take a while.")
        self.model = api.load(model_name)
        self.vector_size = self.model.vector_size
    
    def load_from_file(self, filepath: str):
        """
        Load GloVe vectors from text file.
        
        Args:
            filepath: Path to GloVe .txt file
        """
        self.model = KeyedVectors.load_word2vec_format(
            filepath, 
            binary=False, 
            no_header=True
        )
        self.vector_size = self.model.vector_size
    
    def get_word_vector(self, word: str) -> Optional[np.ndarray]:
        """Get vector for a single word."""
        if self.model and word in self.model:
            return self.model[word]
        return None
    
    def document_vector(self, tokens: List[str], method: str = 'mean') -> np.ndarray:
        """Convert document to vector by aggregating word vectors."""
        vectors = [self.get_word_vector(word) for word in tokens]
        vectors = [v for v in vectors if v is not None]
        
        if not vectors:
            return np.zeros(self.vector_size)
        
        if method == 'mean':
            return np.mean(vectors, axis=0)
        elif method == 'max':
            return np.max(vectors, axis=0)
        elif method == 'sum':
            return np.sum(vectors, axis=0)
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def transform(self, tokenized_texts: List[List[str]], method: str = 'mean') -> np.ndarray:
        """Transform tokenized texts to document vectors."""
        return np.array([
            self.document_vector(tokens, method) 
            for tokens in tokenized_texts
        ])


class FastTextEmbedding:
    """
    FastText embeddings with subword information.
    
    Usage:
        fasttext = FastTextEmbedding(vector_size=100)
        fasttext.train(tokenized_texts)
        X_train = fasttext.transform(tokenized_texts)
    """
    
    def __init__(
        self,
        vector_size: int = 100,
        window: int = 5,
        min_count: int = 2,
        sg: int = 1,  # 1 for Skip-gram, 0 for CBOW
        workers: int = 4,
        epochs: int = 10
    ):
        """Initialize FastText model."""
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.sg = sg
        self.workers = workers
        self.epochs = epochs
        self.model = None
    
    def train(self, tokenized_texts: List[List[str]]):
        """Train FastText model on tokenized texts."""
        self.model = FastText(
            sentences=tokenized_texts,
            vector_size=self.vector_size,
            window=self.window,
            min_count=self.min_count,
            sg=self.sg,
            workers=self.workers,
            epochs=self.epochs
        )
    
    def get_word_vector(self, word: str) -> np.ndarray:
        """
        Get vector for a word (even if not in training vocabulary).
        FastText can generate vectors for OOV words using subword info.
        """
        if self.model:
            return self.model.wv[word]
        return None
    
    def document_vector(self, tokens: List[str], method: str = 'mean') -> np.ndarray:
        """Convert document to vector."""
        vectors = [self.get_word_vector(word) for word in tokens]
        vectors = [v for v in vectors if v is not None]
        
        if not vectors:
            return np.zeros(self.vector_size)
        
        if method == 'mean':
            return np.mean(vectors, axis=0)
        elif method == 'max':
            return np.max(vectors, axis=0)
        elif method == 'sum':
            return np.sum(vectors, axis=0)
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def transform(self, tokenized_texts: List[List[str]], method: str = 'mean') -> np.ndarray:
        """Transform tokenized texts to document vectors."""
        return np.array([
            self.document_vector(tokens, method) 
            for tokens in tokenized_texts
        ])
    
    def save(self, filepath: str):
        """Save trained model."""
        if self.model:
            self.model.save(filepath)
    
    def load(self, filepath: str):
        """Load trained model."""
        self.model = FastText.load(filepath)


if __name__ == "__main__":
    # Example usage
    sample_texts = [
        "this is a sample document",
        "another example text for testing",
        "word embeddings are useful"
    ]
    
    # TF-IDF example
    print("TF-IDF Example:")
    tfidf = TFIDFEmbedding(max_features=100)
    X_tfidf = tfidf.fit_transform(sample_texts)
    print(f"TF-IDF shape: {X_tfidf.shape}")
    
    # Word2Vec example
    print("\nWord2Vec Example:")
    tokenized = [text.split() for text in sample_texts]
    w2v = Word2VecEmbedding(vector_size=50, min_count=1)
    w2v.train(tokenized)
    X_w2v = w2v.transform(tokenized)
    print(f"Word2Vec shape: {X_w2v.shape}")
