"""
Shared preprocessing utilities for text classification project.

This module provides common preprocessing functions used across all models.
All team members should use these functions to ensure consistency.
"""

import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
import pandas as pd
from typing import List, Union

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')


class TextPreprocessor:
    """
    Text preprocessing pipeline for spam classification.
    
    Usage:
        preprocessor = TextPreprocessor(lowercase=True, remove_stopwords=True)
        clean_text = preprocessor.preprocess("This is a sample text!")
    """
    
    def __init__(
        self,
        lowercase: bool = True,
        remove_punctuation: bool = True,
        remove_numbers: bool = False,
        remove_stopwords: bool = True,
        stemming: bool = False,
        lemmatization: bool = False,
        remove_extra_whitespace: bool = True
    ):
        """
        Initialize preprocessor with desired options.
        
        Args:
            lowercase: Convert text to lowercase
            remove_punctuation: Remove punctuation marks
            remove_numbers: Remove numeric characters
            remove_stopwords: Remove common stopwords
            stemming: Apply Porter stemming
            lemmatization: Apply WordNet lemmatization
            remove_extra_whitespace: Remove extra spaces
        """
        self.lowercase = lowercase
        self.remove_punctuation = remove_punctuation
        self.remove_numbers = remove_numbers
        self.remove_stopwords = remove_stopwords
        self.stemming = stemming
        self.lemmatization = lemmatization
        self.remove_extra_whitespace = remove_extra_whitespace
        
        # Initialize tools
        self.stop_words = set(stopwords.words('english'))
        self.stemmer = PorterStemmer() if stemming else None
        self.lemmatizer = WordNetLemmatizer() if lemmatization else None
    
    def clean_text(self, text: str) -> str:
        """
        Apply basic cleaning operations.
        
        Args:
            text: Input text string
            
        Returns:
            Cleaned text string
        """
        if not isinstance(text, str):
            return ""
        
        # Lowercase
        if self.lowercase:
            text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove numbers
        if self.remove_numbers:
            text = re.sub(r'\d+', '', text)
        
        # Remove punctuation
        if self.remove_punctuation:
            text = text.translate(str.maketrans('', '', string.punctuation))
        
        # Remove extra whitespace
        if self.remove_extra_whitespace:
            text = ' '.join(text.split())
        
        return text
    
    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize text into words.
        
        Args:
            text: Input text string
            
        Returns:
            List of tokens
        """
        return word_tokenize(text)
    
    def remove_stop_words(self, tokens: List[str]) -> List[str]:
        """
        Remove stopwords from token list.
        
        Args:
            tokens: List of word tokens
            
        Returns:
            Filtered token list
        """
        return [word for word in tokens if word not in self.stop_words]
    
    def apply_stemming(self, tokens: List[str]) -> List[str]:
        """
        Apply Porter stemming to tokens.
        
        Args:
            tokens: List of word tokens
            
        Returns:
            Stemmed token list
        """
        return [self.stemmer.stem(word) for word in tokens]
    
    def apply_lemmatization(self, tokens: List[str]) -> List[str]:
        """
        Apply WordNet lemmatization to tokens.
        
        Args:
            tokens: List of word tokens
            
        Returns:
            Lemmatized token list
        """
        return [self.lemmatizer.lemmatize(word) for word in tokens]
    
    def preprocess(self, text: str, return_tokens: bool = False) -> Union[str, List[str]]:
        """
        Apply full preprocessing pipeline.
        
        Args:
            text: Input text string
            return_tokens: If True, return list of tokens; if False, return string
            
        Returns:
            Preprocessed text as string or token list
        """
        # Clean text
        text = self.clean_text(text)
        
        # Tokenize
        tokens = self.tokenize(text)
        
        # Remove stopwords
        if self.remove_stopwords:
            tokens = self.remove_stop_words(tokens)
        
        # Stemming
        if self.stemming:
            tokens = self.apply_stemming(tokens)
        
        # Lemmatization
        if self.lemmatization:
            tokens = self.apply_lemmatization(tokens)
        
        if return_tokens:
            return tokens
        else:
            return ' '.join(tokens)
    
    def preprocess_dataframe(
        self, 
        df: pd.DataFrame, 
        text_column: str, 
        new_column: str = 'cleaned_text'
    ) -> pd.DataFrame:
        """
        Apply preprocessing to entire DataFrame column.
        
        Args:
            df: Input DataFrame
            text_column: Name of column containing text
            new_column: Name for new column with cleaned text
            
        Returns:
            DataFrame with new preprocessed column
        """
        df = df.copy()
        df[new_column] = df[text_column].apply(self.preprocess)
        return df


def load_spam_data(filepath: str = 'data/raw/spam.csv') -> pd.DataFrame:
    """
    Load spam dataset with proper encoding handling.
    
    Args:
        filepath: Path to spam.csv file
        
    Returns:
        DataFrame with text and label columns
    """
    # Try different encodings
    encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']
    
    for encoding in encodings:
        try:
            df = pd.read_csv(filepath, encoding=encoding)
            print(f"Successfully loaded data with {encoding} encoding")
            return df
        except UnicodeDecodeError:
            continue
    
    raise ValueError(f"Could not load {filepath} with any standard encoding")


def get_text_statistics(texts: pd.Series) -> dict:
    """
    Calculate basic statistics about text data.
    
    Args:
        texts: Series of text strings
        
    Returns:
        Dictionary with statistics
    """
    stats = {
        'total_documents': len(texts),
        'avg_length': texts.str.len().mean(),
        'median_length': texts.str.len().median(),
        'max_length': texts.str.len().max(),
        'min_length': texts.str.len().min(),
        'avg_word_count': texts.str.split().str.len().mean(),
    }
    return stats


if __name__ == "__main__":
    # Example usage
    preprocessor = TextPreprocessor(
        lowercase=True,
        remove_stopwords=True,
        lemmatization=True
    )
    
    sample_text = "This is a SAMPLE text with some Numbers 123 and URLs http://example.com!"
    cleaned = preprocessor.preprocess(sample_text)
    print(f"Original: {sample_text}")
    print(f"Cleaned: {cleaned}")
