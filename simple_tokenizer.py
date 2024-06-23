from collections import defaultdict
import re
import simple_tokenizer

class SimpleTokenizer:
    def __init__(self, num_words=None):
        """
        Initialize the SimpleTokenizer.
        
        Args:
        num_words (int, optional): Maximum number of words to keep.
        """
        # Dictionary to store word-to-index mapping
        self.word_index = defaultdict(lambda: len(self.word_index) + 1)
        self.num_words = num_words
        # Dictionary to store word frequency counts
        self.word_counts = defaultdict(int)

    def tokenize(self, text):
        """
        Tokenize a single text string.
        
        Args:
        text (str): Input text.
        
        Returns:
        list of str: List of tokens.
        """
        # Simple tokenization: lowercase and split on non-word characters
        return re.findall(r'\w+', text.lower())

    def fit_on_texts(self, texts):
        """
        Fit the tokenizer on a list of texts.
        
        Args:
        texts (list of str): List of input texts.
        """
        # Count word frequencies
        for text in texts:
            for word in self.tokenize(text):
                self.word_counts[word] += 1
        
        # Sort words by frequency
        sorted_words = sorted(self.word_counts.items(), key=lambda x: x[1], reverse=True)
        # Limit vocabulary size if num_words is specified
        if self.num_words:
            sorted_words = sorted_words[:self.num_words]
        
        # Reset and rebuild word_index with the most frequent words
        self.word_index.clear()
        for word, _ in sorted_words:
            _ = self.word_index[word]

    def texts_to_sequences(self, texts):
        """
        Convert a list of texts to sequences of token indices.
        
        Args:
        texts (list of str): List of input texts.
        
        Returns:
        list of list of int: List of token index sequences.
        """
        sequences = []
        for text in texts:
            # Convert each text to a sequence of token indices
            sequence = [self.word_index[word] for word in self.tokenize(text) if word in self.word_index]
            sequences.append(sequence)
        return sequences