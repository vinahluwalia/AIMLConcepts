import simple_tokenizer

class EmbeddingModel(object):
    def __init__(self, vocab_size, embedding_dim):
        """
        Initialize the EmbeddingModel.
        
        Args:
        vocab_size (int): Maximum size of the vocabulary.
        embedding_dim (int): Dimension of the embeddings.
        """
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        # Initialize the tokenizer with the specified vocabulary size
        self.tokenizer = simple_tokenizer.SimpleTokenizer(num_words=self.vocab_size)

    def tokenize_text_corpus(self, texts):
        """
        Convert a list of text strings into word sequences.
        
        Args:
        texts (list of str): List of input texts.
        
        Returns:
        list of list of int: Tokenized sequences.
        """
        # Fit the tokenizer on the input texts
        self.tokenizer.fit_on_texts(texts)
        # Convert the texts into sequences of token indices
        sequences = self.tokenizer.texts_to_sequences(texts)
        return sequences



# Usage example
model = EmbeddingModel(vocab_size=1000, embedding_dim=100)
texts = ["This is a sample text", "Another example text"]
sequences = model.tokenize_text_corpus(texts)
print(sequences)

# Example usage with different embedding sizes
texts = [
    "The quick brown fox jumps over the lazy dog",
    "A journey of a thousand miles begins with a single step",
    "To be or not to be, that is the question"
]
print("Input texts:")
for text in texts:
    print(text)

print("\nTokenized sequences:")

# Example 1: Small embedding size
model_small = EmbeddingModel(vocab_size=100, embedding_dim=10)
sequences_small = model_small.tokenize_text_corpus(texts)
print("Small embedding model (dim=10):")
print("Sequences:", sequences_small)
print("Vocabulary size:", len(model_small.tokenizer.word_index))

# Example 2: Medium embedding size
model_medium = EmbeddingModel(vocab_size=200, embedding_dim=50)
sequences_medium = model_medium.tokenize_text_corpus(texts)
print("\nMedium embedding model (dim=50):")
print("Sequences:", sequences_medium)
print("Vocabulary size:", len(model_medium.tokenizer.word_index))

# Example 3: Large embedding size
model_large = EmbeddingModel(vocab_size=300, embedding_dim=200)
sequences_large = model_large.tokenize_text_corpus(texts)
print("\nLarge embedding model (dim=200):")
print("Sequences:", sequences_large)
print("Vocabulary size:", len(model_large.tokenizer.word_index))