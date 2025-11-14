import nltk
from collections import Counter
import re

# Download NLTK tokenizer data (used for splitting text into words)
nltk.download('punkt')

class Vocabulary:
    def __init__(self, freq_threshold):
        self.freq_threshold = freq_threshold

        # Initialize special tokens
        self.itos = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"}
        self.stoi = {v: k for k, v in self.itos.items()}

    def __len__(self):
        return len(self.itos)

    @staticmethod
    def tokenizer_eng(text):
        # Clean text: remove non-alphanumeric characters and lowercase
        text = re.sub(r"[^a-zA-Z0-9]+", " ", text.lower())
        # Tokenize into words
        return nltk.tokenize.word_tokenize(text)

    def build_vocabulary(self, sentence_list):
        frequencies = Counter()
        idx = 4  # Start after special tokens

        # Count word frequencies
        for sentence in sentence_list:
            tokens = self.tokenizer_eng(sentence)
            frequencies.update(tokens)

        # Add words to vocabulary if they meet the frequency threshold
        for word, freq in frequencies.items():
            if freq >= self.freq_threshold:
                self.stoi[word] = idx
                self.itos[idx] = word
                idx += 1

    def numericalize(self, text):
        # Convert text into a list of word indices
        tokenized_text = self.tokenizer_eng(text)
        return [
            self.stoi.get(token, self.stoi["<UNK>"])  # Use <UNK> if word is not found
            for token in tokenized_text
        ]
