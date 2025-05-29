import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import TextVectorization


class NusaXLoader:
    def __init__(
        self,
        data_dir=None,
        batch_size=32,
        max_sequence_length=100,
        vocab_size=10000,
    ):
        # Set default data_dir to an absolute path if not provided
        if data_dir is None:
            # Get the project root directory (assuming the module is in src/models/src/models/lstm)
            current_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.abspath(os.path.join(current_dir, "../../../../../"))
            data_dir = os.path.join(project_root, "src/datasets/nusax/")

        self.data_dir = data_dir
        self.batch_size = batch_size
        self.max_sequence_length = max_sequence_length
        self.vocab_size = vocab_size
        self.vectorizer = None
        self.vocabulary = None
        self.num_classes = (
            3  # NusaX-Sentiment has 3 classes (negative, neutral, positive)
        )

    def load_data(self, split="train"):
        """Load data from CSV file and return DataFrame"""
        filepath = os.path.join(self.data_dir, f"{split}.csv")
        return pd.read_csv(filepath)

    def _create_vectorizer(self, texts):
        """Create and fit the text vectorizer"""
        self.vectorizer = TextVectorization(
            max_tokens=self.vocab_size,
            output_mode="int",
            output_sequence_length=self.max_sequence_length,
        )
        self.vectorizer.adapt(texts)
        self.vocabulary = self.vectorizer.get_vocabulary()
        return self.vectorizer

    def preprocess_data(self, df):
        """Preprocess the data and return texts and labels"""
        texts = df["text"].values
        # Map labels to integer indices if they are not already
        if df["label"].dtype == "object":
            label_map = {"negative": 0, "neutral": 1, "positive": 2}
            labels = df["label"].map(label_map).values
        else:
            labels = df["label"].values
        return texts, labels

    def get_dataset(self, split="train"):
        """Get TensorFlow dataset for the specified split"""
        df = self.load_data(split)
        texts, labels = self.preprocess_data(df)

        # Create vectorizer if it doesn't exist
        if self.vectorizer is None and split == "train":
            self._create_vectorizer(texts)

        # Vectorize the texts
        if self.vectorizer is None:
            raise ValueError("Vectorizer is not initialized. Load train data first.")

        # Create dataset and batch it
        dataset = tf.data.Dataset.from_tensor_slices((texts, labels))

        # Define preprocessing function that vectorizes text
        def vectorize_text(text, label):
            return self.vectorizer(text), label

        # Apply vectorization and batching
        dataset = dataset.map(vectorize_text)
        dataset = dataset.batch(self.batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)

        return dataset

    def get_vectorized_data(self, split="train"):
        """Get vectorized texts and labels as numpy arrays"""
        df = self.load_data(split)
        texts, labels = self.preprocess_data(df)

        # Create vectorizer if it doesn't exist
        if self.vectorizer is None and split == "train":
            self._create_vectorizer(texts)

        # Vectorize the texts
        if self.vectorizer is None:
            raise ValueError("Vectorizer is not initialized. Load train data first.")

        vectorized_texts = self.vectorizer(texts).numpy()

        return vectorized_texts, labels

    def get_vocabulary(self):
        """Get the vocabulary of the vectorizer"""
        if self.vocabulary is None:
            raise ValueError("Vocabulary is not initialized. Load train data first.")
        return self.vocabulary
