"""Text embedding generation using sentence-transformers."""

import logging
from typing import List, Union

import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class EmbeddingGenerator:
    """Generate embeddings using sentence-transformers (CPU-based)."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """Initialize the embedding model.
        
        Args:
            model_name: HuggingFace model name (default: all-MiniLM-L6-v2, 384-dim)
        """
        self.model_name = model_name
        self.model = None
        self.dimension = 384  # all-MiniLM-L6-v2 produces 384-dim embeddings

    def load(self):
        """Load the model into memory."""
        if self.model is None:
            logger.info(f"Loading embedding model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            logger.info(f"Model loaded. Embedding dimension: {self.dimension}")

    def embed(self, text: Union[str, List[str]]) -> Union[List[float], List[List[float]]]:
        """Generate embeddings for text or list of texts.
        
        Args:
            text: Single text string or list of strings
            
        Returns:
            Single embedding vector or list of embedding vectors
        """
        if self.model is None:
            self.load()

        # Handle single string
        if isinstance(text, str):
            embedding = self.model.encode(text, convert_to_numpy=True)
            return embedding.tolist()

        # Handle list of strings
        embeddings = self.model.encode(text, convert_to_numpy=True, show_progress_bar=True)
        return embeddings.tolist()

    def embed_batch(
        self,
        texts: List[str],
        batch_size: int = 32,
    ) -> List[List[float]]:
        """Generate embeddings for a large batch of texts.
        
        Args:
            texts: List of text strings
            batch_size: Number of texts to process at once
            
        Returns:
            List of embedding vectors
        """
        if self.model is None:
            self.load()

        logger.info(f"Generating embeddings for {len(texts)} texts...")
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            show_progress_bar=True,
        )
        logger.info(f"Generated {len(embeddings)} embeddings")
        
        return embeddings.tolist()


# Global instance for reuse
_embedding_generator = None


def get_embedding_generator() -> EmbeddingGenerator:
    """Get or create global embedding generator instance."""
    global _embedding_generator
    if _embedding_generator is None:
        _embedding_generator = EmbeddingGenerator()
        _embedding_generator.load()
    return _embedding_generator


if __name__ == "__main__":
    # Test the embedding generator
    logging.basicConfig(level=logging.INFO)
    
    gen = EmbeddingGenerator()
    gen.load()
    
    # Test single embedding
    text = "I have 5 years of experience in hospitality management"
    embedding = gen.embed(text)
    print(f"Single embedding shape: {len(embedding)}")
    
    # Test batch embedding
    texts = [
        "Software engineer with Python experience",
        "Healthcare professional seeking new opportunities",
        "Administrative support specialist",
    ]
    embeddings = gen.embed_batch(texts)
    print(f"Batch embeddings shape: {len(embeddings)} x {len(embeddings[0])}")
