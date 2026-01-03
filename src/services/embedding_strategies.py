from abc import ABC, abstractmethod
from typing import List, Any
import unicodedata

class EmbeddingStrategy(ABC):
    @abstractmethod
    def encode(self, texts: List[str]) -> List[Any]:
        pass

    @abstractmethod
    def encode_single(self, text: str) -> Any:
        pass

class SentenceTransformerEmbedding(EmbeddingStrategy):
    def __init__(self, model_name: str = "BAAI/bge-m3"):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)

    def encode(self, texts: List[str]) -> List[Any]:
        texts = [unicodedata.normalize("NFKC", t) for t in texts]
        return self.model.encode(
            texts,
            normalize_embeddings=True,
            batch_size=16,
            show_progress_bar=False
        ).tolist()

    def encode_single(self, text: str) -> Any:
        text = unicodedata.normalize("NFKC", text)
        return self.model.encode(text, normalize_embeddings=True).tolist()
