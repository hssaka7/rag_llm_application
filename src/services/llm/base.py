from abc import ABC, abstractmethod


# Abstract base class for LLM service
class LLMService(ABC):
    @abstractmethod
    def generate_content_stream(self, prompt: str, system_instruction: str = None):
        pass
