
from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate

class OllamaService:
    def __init__(self, model: str = "gemma3:1b", temperature: float = 0.1):
        self.llm = OllamaLLM(model=model, temperature=temperature)

    def generate_content_stream(self, prompt: str, system_instruction: str = None):
        """
        Stream content using Ollama LLM with optional system instruction and input variables.
        Args:
            prompt (str): The user prompt (can include template variables).
            system_instruction (str): Optional system prompt.
            input_vars (dict): Variables for the prompt template.
        Yields:
            str: The generated content, token by token.
        """
        if system_instruction:
            chat_prompt = ChatPromptTemplate.from_messages([
                ("system", system_instruction),
                ("human", "{prompt}"),
            ])
        else:
            chat_prompt = ChatPromptTemplate.from_messages([
                ("human", "{prompt}"),
            ])
        chain = chat_prompt | self.llm
       
        # Use stream=True to get a generator of tokens
        for chunk in chain.stream(prompt):
           yield chunk