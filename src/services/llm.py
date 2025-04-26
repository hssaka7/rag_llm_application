from google import genai
from google.genai import types

class GeminiService:
    def __init__(self, api_key: str):
        """
        Initializes the GeminiService with the provided API key.
        """
        self.client = genai.Client(api_key=api_key)
    


    def generate_content_stream(self,  
                                contents: list,
                                system_instruction = None,
                                model: str = "gemini-2.0-flash",
                                max_output_tokens=1000,
                                temperature=0.1):
        """
        Generates content from the Gemini API using streaming.

        Args:
            model (str): The model identifier (e.g., "gemini-2.0-flash").
            contents (list): A list of content to be processed.

        Returns:
            str: The generated content as a string.
        """

        config = types.GenerateContentConfig(
                    max_output_tokens=max_output_tokens,
                    temperature=temperature,
                    system_instruction = system_instruction,
                )
        
        try:
            # Request content generation from the model
            response = self.client.models.generate_content_stream(
                model=model,
                contents=contents,
                config = config
            )

            # Collect and return the streamed content
            generated_text = ""
            for chunk in response:
                generated_text += chunk.text
                yield generated_text
            # return generated_text

        except Exception as e:
            print(f"An error occurred: {e}")
            return None

    def get_chat(self, model: str = "gemini-2.0-flash",):
        chat = self.client.chats.create(model=model)
        return chat