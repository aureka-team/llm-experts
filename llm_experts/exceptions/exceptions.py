class LLMResponseError(Exception):
    def __init__(self, response_text: str):
        message = f"Malformed LLM response: {response_text}"
        super().__init__(message)
