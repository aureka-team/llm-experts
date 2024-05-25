class LLMResponseError(Exception):
    def __init__(self, response_text: str):
        message = f"Malformed LLM response: {response_text}"
        super().__init__(message)


class ParameterDependencyException(Exception):
    def __init__(self, dependent_param: str, required_param: str):
        message = (
            f"Parameter `{dependent_param}` requires "
            f"parameter `{required_param}` to be set."
        )

        super().__init__(message)
