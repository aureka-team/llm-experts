# LLM Experts

This library builds upon LangChain to provide a simple interface for creating and interacting with LLM experts or agents.
It includes features such as structured output validation and correction, caching functionality,
and seamless integration of memory or history into the experts.

---

## Features

-   Ensures consistent and reliable outputs with structured validation and correction.
-   Reuses previous results using Redis-based caching to avoid redundant computations.
-   Supports seamless chat history integration via MongoDB.
-   Compatible with OpenAI API and LLama (via Ollama).
-   Enables concurrent processing of multiple requests for faster results.

---

## Setup example with `uv`

1. Install `uv` by following the [installation guide](https://docs.astral.sh/uv/getting-started/installation/).
2. Install `llm-experts` using `uv`:

    ```bash
    uv python install 3.12
    uv venv
    uv pip install git+https://git@github.com/aureka-team/llm-experts.git
    ```

---

## Usage Examples

### Generic Assistant Expert with Chat History

```python
import uuid

from llm_experts.experts import GenericAssistant, GenericAssistantInput
from llm_experts.utils.chat_history import get_mongodb_chat_history

# Initialize MongoDB chat history to persist assistant interactions
mongodb_chat_history = get_mongodb_chat_history(
    session_id=uuid.uuid4().hex,
    collection_name="generic-assistant",
)

# Create a Generic Assistant with chat history
generic_assistant = GenericAssistant(
    mongodb_chat_history=mongodb_chat_history,
)

# First query
query_1 = "Hello, I'm from Argentina."
output_1 = generic_assistant.generate(
    expert_input=GenericAssistantInput(
        user_query=query_1,
    )
)
print(output_1)
# Output: GenericAssistantOutput(assistant_response="Hello! It's great to hear from someone in Argentina. How can I assist you today?")

# Follow-up query
query_2 = "Do you know my country of origin?"
output_2 = generic_assistant.generate(
    expert_input=GenericAssistantInput(
        user_query=query_2,
    )
)
print(output_2)
# Output: GenericAssistantOutput(assistant_response="Yes, you mentioned that you are from Argentina.")
```

### Language Translator Expert with Cache and Async Batch Generation

```python
from common.cache import RedisCache  # From the open-source `common` library by aureka-team
from llm_experts.experts import (
    LanguageTranslatorExpert,
    LanguageTranslatorInput,
)

# Initialize Redis cache to store and reuse translations
redis_cache = RedisCache()

# Create a Language Translator expert with caching
language_translator = LanguageTranslatorExpert(
    cache=redis_cache,
)

# Prepare batch inputs
n_inputs = 5
template_text = "This is example number: {number}"
expert_inputs = [
    LanguageTranslatorInput(
        source_text=template_text.format(number=idx),
        source_language="English",
        target_language="Spanish",
    )
    for idx in range(1, n_inputs + 1)
]

# Perform asynchronous batch generation
expert_outputs = await language_translator.batch_generate(
    expert_inputs=expert_inputs
)
```

---

## Creating a New Expert

To create a new expert, refer to the following resources:

-   [Example Experts](llm_experts/experts/)
-   [Corresponding Prompts](llm_experts/conf/experts/)
-   [LLMExpert Interface](llm_experts/meta/interfaces/llm_expert.py)

All experts follow the LLMExpert interface.

---

# Contributing

Contributions are welcome! Please submit a pull request or open an issue if you encounter any bugs or have suggestions for new features.

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
