import openai
import sys

class LLM_Callable:
    def __init__(self, base_url, api_key, model_name):
        self.base_url = base_url
        self.api_key = api_key
        self.model_name = model_name
        self.client = openai.Client(
            base_url=self.base_url,
            api_key=self.api_key,
        )

    def invoke(self, inputs, timeout: int = 300):
        """Non-streaming call with timeout. Raises RuntimeError on failure."""
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": inputs}],
                timeout=timeout,
            )
            content = response.choices[0].message.content
            return content if content is not None else ""
        except Exception as e:
            raise RuntimeError(f"LLM invoke failed: {type(e).__name__}: {e}") from e

    def invoke_stream(self, inputs, timeout: int = 600, print_progress: bool = True):
        """Streaming call — collects tokens and prints them as they arrive.
        Returns the full response text. Raises RuntimeError on failure."""
        try:
            stream = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": inputs}],
                stream=True,
                timeout=timeout,
            )
            chunks = []
            char_count = 0
            for chunk in stream:
                delta = chunk.choices[0].delta if chunk.choices else None
                token = (delta.content or "") if delta else ""
                if token:
                    chunks.append(token)
                    char_count += len(token)
                    if print_progress:
                        print(token, end="", flush=True)
            if print_progress and chunks:
                print()  # newline after streaming
                print(f"[llm_stream] total chars received: {char_count}")
            return "".join(chunks)
        except Exception as e:
            raise RuntimeError(f"LLM invoke_stream failed: {type(e).__name__}: {e}") from e

