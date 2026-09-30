from llm.ollama import OllamaClient


def main():
    print("=" * 60)
    print("OLLAMA CONNECTION TEST")
    print("=" * 60)

    client = OllamaClient(
        base_url="http://localhost:11434",
        model="llama3.2:3b",
    )

    print("\nChecking Ollama server...")

    if not client.is_available():
        print("FAIL: Ollama server is not available.")
        print(
            "Make sure Ollama is running and available at "
            "http://localhost:11434"
        )
        return

    print("PASS: Ollama server is available.")

    print("\nChecking installed models...")

    try:
        models = client.list_models()

    except RuntimeError as exc:
        print(f"FAIL: Could not list models.")
        print(exc)
        return

    if not models:
        print("FAIL: No Ollama models found.")
        return

    print("Available models:")

    for model in models:
        print(f"  - {model}")

    print("\nChecking configured model...")

    if not client.model_available():
        print(
            f"FAIL: Model '{client.model}' is not installed."
        )
        print(
            f"Run: ollama pull {client.model}"
        )
        return

    print(
        f"PASS: Model '{client.model}' is available."
    )

    print("\nTesting generation...")

    prompt = (
        "Answer this question in one short sentence: "
        "What is an Indian Standard?"
    )

    try:
        response = client.generate(
            prompt=prompt,
            temperature=0.0,
        )

    except RuntimeError as exc:
        print("FAIL: Generation failed.")
        print(exc)
        return

    if not response:
        print("FAIL: Ollama returned an empty response.")
        return

    print("PASS: Generation successful.")

    print("\nModel response:")
    print("-" * 60)
    print(response)
    print("-" * 60)

    print("\nOllama test complete.")


if __name__ == "__main__":
    main()