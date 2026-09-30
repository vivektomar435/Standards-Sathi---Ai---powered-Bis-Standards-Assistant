import json
from typing import Optional
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


DEFAULT_OLLAMA_URL = "http://localhost:11434"
DEFAULT_MODEL = "llama3.2:3b"
DEFAULT_TIMEOUT = 120


class OllamaClient:
    """
    Simple client for communicating with a locally running Ollama server.
    """

    def __init__(
        self,
        base_url: str = DEFAULT_OLLAMA_URL,
        model: str = DEFAULT_MODEL,
        timeout: int = DEFAULT_TIMEOUT,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout

    def _request(self, endpoint: str, payload: dict) -> dict:
        """
        Send a POST request to Ollama and return the JSON response.
        """

        url = f"{self.base_url}{endpoint}"

        data = json.dumps(payload).encode("utf-8")

        request = Request(
            url,
            data=data,
            headers={
                "Content-Type": "application/json",
            },
            method="POST",
        )

        try:
            with urlopen(request, timeout=self.timeout) as response:
                response_data = response.read().decode("utf-8")

        except HTTPError as exc:
            error_body = exc.read().decode("utf-8", errors="replace")

            raise RuntimeError(
                f"Ollama returned HTTP {exc.code}: {error_body}"
            ) from exc

        except URLError as exc:
            raise RuntimeError(
                "Could not connect to Ollama. "
                "Make sure Ollama is running at "
                f"{self.base_url}."
            ) from exc

        except TimeoutError as exc:
            raise RuntimeError(
                f"Ollama request timed out after {self.timeout} seconds."
            ) from exc

        try:
            return json.loads(response_data)

        except json.JSONDecodeError as exc:
            raise RuntimeError(
                "Ollama returned an invalid JSON response."
            ) from exc

    def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        temperature: float = 0.0,
    ) -> str:
        """
        Generate a response from the configured Ollama model.
        """

        if not prompt or not prompt.strip():
            raise ValueError("Prompt cannot be empty.")

        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
            },
        }

        if system:
            payload["system"] = system

        response = self._request(
            endpoint="/api/generate",
            payload=payload,
        )

        return response.get("response", "").strip()

    def chat(
        self,
        messages: list[dict],
        temperature: float = 0.0,
    ) -> str:
        """
        Send a list of chat messages to Ollama.
        """

        if not messages:
            raise ValueError("Messages cannot be empty.")

        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature,
            },
        }

        response = self._request(
            endpoint="/api/chat",
            payload=payload,
        )

        message = response.get("message", {})

        return message.get("content", "").strip()

    def is_available(self) -> bool:
        """
        Check whether the Ollama server is reachable.
        """

        url = f"{self.base_url}/api/tags"

        request = Request(
            url,
            method="GET",
        )

        try:
            with urlopen(request, timeout=10):
                return True

        except (HTTPError, URLError, TimeoutError):
            return False

    def list_models(self) -> list[str]:
        """
        Return the names of models available in Ollama.
        """

        url = f"{self.base_url}/api/tags"

        request = Request(
            url,
            method="GET",
        )

        try:
            with urlopen(request, timeout=10) as response:
                response_data = response.read().decode("utf-8")

        except HTTPError as exc:
            error_body = exc.read().decode("utf-8", errors="replace")

            raise RuntimeError(
                f"Ollama returned HTTP {exc.code}: {error_body}"
            ) from exc

        except URLError as exc:
            raise RuntimeError(
                "Could not connect to Ollama. "
                f"Make sure Ollama is running at {self.base_url}."
            ) from exc

        try:
            data = json.loads(response_data)

        except json.JSONDecodeError as exc:
            raise RuntimeError(
                "Ollama returned invalid JSON while listing models."
            ) from exc

        models = data.get("models", [])

        return [
            model.get("name")
            for model in models
            if model.get("name")
        ]

    def model_available(self) -> bool:
        """
        Check whether the configured model exists locally.
        """

        return self.model in self.list_models()