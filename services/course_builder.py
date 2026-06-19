from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import TypeVar

from dotenv import load_dotenv
from pydantic import BaseModel, ValidationError

try:
    from openai import APIError, APITimeoutError, AzureOpenAI, OpenAIError
except ImportError:  # pragma: no cover - handled at runtime in Streamlit
    APIError = Exception
    APITimeoutError = Exception
    AzureOpenAI = None
    OpenAIError = Exception


T = TypeVar("T", bound=BaseModel)


class GPTGenerationError(Exception):
    """Raised when Azure OpenAI generation fails."""


@dataclass(frozen=True)
class AzureOpenAISettings:
    endpoint: str
    api_key: str
    deployment: str
    api_version: str

    @classmethod
    def from_env(cls) -> "AzureOpenAISettings":
        load_dotenv()
        return cls(
            endpoint=os.getenv("AZURE_OPENAI_ENDPOINT", "").strip(),
            api_key=os.getenv("AZURE_OPENAI_API_KEY", "").strip(),
            deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT", "").strip(),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION", "").strip(),
        )

    def missing_required(self) -> list[str]:
        missing = []
        if not self.endpoint:
            missing.append("AZURE_OPENAI_ENDPOINT")
        if not self.api_key:
            missing.append("AZURE_OPENAI_API_KEY")
        if not self.deployment:
            missing.append("AZURE_OPENAI_DEPLOYMENT")
        if not self.api_version:
            missing.append("AZURE_OPENAI_API_VERSION")
        return missing


class CourseBuilderAgent:
    def __init__(self, settings: AzureOpenAISettings | None = None) -> None:
        self.settings = settings or AzureOpenAISettings.from_env()
        self._client = None

    def generate_structured(
        self,
        prompt: str,
        response_model: type[T],
        temperature: float = 0.2,
        max_tokens: int = 4000,
    ) -> T:
        self._ensure_ready()
        schema = json.dumps(response_model.model_json_schema(), indent=2)
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a senior instructional designer and Azure AI education architect. "
                    "Return accurate, practical learning material as valid JSON only."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"{prompt}\n\n"
                    "Return a single JSON object that conforms to this schema:\n"
                    f"{schema}\n\n"
                    "Do not include markdown fences, commentary, or extra keys."
                ),
            },
        ]

        try:
            completion = self._chat_completion(messages, temperature, max_tokens, use_json_mode=True)
        except OpenAIError as exc:
            if "response_format" not in str(exc).lower():
                raise GPTGenerationError(f"Azure OpenAI generation failed: {exc}") from exc
            completion = self._chat_completion(messages, temperature, max_tokens, use_json_mode=False)
        except Exception as exc:
            raise GPTGenerationError(f"Azure OpenAI generation failed: {exc}") from exc

        try:
            content = completion.choices[0].message.content or ""
            if getattr(completion.choices[0].message, "refusal", None):
                raise GPTGenerationError("The model refused to generate the requested learning media.")
            data = _extract_json_object(content)
            return response_model.model_validate(data)
        except (json.JSONDecodeError, ValidationError, KeyError, IndexError) as exc:
            raise GPTGenerationError(f"Azure OpenAI returned invalid JSON for {response_model.__name__}: {exc}") from exc

    def _ensure_ready(self) -> None:
        if AzureOpenAI is None:
            raise GPTGenerationError("openai is not installed. Install requirements.txt first.")
        missing = self.settings.missing_required()
        if missing:
            raise GPTGenerationError("Missing Azure OpenAI configuration: " + ", ".join(missing))

    def _client_instance(self) -> AzureOpenAI:
        if self._client is None:
            self._client = AzureOpenAI(
                azure_endpoint=self.settings.endpoint,
                api_key=self.settings.api_key,
                api_version=self.settings.api_version,
            )
        return self._client

    def _chat_completion(
        self,
        messages: list[dict[str, str]],
        temperature: float,
        max_tokens: int,
        use_json_mode: bool,
    ):
        kwargs = {
            "model": self.settings.deployment,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if use_json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        return self._client_instance().chat.completions.create(**kwargs)


def _extract_json_object(content: str) -> dict:
    text = content.strip()
    if text.startswith("```"):
        text = text.strip("`")
        text = text.removeprefix("json").strip()

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise
        parsed = json.loads(text[start : end + 1])

    if not isinstance(parsed, dict):
        raise json.JSONDecodeError("Expected a JSON object", content, 0)
    return parsed
