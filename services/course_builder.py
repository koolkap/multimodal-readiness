from __future__ import annotations

import json
import os
from dataclasses import dataclass
from urllib.parse import urlparse, urlunparse
from typing import TypeVar

from dotenv import load_dotenv
from pydantic import BaseModel, ValidationError

try:
    from azure.identity import DefaultAzureCredential, get_bearer_token_provider
except ImportError:  # pragma: no cover - azure-identity is in requirements
    DefaultAzureCredential = None
    get_bearer_token_provider = None

try:
    from openai import APIError, APITimeoutError, OpenAI, OpenAIError
except ImportError:  # pragma: no cover - handled at runtime in Streamlit
    APIError = Exception
    APITimeoutError = Exception
    OpenAI = None
    OpenAIError = Exception


T = TypeVar("T", bound=BaseModel)


class GPTGenerationError(Exception):
    """Raised when Azure OpenAI generation fails."""


@dataclass(frozen=True)
class AzureOpenAISettings:
    endpoint: str
    deployment: str
    api_key: str = ""
    api_version: str = ""
    auth_method: str = ""

    @classmethod
    def from_env(cls) -> "AzureOpenAISettings":
        import sys
        
        # Debug: Check .env file paths
        print(f"[DEBUG] Current working directory: {os.getcwd()}")
        print(f"[DEBUG] __file__ location: {os.path.abspath(__file__)}")
        
        # Try to find .env file
        possible_env_paths = [
            os.path.join(os.getcwd(), ".env"),
            os.path.join(os.path.dirname(__file__), "..", ".env"),
            os.path.expanduser("~/.env"),
        ]
        for path in possible_env_paths:
            abs_path = os.path.abspath(path)
            exists = os.path.exists(abs_path)
            print(f"[DEBUG] .env path check: {abs_path} - exists: {exists}")
        
        # Load only from .env file, not from system environment
        env_file_path = os.path.join(os.getcwd(), ".env")
        if os.path.exists(env_file_path):
            print(f"[DEBUG] Loading from .env file: {env_file_path}")
            load_dotenv(env_file_path, override=True)  # override=True ensures .env takes precedence
        else:
            print("[DEBUG] No .env file found in current directory, loading from system environment")
            load_dotenv()
        
        # Debug: Check environment variables
        print("[DEBUG] Loading AzureOpenAISettings from environment...")
        raw_api_key = os.getenv("AZURE_OPENAI_API_KEY", "")
        print(f"[DEBUG] Raw AZURE_OPENAI_API_KEY from os.getenv: {raw_api_key[:20]}***" if raw_api_key else "[DEBUG] Raw AZURE_OPENAI_API_KEY: EMPTY")
        
        endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "").strip()
        api_key = os.getenv("AZURE_OPENAI_API_KEY", "").strip()
        deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT", "").strip()
        api_version = os.getenv("AZURE_OPENAI_API_VERSION", "").strip()
        auth_method = os.getenv("AZURE_OPENAI_AUTH_METHOD", "").strip().lower()
        
        print(f"[DEBUG] Stripped AZURE_OPENAI_API_KEY: {api_key[:20]}***" if api_key else "[DEBUG] Stripped AZURE_OPENAI_API_KEY: EMPTY")
        print(f"[DEBUG] AZURE_OPENAI_ENDPOINT: {endpoint[:50]}...")
        print(f"[DEBUG] AZURE_OPENAI_DEPLOYMENT: {deployment}")
        print(f"[DEBUG] AZURE_OPENAI_AUTH_METHOD: {auth_method}")
        
        return cls(
            endpoint=endpoint,
            api_key=api_key,
            deployment=deployment,
            api_version=api_version,
            auth_method=auth_method,
        )

    def missing_required(self) -> list[str]:
        missing = []
        if not self.endpoint:
            missing.append("AZURE_OPENAI_ENDPOINT")
        if self.uses_api_key_auth() and not self.api_key:
            missing.append("AZURE_OPENAI_API_KEY")
        if not self.deployment:
            missing.append("AZURE_OPENAI_DEPLOYMENT")
        if self.uses_entra_auth() and (DefaultAzureCredential is None or get_bearer_token_provider is None):
            missing.append("azure-identity")
        return missing

    def uses_entra_auth(self) -> bool:
        return self.auth_method in {
            "aad",
            "azure_ad",
            "default_azure_credential",
            "entra",
            "entra_id",
            "managed_identity",
        }

    def uses_api_key_auth(self) -> bool:
        return not self.uses_entra_auth()

    def openai_base_url(self) -> str:
        endpoint = self.endpoint
        
        print(f"@@@@@@@@@@@@@@@@@@@@Original endpoint: {endpoint}")
        return self.endpoint 


class CourseBuilderAgent:
    def __init__(self, settings: AzureOpenAISettings | None = None) -> None:
        if settings is None:
            print("[DEBUG] CourseBuilderAgent: Creating settings from environment...")
            settings = AzureOpenAISettings.from_env()
        else:
            print(f"[DEBUG] CourseBuilderAgent: Using provided settings with API key: {settings.api_key[:20]}***" if settings.api_key else "[DEBUG] CourseBuilderAgent: Using provided settings with empty API key")
        
        self.settings = settings
        print(f"[DEBUG] CourseBuilderAgent initialized with API key: {self.settings.api_key[:20]}***" if self.settings.api_key else "[DEBUG] CourseBuilderAgent initialized with empty API key")
        self._client = None
        self._credential = None

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
        if OpenAI is None:
            raise GPTGenerationError("openai is not installed. Install requirements.txt first.")
        missing = self.settings.missing_required()
        if missing:
            raise GPTGenerationError("Missing Azure OpenAI configuration: " + ", ".join(missing))

    def _client_instance(self) -> OpenAI:
        if self._client is None:
            kwargs = {"base_url": self.settings.openai_base_url()}
            print(f"#############OpenAI kwargs: {kwargs}")
            if self.settings.uses_entra_auth():
                print("############Using Entra ID authentication for Azure OpenAI")
                self._credential = DefaultAzureCredential()
                kwargs["api_key"] = get_bearer_token_provider(
                    self._credential,
                    "https://ai.azure.com/.default",
                )
            else:
                print("############Using API key authentication for Azure OpenAI")
                print(f"[DEBUG] API Key from settings: {self.settings.api_key[:10]}***" if self.settings.api_key else "[DEBUG] API Key is EMPTY")
                kwargs["api_key"] = "unused"
                kwargs["default_headers"] = {"api-key": self.settings.api_key}
                print(f"[DEBUG] Headers being set: {kwargs['default_headers']}")
            print(f"[DEBUG] Final kwargs keys: {kwargs.keys()}")
            self._client = OpenAI(**kwargs)
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
