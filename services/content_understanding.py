from __future__ import annotations

import os
import re
import sys
from collections.abc import Mapping
from concurrent.futures import TimeoutError as FuturesTimeoutError
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from models.course_models import ContentUnderstandingResult, CourseKnowledge

CONTENT_UNDERSTANDING_IMPORT_ERROR: str | None = None

try:
    from azure.ai.contentunderstanding import ContentUnderstandingClient, to_llm_input
    from azure.core.credentials import AzureKeyCredential
    from azure.core.exceptions import AzureError, HttpResponseError
    from azure.identity import DefaultAzureCredential
except ImportError:  # pragma: no cover - handled at runtime in Streamlit
    CONTENT_UNDERSTANDING_IMPORT_ERROR = str(sys.exc_info()[1])
    ContentUnderstandingClient = None
    to_llm_input = None
    AzureKeyCredential = None
    AzureError = Exception
    HttpResponseError = Exception
    DefaultAzureCredential = None


SUPPORTED_EXTENSIONS = {
    ".pdf": "application/pdf",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
}

FIELD_ALIASES = {
    "course_title": ["course_title", "CourseTitle", "title", "Title", "DocumentTitle"],
    "instructor_name": ["instructor_name", "InstructorName", "Instructor", "Lecturer"],
    "course_code": ["course_code", "CourseCode", "Code", "CourseIdentifier"],
    "term": ["term", "Term", "AcademicTerm", "Year"],
    "emphasis_language": [
        "emphasis_language",
        "EmphasisLanguage",
        "CourseStructure.EmphasisLanguage",
        "ProgrammingLanguage",
    ],
    "topics": [
        "topics",
        "Topics",
        "course_topics",
        "CourseTopics",
        "CourseStructure.Parts",
        "Parts",
    ],
    "concepts": ["concepts", "Concepts", "core_concepts", "CoreConcepts"],
    "learning_objectives": [
        "learning_objectives",
        "LearningObjectives",
        "objectives",
        "Objectives",
        "learning_outcomes",
        "LearningOutcomes",
    ],
    "key_terms": ["key_terms", "KeyTerms", "terms", "Terms", "glossary", "Glossary"],
}

COURSEBUILDER_CONCEPT_FIELDS = [
    "ComputerFundamentals",
    "HardwareFundamentals",
    "MachineArchitectures",
    "PrimitiveTypesInJava",
    "ReferenceTypesAndClasses",
    "PointersAndReferences",
]

COURSEBUILDER_RESOURCE_FIELDS = [
    "BooksAndResources.OopBooks",
    "BooksAndResources.JavaBooks",
    "BooksAndResources.DesignPatternsBooks",
    "BooksAndResources.WebResources",
    "BooksAndResources.CourseWebPageUrl",
]


class ContentUnderstandingError(Exception):
    """Base error for content understanding failures."""


class InvalidContentError(ContentUnderstandingError):
    """Raised when the uploaded educational content is invalid."""


class EmptyContentUnderstandingResult(ContentUnderstandingError):
    """Raised when Azure returns no usable content."""


class AzureContentUnderstandingError(ContentUnderstandingError):
    """Raised when Azure Content Understanding fails."""


@dataclass(frozen=True)
class _CapturedAnalysisResponse:
    result: Any
    raw_response: dict[str, Any]


@dataclass(frozen=True)
class ContentUnderstandingSettings:
    endpoint: str
    key: str | None
    analyzer_id: str
    api_version: str | None

    @classmethod
    def from_env(cls) -> "ContentUnderstandingSettings":
        load_dotenv()
        return cls(
            endpoint=_first_env(
                "CONTENTUNDERSTANDING_ENDPOINT",
                "AZURE_CONTENT_UNDERSTANDING_ENDPOINT",
            ),
            key=_first_env(
                "CONTENTUNDERSTANDING_KEY",
                "CONTENT_UNDERSTANDING_KEY",
                "AZURE_CONTENT_UNDERSTANDING_KEY",
            )
            or None,
            analyzer_id=_first_env(
                "CONTENTUNDERSTANDING_ANALYZER_ID",
                "CONTENT_UNDERSTANDING_ANALYZER_ID",
                "ANALYZER_ID",
            ),
            api_version=_first_env(
                "CONTENTUNDERSTANDING_API_VERSION",
                "CONTENT_UNDERSTANDING_API_VERSION",
                "AZURE_CONTENT_UNDERSTANDING_API_VERSION",
                "API_VERSION",
            )
            or None,
        )

    def missing_required(self) -> list[str]:
        missing = []
        if not self.endpoint:
            missing.append("CONTENTUNDERSTANDING_ENDPOINT or AZURE_CONTENT_UNDERSTANDING_ENDPOINT")
        if not self.analyzer_id:
            missing.append("CONTENTUNDERSTANDING_ANALYZER_ID")
        if not self.api_version:
            missing.append("CONTENTUNDERSTANDING_API_VERSION")
        return missing


class ContentUnderstandingService:
    def __init__(self, settings: ContentUnderstandingSettings | None = None) -> None:
        self.settings = settings or ContentUnderstandingSettings.from_env()

    def analyze_uploaded_file(
        self,
        file_bytes: bytes,
        file_name: str,
        content_type: str | None = None,
    ) -> ContentUnderstandingResult:
        self._ensure_ready()
        detected_content_type = self._validate_upload(file_bytes, file_name, content_type)

        client = None
        credential = None
        poller = None
        try:
            client, credential = self._create_client()
            poller = client.begin_analyze_binary(
                analyzer_id=self.settings.analyzer_id,
                binary_input=file_bytes,
                content_type=detected_content_type,
                cls=_capture_analysis_response,
            )
            captured = poller.result(timeout=90)
            result = captured.result if isinstance(captured, _CapturedAnalysisResponse) else captured
            raw_response = captured.raw_response if isinstance(captured, _CapturedAnalysisResponse) else {}
            return self._build_result(result, file_name, detected_content_type, raw_response)
        except (InvalidContentError, EmptyContentUnderstandingResult):
            raise
        except FuturesTimeoutError as exc:
            operation_id = getattr(poller, "operation_id", None)
            detail = (
                "Azure AI Content Understanding analysis did not complete within 90 seconds. "
                "This usually means the analyzer is still processing, the analyzer configuration is invalid, "
                "or required model deployments/default mappings are missing in the Foundry resource."
            )
            if operation_id:
                detail += f" Operation ID: {operation_id}."
            raise AzureContentUnderstandingError(detail) from exc
        except HttpResponseError as exc:
            detail = getattr(exc, "message", None) or str(exc)
            raise AzureContentUnderstandingError(f"Azure Content Understanding request failed: {detail}") from exc
        except AzureError as exc:
            raise AzureContentUnderstandingError(f"Azure Content Understanding request failed: {exc}") from exc
        except Exception as exc:
            raise ContentUnderstandingError(f"Content analysis failed: {exc}") from exc
        finally:
            if client and hasattr(client, "close"):
                client.close()
            if credential and hasattr(credential, "close"):
                credential.close()

    def _ensure_ready(self) -> None:
        if ContentUnderstandingClient is None:
            detail = CONTENT_UNDERSTANDING_IMPORT_ERROR or "unknown import error"
            raise ContentUnderstandingError(
                "Azure AI Content Understanding could not be imported in the active Python environment. "
                f"Interpreter: {sys.executable}. Import error: {detail}"
            )
        missing = self.settings.missing_required()
        if missing:
            raise ContentUnderstandingError(
                "Missing Content Understanding configuration: " + ", ".join(missing)
            )

    def _create_client(self) -> tuple[Any, Any]:
        if self.settings.key:
            credential = AzureKeyCredential(self.settings.key)
        else:
            credential = DefaultAzureCredential()

        kwargs: dict[str, Any] = {
            "endpoint": self.settings.endpoint,
            "credential": credential,
        }
        if self.settings.api_version:
            kwargs["api_version"] = self.settings.api_version

        try:
            return ContentUnderstandingClient(**kwargs), credential
        except TypeError:
            kwargs.pop("api_version", None)
            return ContentUnderstandingClient(**kwargs), credential

    def _validate_upload(
        self,
        file_bytes: bytes,
        file_name: str,
        content_type: str | None,
    ) -> str:
        if not file_bytes:
            raise InvalidContentError("The uploaded file is empty.")

        suffix = Path(file_name).suffix.lower()
        if suffix not in SUPPORTED_EXTENSIONS:
            raise InvalidContentError("Only PDF and DOCX files are supported.")

        if suffix == ".pdf" and b"%PDF" not in file_bytes[:1024]:
            raise InvalidContentError("Invalid PDF: the file does not contain a PDF header.")

        if suffix == ".docx" and not file_bytes.startswith(b"PK"):
            raise InvalidContentError("Invalid DOCX: the file does not look like an Office document.")

        return content_type or SUPPORTED_EXTENSIONS[suffix]

    def _build_result(
        self,
        result: Any,
        file_name: str,
        content_type: str,
        raw_response: dict[str, Any] | None = None,
    ) -> ContentUnderstandingResult:
        raw_result = _analysis_payload(raw_response) or _to_plain_mapping(result)
        markdown = _extract_llm_ready_text(result, raw_result)
        analyzer_fields = _collect_field_values(result, raw_result)
        knowledge = _extract_course_knowledge(analyzer_fields, markdown)
        warnings = _normalize_list(raw_result.get("warnings", []))
        usage = raw_result.get("usage") if isinstance(raw_result.get("usage"), dict) else {}

        if not markdown.strip() and knowledge.is_empty():
            raise EmptyContentUnderstandingResult(
                "Azure Content Understanding returned no markdown, fields, or educational knowledge."
            )

        return ContentUnderstandingResult(
            file_name=file_name,
            content_type=content_type,
            analyzer_id=self.settings.analyzer_id,
            markdown=markdown,
            extracted_knowledge=knowledge,
            analyzer_fields=_top_level_fields(analyzer_fields),
            raw_result=raw_result,
            warnings=warnings,
            usage=usage,
        )


def _capture_analysis_response(pipeline_response: Any, result: Any, _headers: Any) -> _CapturedAnalysisResponse:
    try:
        raw_response = pipeline_response.http_response.json()
    except Exception:
        raw_response = {}
    return _CapturedAnalysisResponse(result=result, raw_response=_to_plain_mapping(raw_response))


def _extract_llm_ready_text(result: Any, raw_result: dict[str, Any] | None = None) -> str:
    if to_llm_input is not None:
        try:
            text = to_llm_input(result)
            if isinstance(text, str) and _has_substantive_text(text):
                return text
        except Exception:
            pass

    raw_result = raw_result or _to_plain_mapping(result)
    raw_text = _extract_text_from_raw_result(raw_result)
    if raw_text.strip():
        return raw_text

    contents = getattr(result, "contents", None) or []
    markdown_parts: list[str] = []
    for content in contents:
        markdown_parts.extend(_extract_text_from_content(content))
    return "\n\n".join(markdown_parts)


def _has_substantive_text(text: str) -> bool:
    if not text.strip():
        return False
    body = _strip_front_matter(text).strip()
    return bool(body or re.search(r"(?m)^\s*fields\s*:", text))


def _extract_text_from_raw_result(raw_result: dict[str, Any]) -> str:
    payload = _analysis_payload(raw_result)
    parts: list[str] = []
    raw_contents = payload.get("contents") if isinstance(payload, dict) else None

    if isinstance(raw_contents, list):
        for content in raw_contents:
            parts.extend(_extract_text_from_content(content))
    elif isinstance(payload, dict):
        parts.extend(_extract_text_from_content(payload))

    return "\n\n".join(_dedupe_text_blocks(parts))


def _extract_text_from_content(content: Any) -> list[str]:
    markdown = _value_for(content, "markdown")
    if isinstance(markdown, str) and markdown.strip():
        return [markdown]

    parts = _extract_paragraph_text(_value_for(content, "paragraphs"))
    if parts:
        return parts

    parts = _extract_page_text(_value_for(content, "pages"))
    if parts:
        return parts

    parts = _extract_table_text(_value_for(content, "tables"))
    if parts:
        return parts

    text = _value_for(content, "content", "text")
    if isinstance(text, str) and text.strip():
        return [text]
    return []


def _extract_paragraph_text(paragraphs: Any) -> list[str]:
    if not isinstance(paragraphs, (list, tuple)):
        return []

    parts: list[str] = []
    for paragraph in paragraphs:
        text = _value_for(paragraph, "content", "text")
        if not isinstance(text, str) or not text.strip():
            continue
        role = str(_value_for(paragraph, "role") or "").lower()
        if "title" in role:
            parts.append(f"# {text}")
        elif "heading" in role:
            parts.append(f"## {text}")
        else:
            parts.append(text)
    return parts


def _extract_page_text(pages: Any) -> list[str]:
    if not isinstance(pages, (list, tuple)):
        return []

    parts: list[str] = []
    for page in pages:
        line_texts = [
            text
            for line in (_value_for(page, "lines") or [])
            if isinstance((text := _value_for(line, "content", "text")), str) and text.strip()
        ]
        if line_texts:
            page_text = "\n".join(line_texts)
        else:
            word_texts = [
                text
                for word in (_value_for(page, "words") or [])
                if isinstance((text := _value_for(word, "content", "text")), str) and text.strip()
            ]
            page_text = " ".join(word_texts)

        if page_text.strip():
            page_number = _value_for(page, "pageNumber", "page_number")
            marker = f"<!-- InputPageNumber: {page_number} -->\n\n" if page_number else ""
            parts.append(marker + page_text)
    return parts


def _extract_table_text(tables: Any) -> list[str]:
    if not isinstance(tables, (list, tuple)):
        return []

    parts: list[str] = []
    for table in tables:
        cells = _value_for(table, "cells") or []
        rows: dict[int, dict[int, str]] = {}
        for cell in cells:
            text = _value_for(cell, "content", "text")
            row_index = _value_for(cell, "rowIndex", "row_index")
            column_index = _value_for(cell, "columnIndex", "column_index")
            if not isinstance(text, str) or row_index is None or column_index is None:
                continue
            rows.setdefault(int(row_index), {})[int(column_index)] = _clean_text(text)
        for row_index in sorted(rows):
            row_values = [value for _, value in sorted(rows[row_index].items()) if value]
            if row_values:
                parts.append(" | ".join(row_values))
    return parts


def _iter_raw_field_sets(raw_result: dict[str, Any]) -> list[Mapping[str, Any]]:
    field_sets: list[Mapping[str, Any]] = []

    def walk(value: Any, depth: int = 0) -> None:
        if depth > 10:
            return
        if isinstance(value, Mapping):
            fields = value.get("fields")
            if isinstance(fields, Mapping):
                field_sets.append(fields)
            for key in ("fieldValues", "extractedFields"):
                alternate_fields = value.get(key)
                if isinstance(alternate_fields, Mapping):
                    field_sets.append(alternate_fields)
            for child in value.values():
                walk(child, depth + 1)
        elif isinstance(value, list):
            for child in value:
                walk(child, depth + 1)

    walk(_analysis_payload(raw_result))
    return field_sets


def _extract_course_knowledge(field_values: dict[str, Any], markdown: str) -> CourseKnowledge:
    course_title = _first_value(_lookup_field(field_values, FIELD_ALIASES["course_title"]))
    instructor_name = _first_value(_lookup_field(field_values, FIELD_ALIASES["instructor_name"]))
    course_code = _first_value(_lookup_field(field_values, FIELD_ALIASES["course_code"]))
    term = _first_value(_lookup_field(field_values, FIELD_ALIASES["term"]))
    emphasis_language = _first_value(_lookup_field(field_values, FIELD_ALIASES["emphasis_language"]))

    topics = _normalize_list(_lookup_field(field_values, FIELD_ALIASES["topics"]))
    concepts = _normalize_list(_lookup_field(field_values, FIELD_ALIASES["concepts"]))
    objectives = _normalize_list(_lookup_field(field_values, FIELD_ALIASES["learning_objectives"]))
    key_terms = _normalize_list(_lookup_field(field_values, FIELD_ALIASES["key_terms"]))
    resources = _coursebuilder_resources(field_values)
    coursebuilder_topics = _coursebuilder_topics(field_values)
    coursebuilder_concepts = _coursebuilder_concepts(field_values)
    coursebuilder_terms = _coursebuilder_key_terms(field_values)

    inferred_title = _infer_title(markdown)
    inferred_topics = _infer_topics(markdown)
    inferred_objectives = _infer_objectives(markdown)
    inferred_terms = _infer_key_terms(markdown)
    synthesized_objectives = _synthesize_objectives(
        topics + coursebuilder_topics + inferred_topics,
        coursebuilder_concepts,
    )

    return CourseKnowledge(
        analyzer_fields=_top_level_fields(field_values),
        course_title=course_title or inferred_title,
        instructor_name=instructor_name,
        course_code=course_code,
        term=term,
        emphasis_language=emphasis_language,
        topics=_dedupe(topics + coursebuilder_topics + inferred_topics, limit=16),
        concepts=_dedupe(
            concepts + coursebuilder_concepts + inferred_topics[1:] + inferred_terms[:8],
            limit=32,
        ),
        learning_objectives=_dedupe(objectives + inferred_objectives + synthesized_objectives, limit=16),
        key_terms=_dedupe(key_terms + coursebuilder_terms + inferred_terms, limit=32),
        resources=_dedupe(resources, limit=24),
    )


def _collect_field_values(result: Any, raw_result: dict[str, Any]) -> dict[str, Any]:
    collected: dict[str, Any] = {}

    contents = getattr(result, "contents", None) or []
    for content in contents:
        fields = getattr(content, "fields", None)
        if isinstance(fields, dict):
            for key, value in fields.items():
                _set_field(collected, key, _field_value(value))

    for fields in _iter_raw_field_sets(raw_result):
        for key, value in fields.items():
            if key not in collected:
                _set_field(collected, key, _field_value(value))

    return collected


def _lookup_field(fields: dict[str, Any], aliases: list[str]) -> Any:
    normalized = {_normalize_key(key): value for key, value in _flatten_fields(fields).items()}
    for alias in aliases:
        value = normalized.get(_normalize_key(alias))
        if value not in (None, "", [], {}):
            return value
    return None


def _field_value(field: Any) -> Any:
    if field is None:
        return None
    if isinstance(field, (str, int, float, bool)):
        return field
    if isinstance(field, (list, tuple, set)):
        return [_field_value(item) for item in field]
    if isinstance(field, Mapping):
        if "valueArray" in field and isinstance(field["valueArray"], list):
            return [_field_value(item) for item in field["valueArray"]]
        if "valueObject" in field and isinstance(field["valueObject"], Mapping):
            return {key: _field_value(value) for key, value in field["valueObject"].items()}
        for key in (
            "valueJson",
            "valueString",
            "valueNumber",
            "valueInteger",
            "valueBoolean",
            "valueDate",
            "valueTime",
            "value",
            "content",
            "text",
        ):
            if key in field and not _is_empty_value(field[key]):
                return _field_value(field[key])
        return {
            key: _field_value(value)
            for key, value in field.items()
            if key not in {"type", "fieldType", "confidence", "source", "span"}
        }
    for attr in (
        "value_array",
        "value_object",
        "value_json",
        "value_string",
        "value_number",
        "value_integer",
        "value_boolean",
        "value_date",
        "value_time",
        "value",
        "content",
        "text",
    ):
        if hasattr(field, attr):
            value = getattr(field, attr)
            if not _is_empty_value(value):
                return _field_value(value)
    return str(field)


def _set_field(target: dict[str, Any], key: str, value: Any) -> None:
    target[key] = value
    if isinstance(value, dict):
        for child_key, child_value in value.items():
            target[f"{key}.{child_key}"] = child_value


def _flatten_fields(fields: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    flattened: dict[str, Any] = {}
    for key, value in fields.items():
        path = f"{prefix}.{key}" if prefix else key
        flattened[path] = value
        if isinstance(value, dict):
            flattened.update(_flatten_fields(value, path))
    return flattened


def _top_level_fields(fields: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in fields.items()
        if "." not in key and value not in (None, "", [], {})
    }


def _coursebuilder_topics(fields: dict[str, Any]) -> list[str]:
    topics = _normalize_list(_lookup_field(fields, ["CourseStructure.Parts", "Parts"]))
    topics.extend(_humanize_name(field_name) for field_name in COURSEBUILDER_CONCEPT_FIELDS if _field_exists(fields, field_name))
    emphasis_language = _first_value(_lookup_field(fields, ["CourseStructure.EmphasisLanguage"]))
    if emphasis_language:
        topics.append(f"{emphasis_language} programming")
    return _dedupe(topics, limit=16)


def _coursebuilder_concepts(fields: dict[str, Any]) -> list[str]:
    concepts: list[str] = []
    for field_name in COURSEBUILDER_CONCEPT_FIELDS:
        field_value = _lookup_field(fields, [field_name])
        if field_value not in (None, "", [], {}):
            concepts.append(_humanize_name(field_name))
            concepts.extend(_section_values(field_name, field_value))
    return _dedupe(concepts, limit=40)


def _coursebuilder_key_terms(fields: dict[str, Any]) -> list[str]:
    terms: list[str] = []
    for field_name in COURSEBUILDER_CONCEPT_FIELDS:
        field_value = _lookup_field(fields, [field_name])
        if isinstance(field_value, dict):
            terms.extend(_humanize_name(key) for key in field_value)
        elif field_value not in (None, "", [], {}):
            terms.append(_humanize_name(field_name))
    return _dedupe(terms, limit=32)


def _coursebuilder_resources(fields: dict[str, Any]) -> list[str]:
    resources: list[str] = []
    for field_path in COURSEBUILDER_RESOURCE_FIELDS:
        resources.extend(_normalize_list(_lookup_field(fields, [field_path])))
    return _dedupe(resources, limit=24)


def _section_values(section_name: str, value: Any) -> list[str]:
    values: list[str] = []
    if isinstance(value, dict):
        for key, child_value in value.items():
            label = _humanize_name(key)
            if isinstance(child_value, (dict, list, tuple, set)):
                for item in _normalize_list(child_value):
                    values.append(f"{label}: {item}")
            else:
                item = _first_value(child_value)
                if item:
                    values.append(f"{label}: {item}")
    else:
        values.extend(_normalize_list(value))

    if not values:
        values.append(_humanize_name(section_name))
    return values


def _synthesize_objectives(topics: list[str], concepts: list[str]) -> list[str]:
    source_items = _dedupe(topics + concepts, limit=8)
    objectives = []
    for item in source_items[:6]:
        objectives.append(f"Explain and apply {item} in an educational or practical context")
    return objectives


def _field_exists(fields: dict[str, Any], field_name: str) -> bool:
    value = _lookup_field(fields, [field_name])
    return value not in (None, "", [], {})


def _normalize_list(value: Any) -> list[str]:
    if value in (None, "", [], {}):
        return []
    if isinstance(value, dict):
        flattened = []
        for item in value.values():
            flattened.extend(_normalize_list(item))
        return flattened
    if isinstance(value, (list, tuple, set)):
        flattened = []
        for item in value:
            flattened.extend(_normalize_list(item))
        return flattened

    text = _clean_text(str(value))
    if not text:
        return []

    parts = re.split(r"\n|;|\u2022", text)
    if len(parts) == 1 and len(text) < 240 and text.count(",") >= 2:
        parts = text.split(",")
    return [_clean_text(part) for part in parts if _clean_text(part)]


def _first_value(value: Any) -> str:
    values = _normalize_list(value)
    return values[0] if values else ""


def _infer_title(markdown: str) -> str:
    body = _strip_front_matter(markdown)
    if not body.strip():
        return ""

    for pattern in (r"^\s*#\s+(.+)$", r"^\s*title:\s*(.+)$"):
        match = re.search(pattern, body, flags=re.IGNORECASE | re.MULTILINE)
        if match:
            return _clean_text(match.group(1))[:120]

    for line in body.splitlines():
        cleaned = _clean_text(line)
        if cleaned and len(cleaned) <= 120 and not cleaned.startswith("<!--"):
            return cleaned
    return ""


def _infer_topics(markdown: str) -> list[str]:
    body = _strip_front_matter(markdown)
    headings = re.findall(r"^\s{0,3}#{1,4}\s+(.+)$", body, flags=re.MULTILINE)
    cleaned = [_clean_text(heading) for heading in headings]
    return _dedupe([item for item in cleaned if 3 <= len(item) <= 120], limit=12)


def _infer_objectives(markdown: str) -> list[str]:
    body = _strip_front_matter(markdown)
    objective_lines = []
    keywords = (
        "objective",
        "outcome",
        "you will learn",
        "learners will",
        "students will",
        "able to",
    )
    for line in body.splitlines():
        cleaned = _clean_text(line)
        lowered = cleaned.lower()
        if cleaned and any(keyword in lowered for keyword in keywords):
            objective_lines.append(cleaned)
    return _dedupe(objective_lines, limit=12)


def _infer_key_terms(markdown: str) -> list[str]:
    body = _strip_front_matter(markdown)
    terms = re.findall(r"\*\*([^*\n]{3,80})\*\*", body)
    terms.extend(re.findall(r"^\s*[-*]\s*([A-Z][A-Za-z0-9 /-]{2,80}):", body, flags=re.MULTILINE))

    glossary_block = re.search(
        r"(?:key terms|glossary)\s*(.*?)(?:\n#{1,4}\s+|\Z)",
        body,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if glossary_block:
        for line in glossary_block.group(1).splitlines():
            cleaned = _clean_text(line)
            if cleaned:
                terms.append(cleaned.split(":")[0])

    return _dedupe([_clean_text(term) for term in terms if _clean_text(term)], limit=20)


def _dedupe(values: list[str], limit: int) -> list[str]:
    seen = set()
    deduped = []
    for value in values:
        cleaned = _clean_text(value)
        key = cleaned.lower()
        if cleaned and key not in seen:
            seen.add(key)
            deduped.append(cleaned)
        if len(deduped) >= limit:
            break
    return deduped


def _clean_text(value: str) -> str:
    value = re.sub(r"<[^>]+>", " ", value)
    value = value.replace("==", "")
    value = value.strip(" -*#:\t\r\n")
    value = re.sub(r"\s+", " ", value)
    return value.strip()


def _humanize_name(value: str) -> str:
    value = value.split(".")[-1]
    value = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", value)
    value = value.replace("_", " ").replace("-", " ")
    return _clean_text(value)


def _normalize_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.lower())


def _value_for(source: Any, *names: str) -> Any:
    if source is None:
        return None
    if isinstance(source, Mapping):
        for name in names:
            if name in source:
                return source[name]
    for name in names:
        if hasattr(source, name):
            return getattr(source, name)
    return None


def _is_empty_value(value: Any) -> bool:
    return value is None or value == "" or value == [] or value == {}


def _dedupe_text_blocks(values: list[str]) -> list[str]:
    seen = set()
    deduped = []
    for value in values:
        text = str(value).strip()
        key = re.sub(r"\s+", " ", text).lower()
        if text and key not in seen:
            seen.add(key)
            deduped.append(text)
    return deduped


def _strip_front_matter(text: str) -> str:
    return re.sub(r"\A\s*---\s*.*?\s*---\s*", "", text, flags=re.DOTALL)


def _analysis_payload(value: Any) -> dict[str, Any]:
    data = _to_plain_mapping(value)
    if not data:
        return {}

    for key in ("result", "analyzeResult", "analysisResult"):
        nested = data.get(key)
        if isinstance(nested, Mapping) and nested:
            payload = _analysis_payload(nested)
            if payload:
                return payload

    if any(key in data for key in ("contents", "fields", "warnings", "usage")):
        return data

    for nested in data.values():
        if isinstance(nested, Mapping) and any(
            key in nested for key in ("contents", "fields", "warnings", "usage")
        ):
            return _analysis_payload(nested)

    return data


def _to_plain_mapping(value: Any) -> dict[str, Any]:
    plain = _to_plain_data(value)
    return plain if isinstance(plain, dict) else {}


def _first_env(*names: str) -> str:
    for name in names:
        value = os.getenv(name, "").strip()
        if value:
            return value
    return ""


def _to_plain_data(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _to_plain_data(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_to_plain_data(item) for item in value]
    if hasattr(value, "as_dict"):
        try:
            return _to_plain_data(value.as_dict())
        except Exception:
            pass
    if hasattr(value, "model_dump"):
        try:
            return _to_plain_data(value.model_dump())
        except Exception:
            pass
    if hasattr(value, "__dict__"):
        return {
            key.lstrip("_"): _to_plain_data(item)
            for key, item in vars(value).items()
            if not key.startswith("__")
        }
    return str(value)
