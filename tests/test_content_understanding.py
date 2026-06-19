from __future__ import annotations

import unittest

from services.content_understanding import (
    ContentUnderstandingService,
    ContentUnderstandingSettings,
    _collect_field_values,
    _extract_llm_ready_text,
)


class ContentUnderstandingParsingTests(unittest.TestCase):
    def test_extracts_text_and_fields_from_analyze_result_envelope(self) -> None:
        raw_response = {
            "status": "Succeeded",
            "analyzeResult": {
                "contents": [
                    {
                        "paragraphs": [
                            {"role": "title", "content": "OOP Course"},
                            {"content": "Students will learn classes and inheritance."},
                        ],
                        "fields": {
                            "CourseTitle": {"type": "string", "valueString": "OOP Course"},
                            "Topics": {
                                "type": "array",
                                "valueArray": [{"type": "string", "valueString": "Classes"}],
                            },
                        },
                    }
                ]
            },
        }

        text = _extract_llm_ready_text({}, raw_response)
        fields = _collect_field_values({}, raw_response)

        self.assertIn("# OOP Course", text)
        self.assertIn("Students will learn classes", text)
        self.assertEqual(fields["CourseTitle"], "OOP Course")
        self.assertEqual(fields["Topics"], ["Classes"])

    def test_build_result_uses_raw_response_when_sdk_result_is_empty(self) -> None:
        service = ContentUnderstandingService(
            ContentUnderstandingSettings(
                endpoint="https://example.test",
                key="key",
                analyzer_id="coursebuilder",
                api_version="2025-11-01",
            )
        )
        raw_response = {
            "status": "Succeeded",
            "result": {
                "contents": [
                    {
                        "pages": [
                            {
                                "pageNumber": 1,
                                "lines": [
                                    {"content": "Object Oriented Programming"},
                                    {"content": "Students will learn polymorphism."},
                                ],
                            }
                        ],
                        "fields": {
                            "CourseTitle": {
                                "type": "string",
                                "valueString": "Object Oriented Programming",
                            }
                        },
                    }
                ]
            },
        }

        result = service._build_result(
            {},
            "oop.pdf",
            "application/pdf",
            raw_response,
        )

        self.assertIn("InputPageNumber: 1", result.markdown)
        self.assertEqual(result.extracted_knowledge.course_title, "Object Oriented Programming")
        self.assertEqual(result.analyzer_fields["CourseTitle"], "Object Oriented Programming")

    def test_front_matter_is_not_inferred_as_course_title(self) -> None:
        service = ContentUnderstandingService(
            ContentUnderstandingSettings(
                endpoint="https://example.test",
                key="key",
                analyzer_id="coursebuilder",
                api_version="2025-11-01",
            )
        )
        raw_response = {
            "result": {
                "contents": [
                    {
                        "markdown": "---\ncontentType: document\npages: 1\n---",
                        "fields": {
                            "Topics": {
                                "type": "array",
                                "valueArray": [{"type": "string", "valueString": "Classes"}],
                            }
                        },
                    }
                ]
            }
        }

        result = service._build_result({}, "oop.pdf", "application/pdf", raw_response)

        self.assertEqual(result.extracted_knowledge.course_title, "")
        self.assertEqual(result.extracted_knowledge.topics, ["Classes"])


if __name__ == "__main__":
    unittest.main()
