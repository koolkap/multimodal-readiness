from __future__ import annotations

import json
from typing import Any


SOURCE_EXCERPT_LIMIT = 9000


def course_outline_prompt(knowledge: Any, source_markdown: str = "") -> str:
    return f"""
Create a production-quality 4 week course outline from the extracted educational knowledge.

Required output content:
- Course Overview
- Course Duration set to 4 weeks
- Weekly Breakdown with Week 1, Week 2, Week 3, and Week 4
- Learning Outcomes
- Modules with topics and learning outcomes
- Difficulty Breakdown as beginner, intermediate, and advanced percentages that sum to 100

Instructional quality requirements:
- Keep the course realistic for educators to teach.
- Preserve extracted metadata, topics, concepts, learning objectives, key terms, and resources.
- Treat analyzer_fields as authoritative structured output from Azure AI Content Understanding.
- Fill gaps only with reasonable instructional design inferences from the source.
- Do not introduce unrelated domains.

Extracted knowledge:
{_json_block(knowledge)}

Content Understanding source excerpt:
{_source_excerpt(source_markdown)}
""".strip()


def lesson_plan_prompt(knowledge: Any, course_outline: Any, source_markdown: str = "") -> str:
    return f"""
Create one detailed lesson plan for the first major module in the course.

Required output content:
- Objectives
- Agenda
- Activities
- Assessment

The lesson should fit within the 4 week course and align to the generated course outline.
Use extracted analyzer_fields and resources when choosing examples or activities.

Extracted knowledge:
{_json_block(knowledge)}

Course outline:
{_json_block(course_outline)}

Content Understanding source excerpt:
{_source_excerpt(source_markdown)}
""".strip()


def quiz_prompt(knowledge: Any, course_outline: Any, source_markdown: str = "") -> str:
    return f"""
Create an assessment quiz for the course.

Required output content:
- Exactly 5 multiple choice questions
- Each MCQ must include 4 options, the correct answer, and a short rationale
- Exactly 3 short answer questions
- Each short answer question must include an expected answer and rubric

Questions must test the extracted topics, concepts, learning objectives, and key terms.
Prefer facts from analyzer_fields when available.

Extracted knowledge:
{_json_block(knowledge)}

Course outline:
{_json_block(course_outline)}

Content Understanding source excerpt:
{_source_excerpt(source_markdown)}
""".strip()


def assignment_prompt(knowledge: Any, course_outline: Any, source_markdown: str = "") -> str:
    return f"""
Create a practical course assignment.

Required output content:
- Problem Statement
- Deliverables
- Evaluation Criteria
- Estimated Effort

The assignment must reinforce the course learning objectives and be feasible for learners.
Use the extracted course metadata, programming language, resources, and analyzer_fields when relevant.

Extracted knowledge:
{_json_block(knowledge)}

Course outline:
{_json_block(course_outline)}

Content Understanding source excerpt:
{_source_excerpt(source_markdown)}
""".strip()


def project_prompt(knowledge: Any, course_outline: Any, source_markdown: str = "") -> str:
    return f"""
Create a mini project for the course.

Required output content:
- Project Title
- Architecture
- Features
- Deliverables

The project should synthesize the main concepts and be suitable for the course difficulty.
Use extracted analyzer_fields to choose realistic architecture and feature choices.

Extracted knowledge:
{_json_block(knowledge)}

Course outline:
{_json_block(course_outline)}

Content Understanding source excerpt:
{_source_excerpt(source_markdown)}
""".strip()


def presentation_prompt(knowledge: Any, course_outline: Any, source_markdown: str = "") -> str:
    return f"""
Create a 10 slide presentation outline for educators.

Required output content:
- Exactly 10 slides
- Each slide must include Slide Title, Key Points, and Speaker Notes
- The slide sequence should follow the 4 week learning flow and reflect extracted course metadata, topics, concepts, resources, and analyzer_fields.

Extracted knowledge:
{_json_block(knowledge)}

Course outline:
{_json_block(course_outline)}

Content Understanding source excerpt:
{_source_excerpt(source_markdown)}
""".strip()


def _json_block(value: Any) -> str:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    return json.dumps(value, indent=2, ensure_ascii=True)


def _source_excerpt(source_markdown: str) -> str:
    if not source_markdown:
        return "No additional source excerpt available."
    excerpt = source_markdown[:SOURCE_EXCERPT_LIMIT]
    if len(source_markdown) > SOURCE_EXCERPT_LIMIT:
        excerpt += "\n[Source excerpt truncated for prompt length.]"
    return excerpt
