from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class AppModel(BaseModel):
    model_config = ConfigDict(extra="ignore", str_strip_whitespace=True)


class CourseKnowledge(AppModel):
    course_title: str = ""
    topics: list[str] = Field(default_factory=list)
    concepts: list[str] = Field(default_factory=list)
    learning_objectives: list[str] = Field(default_factory=list)
    key_terms: list[str] = Field(default_factory=list)

    def is_empty(self) -> bool:
        return not any(
            [
                self.course_title,
                self.topics,
                self.concepts,
                self.learning_objectives,
                self.key_terms,
            ]
        )


class ContentUnderstandingResult(AppModel):
    file_name: str
    content_type: str
    analyzer_id: str
    markdown: str = ""
    extracted_knowledge: CourseKnowledge = Field(default_factory=CourseKnowledge)
    raw_result: dict[str, Any] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
    usage: dict[str, Any] = Field(default_factory=dict)


class WeekPlan(AppModel):
    week: int
    title: str
    topics: list[str] = Field(default_factory=list)
    activities: list[str] = Field(default_factory=list)
    outcomes: list[str] = Field(default_factory=list)


class ModulePlan(AppModel):
    module_title: str
    topics: list[str] = Field(default_factory=list)
    learning_outcomes: list[str] = Field(default_factory=list)


class DifficultyBreakdown(AppModel):
    beginner: int = Field(default=0, ge=0, le=100)
    intermediate: int = Field(default=0, ge=0, le=100)
    advanced: int = Field(default=0, ge=0, le=100)


class CourseOutline(AppModel):
    course_overview: str = ""
    course_duration: str = "4 weeks"
    weekly_breakdown: list[WeekPlan] = Field(default_factory=list)
    learning_outcomes: list[str] = Field(default_factory=list)
    modules: list[ModulePlan] = Field(default_factory=list)
    difficulty_breakdown: DifficultyBreakdown = Field(default_factory=DifficultyBreakdown)


class LessonPlan(AppModel):
    lesson_title: str
    objectives: list[str] = Field(default_factory=list)
    agenda: list[str] = Field(default_factory=list)
    activities: list[str] = Field(default_factory=list)
    assessment: list[str] = Field(default_factory=list)


class MCQ(AppModel):
    question: str
    options: list[str] = Field(default_factory=list)
    correct_answer: str
    rationale: str = ""


class ShortAnswerQuestion(AppModel):
    question: str
    expected_answer: str
    rubric: str = ""


class Quiz(AppModel):
    mcqs: list[MCQ] = Field(default_factory=list)
    short_answers: list[ShortAnswerQuestion] = Field(default_factory=list)


class Assignment(AppModel):
    problem_statement: str
    deliverables: list[str] = Field(default_factory=list)
    evaluation_criteria: list[str] = Field(default_factory=list)
    estimated_effort: str = ""


class MiniProject(AppModel):
    project_title: str
    architecture: list[str] = Field(default_factory=list)
    features: list[str] = Field(default_factory=list)
    deliverables: list[str] = Field(default_factory=list)


class PresentationSlide(AppModel):
    slide_number: int
    slide_title: str
    key_points: list[str] = Field(default_factory=list)
    speaker_notes: str = ""


class Presentation(AppModel):
    slides: list[PresentationSlide] = Field(default_factory=list)


class LearningMediaBundle(AppModel):
    content_understanding_result: ContentUnderstandingResult | None = None
    course_outline: CourseOutline | None = None
    lesson_plan: LessonPlan | None = None
    quiz: Quiz | None = None
    assignment: Assignment | None = None
    mini_project: MiniProject | None = None
    presentation: Presentation | None = None
