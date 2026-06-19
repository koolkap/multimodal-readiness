from __future__ import annotations

from models.course_models import (
    Assignment,
    CourseKnowledge,
    CourseOutline,
    LearningMediaBundle,
    LessonPlan,
    MiniProject,
    Presentation,
    Quiz,
)
from services.course_builder import CourseBuilderAgent
from utils.prompts import (
    assignment_prompt,
    course_outline_prompt,
    lesson_plan_prompt,
    presentation_prompt,
    project_prompt,
    quiz_prompt,
)


class LearningMediaGenerator:
    def __init__(self, agent: CourseBuilderAgent | None = None) -> None:
        self.agent = agent or CourseBuilderAgent()

    def generate_course_outline(
        self,
        knowledge: CourseKnowledge,
        source_markdown: str = "",
    ) -> CourseOutline:
        return self.agent.generate_structured(
            course_outline_prompt(knowledge, source_markdown),
            CourseOutline,
        )

    def generate_lesson_plan(
        self,
        knowledge: CourseKnowledge,
        course_outline: CourseOutline,
        source_markdown: str = "",
    ) -> LessonPlan:
        return self.agent.generate_structured(
            lesson_plan_prompt(knowledge, course_outline, source_markdown),
            LessonPlan,
        )

    def generate_quiz(
        self,
        knowledge: CourseKnowledge,
        course_outline: CourseOutline,
        source_markdown: str = "",
    ) -> Quiz:
        return self.agent.generate_structured(
            quiz_prompt(knowledge, course_outline, source_markdown),
            Quiz,
        )

    def generate_assignment(
        self,
        knowledge: CourseKnowledge,
        course_outline: CourseOutline,
        source_markdown: str = "",
    ) -> Assignment:
        return self.agent.generate_structured(
            assignment_prompt(knowledge, course_outline, source_markdown),
            Assignment,
        )

    def generate_mini_project(
        self,
        knowledge: CourseKnowledge,
        course_outline: CourseOutline,
        source_markdown: str = "",
    ) -> MiniProject:
        return self.agent.generate_structured(
            project_prompt(knowledge, course_outline, source_markdown),
            MiniProject,
        )

    def generate_presentation(
        self,
        knowledge: CourseKnowledge,
        course_outline: CourseOutline,
        source_markdown: str = "",
    ) -> Presentation:
        return self.agent.generate_structured(
            presentation_prompt(knowledge, course_outline, source_markdown),
            Presentation,
        )

    def generate_all(
        self,
        knowledge: CourseKnowledge,
        source_markdown: str = "",
    ) -> LearningMediaBundle:
        course_outline = self.generate_course_outline(knowledge, source_markdown)
        return LearningMediaBundle(
            course_outline=course_outline,
            lesson_plan=self.generate_lesson_plan(knowledge, course_outline, source_markdown),
            quiz=self.generate_quiz(knowledge, course_outline, source_markdown),
            assignment=self.generate_assignment(knowledge, course_outline, source_markdown),
            mini_project=self.generate_mini_project(knowledge, course_outline, source_markdown),
            presentation=self.generate_presentation(knowledge, course_outline, source_markdown),
        )


def course_outline_to_markdown(outline: CourseOutline, title: str = "Course Outline") -> str:
    lines = [f"# {title}", "", "## Course Overview", outline.course_overview, ""]
    lines.extend(["## Course Duration", outline.course_duration, ""])
    lines.append("## Learning Outcomes")
    lines.extend(_bullet_list(outline.learning_outcomes))
    lines.extend(["", "## Weekly Breakdown"])
    for week in outline.weekly_breakdown:
        lines.extend([f"### Week {week.week}: {week.title}", ""])
        lines.extend(["Topics:", *_bullet_list(week.topics), ""])
        lines.extend(["Activities:", *_bullet_list(week.activities), ""])
        lines.extend(["Outcomes:", *_bullet_list(week.outcomes), ""])
    lines.append("## Modules")
    for module in outline.modules:
        lines.extend([f"### {module.module_title}", ""])
        lines.extend(["Topics:", *_bullet_list(module.topics), ""])
        lines.extend(["Learning Outcomes:", *_bullet_list(module.learning_outcomes), ""])
    return "\n".join(lines).strip() + "\n"


def lesson_plan_to_markdown(lesson: LessonPlan) -> str:
    lines = [f"# {lesson.lesson_title}", "", "## Objectives"]
    lines.extend(_bullet_list(lesson.objectives))
    lines.extend(["", "## Agenda", *_bullet_list(lesson.agenda)])
    lines.extend(["", "## Activities", *_bullet_list(lesson.activities)])
    lines.extend(["", "## Assessment", *_bullet_list(lesson.assessment)])
    return "\n".join(lines).strip() + "\n"


def quiz_to_markdown(quiz: Quiz) -> str:
    lines = ["# Quiz", "", "## Multiple Choice Questions"]
    for index, item in enumerate(quiz.mcqs, start=1):
        lines.extend([f"{index}. {item.question}"])
        for option in item.options:
            lines.append(f"   - {option}")
        lines.extend([f"   - Correct Answer: {item.correct_answer}", f"   - Rationale: {item.rationale}", ""])
    lines.append("## Short Answer Questions")
    for index, item in enumerate(quiz.short_answers, start=1):
        lines.extend(
            [
                f"{index}. {item.question}",
                f"   - Expected Answer: {item.expected_answer}",
                f"   - Rubric: {item.rubric}",
                "",
            ]
        )
    return "\n".join(lines).strip() + "\n"


def assignment_to_markdown(assignment: Assignment) -> str:
    lines = ["# Assignment", "", "## Problem Statement", assignment.problem_statement, ""]
    lines.extend(["## Deliverables", *_bullet_list(assignment.deliverables), ""])
    lines.extend(["## Evaluation Criteria", *_bullet_list(assignment.evaluation_criteria), ""])
    if assignment.estimated_effort:
        lines.extend(["## Estimated Effort", assignment.estimated_effort, ""])
    return "\n".join(lines).strip() + "\n"


def project_to_markdown(project: MiniProject) -> str:
    lines = [f"# {project.project_title}", "", "## Architecture"]
    lines.extend(_bullet_list(project.architecture))
    lines.extend(["", "## Features", *_bullet_list(project.features)])
    lines.extend(["", "## Deliverables", *_bullet_list(project.deliverables)])
    return "\n".join(lines).strip() + "\n"


def presentation_to_markdown(presentation: Presentation) -> str:
    lines = ["# Presentation Outline", ""]
    for slide in presentation.slides:
        lines.extend([f"## Slide {slide.slide_number}: {slide.slide_title}", "", "Key Points:"])
        lines.extend(_bullet_list(slide.key_points))
        lines.extend(["", "Speaker Notes:", slide.speaker_notes, ""])
    return "\n".join(lines).strip() + "\n"


def bundle_to_markdown(bundle: LearningMediaBundle) -> str:
    sections = []
    title = "Course Outline"
    if bundle.content_understanding_result:
        course_title = bundle.content_understanding_result.extracted_knowledge.course_title
        if course_title:
            title = course_title
    if bundle.course_outline:
        sections.append(course_outline_to_markdown(bundle.course_outline, title))
    if bundle.lesson_plan:
        sections.append(lesson_plan_to_markdown(bundle.lesson_plan))
    if bundle.quiz:
        sections.append(quiz_to_markdown(bundle.quiz))
    if bundle.assignment:
        sections.append(assignment_to_markdown(bundle.assignment))
    if bundle.mini_project:
        sections.append(project_to_markdown(bundle.mini_project))
    if bundle.presentation:
        sections.append(presentation_to_markdown(bundle.presentation))
    return "\n\n".join(sections).strip() + "\n"


def _bullet_list(values: list[str]) -> list[str]:
    return [f"- {value}" for value in values] if values else ["- Not specified"]
