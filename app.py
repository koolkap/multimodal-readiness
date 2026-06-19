from __future__ import annotations

import hashlib
import json
from typing import Any

import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv

from models.course_models import (
    Assignment,
    ContentUnderstandingResult,
    CourseOutline,
    LearningMediaBundle,
    LessonPlan,
    MiniProject,
    Presentation,
    Quiz,
)
from services.content_understanding import (
    AzureContentUnderstandingError,
    ContentUnderstandingError,
    ContentUnderstandingService,
    ContentUnderstandingSettings,
    EmptyContentUnderstandingResult,
    InvalidContentError,
)
from services.course_builder import AzureOpenAISettings, CourseBuilderAgent, GPTGenerationError
from services.learning_media import (
    LearningMediaGenerator,
    assignment_to_markdown,
    bundle_to_markdown,
    course_outline_to_markdown,
    lesson_plan_to_markdown,
    presentation_to_markdown,
    project_to_markdown,
    quiz_to_markdown,
)


APP_TITLE = "AI Course Builder & Learning Media Generator"
MEDIA_KEYS = {
    "course_outline": CourseOutline,
    "lesson_plan": LessonPlan,
    "quiz": Quiz,
    "assignment": Assignment,
    "mini_project": MiniProject,
    "presentation": Presentation,
}


def main() -> None:
    load_dotenv()
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    _init_session_state()

    cu_settings = ContentUnderstandingSettings.from_env()
    openai_settings = AzureOpenAISettings.from_env()

    st.title(APP_TITLE)
    _render_sidebar(cu_settings, openai_settings)
    _render_upload_step(cu_settings)

    cu_result = _session_model("content_understanding_result", ContentUnderstandingResult)
    if cu_result:
        _render_content_understanding_result(cu_result)
        _render_generation_controls(cu_result, openai_settings)
        _render_analytics(cu_result, _session_model("course_outline", CourseOutline))
        _render_media_tabs(cu_result)


def _init_session_state() -> None:
    defaults = {
        "content_understanding_result": None,
        "analysis_hash": None,
        "media_generation_key": None,
    }
    defaults.update({key: None for key in MEDIA_KEYS})
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def _render_sidebar(
    cu_settings: ContentUnderstandingSettings,
    openai_settings: AzureOpenAISettings,
) -> None:
    st.sidebar.header("Azure Configuration")
    st.sidebar.caption("Loaded from environment variables or .env.")
    st.sidebar.text_input(
        "Content Understanding Endpoint",
        value=cu_settings.endpoint or "Not configured",
        disabled=True,
    )
    st.sidebar.text_input(
        "Analyzer ID",
        value=cu_settings.analyzer_id or "Not configured",
        disabled=True,
    )
    st.sidebar.text_input(
        "GPT Deployment",
        value=openai_settings.deployment or "Not configured",
        disabled=True,
    )

    cu_missing = cu_settings.missing_required()
    gpt_missing = openai_settings.missing_required()
    if cu_missing:
        st.sidebar.warning("Missing Content Understanding settings: " + ", ".join(cu_missing))
    if gpt_missing:
        st.sidebar.warning("Missing Azure OpenAI settings: " + ", ".join(gpt_missing))


def _render_upload_step(cu_settings: ContentUnderstandingSettings) -> None:
    st.header("Step 1: Upload Educational Content")
    uploaded_file = st.file_uploader(
        "Upload an educational PDF or DOCX file",
        type=["pdf", "docx"],
        accept_multiple_files=False,
    )

    analyze_disabled = uploaded_file is None or bool(cu_settings.missing_required())
    if st.button("Analyze Content", type="primary", disabled=analyze_disabled):
        if uploaded_file is None:
            st.warning("Upload a PDF or DOCX file first.")
            return

        file_bytes = uploaded_file.getvalue()
        analysis_hash = _analysis_hash(file_bytes, uploaded_file.name, cu_settings.analyzer_id)
        if (
            st.session_state.get("analysis_hash") == analysis_hash
            and st.session_state.get("content_understanding_result")
        ):
            st.info("Using the existing Content Understanding result for this upload.")
            return

        try:
            with st.spinner("Analyzing content with Azure AI Content Understanding..."):
                result = ContentUnderstandingService(cu_settings).analyze_uploaded_file(
                    file_bytes=file_bytes,
                    file_name=uploaded_file.name,
                    content_type=uploaded_file.type,
                )
            st.session_state["content_understanding_result"] = result.model_dump(mode="json")
            st.session_state["analysis_hash"] = analysis_hash
            _reset_learning_media()
            st.success("Content analysis completed.")
        except InvalidContentError as exc:
            st.error(f"Invalid upload: {exc}")
        except EmptyContentUnderstandingResult as exc:
            st.error(f"Empty Content Understanding result: {exc}")
        except AzureContentUnderstandingError as exc:
            st.error(f"Azure API failure: {exc}")
        except ContentUnderstandingError as exc:
            st.error(str(exc))


def _render_content_understanding_result(result: ContentUnderstandingResult) -> None:
    with st.expander("Content Understanding Result", expanded=True):
        st.json(result.extracted_knowledge.model_dump(mode="json"))
        if result.warnings:
            st.warning("Warnings: " + "; ".join(result.warnings))

    with st.expander("Content Understanding Markdown Preview", expanded=False):
        preview = result.markdown[:5000] if result.markdown else "No markdown returned."
        st.markdown(preview)


def _render_generation_controls(
    result: ContentUnderstandingResult,
    openai_settings: AzureOpenAISettings,
) -> None:
    st.header("Step 2: Generate Learning Media")
    missing = openai_settings.missing_required()
    if missing:
        st.warning("Configure Azure OpenAI before generating media: " + ", ".join(missing))
        return

    generation_key = _generation_key(st.session_state.get("analysis_hash"), openai_settings)
    media_complete = all(st.session_state.get(key) for key in MEDIA_KEYS)

    col_generate, col_regenerate = st.columns([1, 1])
    with col_generate:
        generate_clicked = st.button("Generate Learning Media", type="primary")
    with col_regenerate:
        regenerate = st.checkbox("Force regenerate", value=False)

    if not generate_clicked:
        return

    if media_complete and st.session_state.get("media_generation_key") == generation_key and not regenerate:
        st.info("Using stored learning media for this analyzed content.")
        return

    if regenerate:
        _reset_learning_media()

    agent = CourseBuilderAgent(openai_settings)
    generator = LearningMediaGenerator(agent)
    progress = st.progress(0)
    status = st.empty()

    try:
        knowledge = result.extracted_knowledge
        source_markdown = result.markdown

        if st.session_state.get("course_outline") is None:
            status.info("Generating course outline...")
            outline = generator.generate_course_outline(knowledge, source_markdown)
            st.session_state["course_outline"] = outline.model_dump(mode="json")
        else:
            outline = _session_model("course_outline", CourseOutline)
        progress.progress(16)

        if outline is None:
            raise GPTGenerationError("Course outline was not generated.")

        steps = [
            ("lesson_plan", "Generating lesson plan...", generator.generate_lesson_plan),
            ("quiz", "Generating quiz...", generator.generate_quiz),
            ("assignment", "Generating assignment...", generator.generate_assignment),
            ("mini_project", "Generating mini project...", generator.generate_mini_project),
            ("presentation", "Generating presentation outline...", generator.generate_presentation),
        ]

        for index, (session_key, label, generate_fn) in enumerate(steps, start=1):
            if st.session_state.get(session_key) is None:
                status.info(label)
                generated = generate_fn(knowledge, outline, source_markdown)
                st.session_state[session_key] = generated.model_dump(mode="json")
            progress.progress(16 + index * 16)

        st.session_state["media_generation_key"] = generation_key
        status.success("Learning media generated.")
    except GPTGenerationError as exc:
        status.empty()
        st.error(f"GPT failure: {exc}")
    finally:
        progress.empty()


def _render_analytics(
    result: ContentUnderstandingResult,
    outline: CourseOutline | None,
) -> None:
    st.header("Learning Analytics")
    col_topics, col_objectives, col_difficulty = st.columns(3)
    with col_topics:
        st.plotly_chart(_topic_distribution_fig(result), use_container_width=True)
    with col_objectives:
        st.plotly_chart(_learning_objectives_fig(result), use_container_width=True)
    with col_difficulty:
        st.plotly_chart(_difficulty_breakdown_fig(outline), use_container_width=True)


def _render_media_tabs(result: ContentUnderstandingResult) -> None:
    if not any(st.session_state.get(key) for key in MEDIA_KEYS):
        return

    tabs = st.tabs(
        [
            "Course Outline",
            "Lesson Plan",
            "Quiz Generator",
            "Assignment Generator",
            "Mini Project",
            "Presentation Generator",
        ]
    )

    outline = _session_model("course_outline", CourseOutline)
    lesson = _session_model("lesson_plan", LessonPlan)
    quiz = _session_model("quiz", Quiz)
    assignment = _session_model("assignment", Assignment)
    project = _session_model("mini_project", MiniProject)
    presentation = _session_model("presentation", Presentation)

    with tabs[0]:
        if outline:
            _render_course_outline(outline)
            st.download_button(
                "Download Course Outline",
                data=course_outline_to_markdown(outline, result.extracted_knowledge.course_title),
                file_name="course_outline.md",
                mime="text/markdown",
            )
        else:
            st.info("Generate learning media to view the course outline.")

    with tabs[1]:
        if lesson:
            _render_lesson_plan(lesson)
            st.download_button(
                "Download Lesson Plan",
                data=lesson_plan_to_markdown(lesson),
                file_name="lesson_plan.md",
                mime="text/markdown",
            )
        else:
            st.info("Generate learning media to view the lesson plan.")

    with tabs[2]:
        if quiz:
            _render_quiz(quiz)
            st.download_button(
                "Download Quiz",
                data=quiz_to_markdown(quiz),
                file_name="quiz.md",
                mime="text/markdown",
            )
        else:
            st.info("Generate learning media to view the quiz.")

    with tabs[3]:
        if assignment:
            _render_assignment(assignment)
        else:
            st.info("Generate learning media to view the assignment.")

    with tabs[4]:
        if project:
            _render_project(project)
        else:
            st.info("Generate learning media to view the mini project.")

    with tabs[5]:
        if presentation:
            _render_presentation(presentation)
        else:
            st.info("Generate learning media to view the presentation outline.")

    _render_exports(result, outline, lesson, quiz, assignment, project, presentation)


def _render_course_outline(outline: CourseOutline) -> None:
    st.subheader("Course Overview")
    st.write(outline.course_overview)
    st.metric("Course Duration", outline.course_duration)

    st.subheader("Learning Outcomes")
    for outcome in outline.learning_outcomes:
        st.markdown(f"- {outcome}")

    st.subheader("Weekly Breakdown")
    for week in outline.weekly_breakdown:
        with st.container(border=True):
            st.markdown(f"### Week {week.week}: {week.title}")
            st.markdown("**Topics**")
            for topic in week.topics:
                st.markdown(f"- {topic}")
            st.markdown("**Activities**")
            for activity in week.activities:
                st.markdown(f"- {activity}")
            st.markdown("**Outcomes**")
            for outcome in week.outcomes:
                st.markdown(f"- {outcome}")

    st.subheader("Modules")
    for module in outline.modules:
        with st.container(border=True):
            st.markdown(f"### {module.module_title}")
            st.markdown("**Topics**")
            for topic in module.topics:
                st.markdown(f"- {topic}")
            st.markdown("**Learning Outcomes**")
            for outcome in module.learning_outcomes:
                st.markdown(f"- {outcome}")


def _render_lesson_plan(lesson: LessonPlan) -> None:
    st.subheader(lesson.lesson_title)
    _render_list("Objectives", lesson.objectives)
    _render_list("Agenda", lesson.agenda)
    _render_list("Activities", lesson.activities)
    _render_list("Assessment", lesson.assessment)


def _render_quiz(quiz: Quiz) -> None:
    st.subheader("Multiple Choice Questions")
    for index, mcq in enumerate(quiz.mcqs, start=1):
        with st.container(border=True):
            st.markdown(f"**{index}. {mcq.question}**")
            for option in mcq.options:
                st.markdown(f"- {option}")
            st.success(f"Correct answer: {mcq.correct_answer}")
            if mcq.rationale:
                st.caption(mcq.rationale)

    st.subheader("Short Answer Questions")
    for index, item in enumerate(quiz.short_answers, start=1):
        with st.container(border=True):
            st.markdown(f"**{index}. {item.question}**")
            st.markdown(f"Expected answer: {item.expected_answer}")
            if item.rubric:
                st.caption(f"Rubric: {item.rubric}")


def _render_assignment(assignment: Assignment) -> None:
    st.subheader("Problem Statement")
    st.write(assignment.problem_statement)
    _render_list("Deliverables", assignment.deliverables)
    _render_list("Evaluation Criteria", assignment.evaluation_criteria)
    if assignment.estimated_effort:
        st.metric("Estimated Effort", assignment.estimated_effort)
    st.download_button(
        "Download Assignment",
        data=assignment_to_markdown(assignment),
        file_name="assignment.md",
        mime="text/markdown",
    )


def _render_project(project: MiniProject) -> None:
    st.subheader(project.project_title)
    _render_list("Architecture", project.architecture)
    _render_list("Features", project.features)
    _render_list("Deliverables", project.deliverables)
    st.download_button(
        "Download Mini Project",
        data=project_to_markdown(project),
        file_name="mini_project.md",
        mime="text/markdown",
    )


def _render_presentation(presentation: Presentation) -> None:
    for slide in presentation.slides:
        with st.container(border=True):
            st.markdown(f"### Slide {slide.slide_number}: {slide.slide_title}")
            _render_list("Key Points", slide.key_points)
            st.markdown("**Speaker Notes**")
            st.write(slide.speaker_notes)
    st.download_button(
        "Download Presentation Outline",
        data=presentation_to_markdown(presentation),
        file_name="presentation_outline.md",
        mime="text/markdown",
    )


def _render_exports(
    result: ContentUnderstandingResult,
    outline: CourseOutline | None,
    lesson: LessonPlan | None,
    quiz: Quiz | None,
    assignment: Assignment | None,
    project: MiniProject | None,
    presentation: Presentation | None,
) -> None:
    bundle = LearningMediaBundle(
        content_understanding_result=result,
        course_outline=outline,
        lesson_plan=lesson,
        quiz=quiz,
        assignment=assignment,
        mini_project=project,
        presentation=presentation,
    )
    st.header("Exports")
    col_json, col_markdown = st.columns(2)
    with col_json:
        st.download_button(
            "Export JSON",
            data=json.dumps(bundle.model_dump(mode="json", exclude_none=True), indent=2),
            file_name="ai_course_builder_export.json",
            mime="application/json",
        )
    with col_markdown:
        st.download_button(
            "Export Markdown",
            data=bundle_to_markdown(bundle),
            file_name="ai_course_builder_export.md",
            mime="text/markdown",
        )


def _render_list(title: str, values: list[str]) -> None:
    st.subheader(title)
    if values:
        for value in values:
            st.markdown(f"- {value}")
    else:
        st.info("Not specified.")


def _topic_distribution_fig(result: ContentUnderstandingResult):
    knowledge = result.extracted_knowledge
    labels = knowledge.topics[:10] or ["No topics extracted"]
    corpus_items = knowledge.concepts + knowledge.learning_objectives + knowledge.key_terms
    values = []
    for topic in labels:
        tokens = [token.lower() for token in topic.split() if len(token) > 3]
        score = 1
        if tokens:
            score = max(
                1,
                sum(
                    1
                    for item in corpus_items
                    if any(token in item.lower() for token in tokens[:4])
                ),
            )
        values.append(score)

    fig = px.bar(
        x=labels,
        y=values,
        labels={"x": "Topic", "y": "Knowledge Mentions"},
        title="Topic Distribution",
        color=values,
        color_continuous_scale="Viridis",
    )
    fig.update_layout(showlegend=False, margin=dict(l=10, r=10, t=50, b=10), height=320)
    return fig


def _learning_objectives_fig(result: ContentUnderstandingResult):
    count = len(result.extracted_knowledge.learning_objectives)
    fig = go.Figure(
        go.Indicator(
            mode="number",
            value=count,
            title={"text": "Learning Objectives Count"},
            number={"font": {"size": 56}},
        )
    )
    fig.update_layout(margin=dict(l=10, r=10, t=50, b=10), height=320)
    return fig


def _difficulty_breakdown_fig(outline: CourseOutline | None):
    labels = ["Beginner", "Intermediate", "Advanced"]
    if outline:
        breakdown = outline.difficulty_breakdown
        values = [breakdown.beginner, breakdown.intermediate, breakdown.advanced]
    else:
        values = [0, 0, 0]

    if sum(values) == 0:
        fig = go.Figure()
        fig.add_annotation(
            text="Generate learning media to show difficulty breakdown.",
            x=0.5,
            y=0.5,
            showarrow=False,
        )
    else:
        fig = px.pie(
            names=labels,
            values=values,
            title="Difficulty Breakdown",
            color_discrete_sequence=px.colors.qualitative.Set2,
        )
    fig.update_layout(margin=dict(l=10, r=10, t=50, b=10), height=320)
    return fig


def _session_model(key: str, model_type):
    value = st.session_state.get(key)
    if value is None:
        return None
    if isinstance(value, model_type):
        return value
    return model_type.model_validate(value)


def _reset_learning_media() -> None:
    for key in MEDIA_KEYS:
        st.session_state[key] = None
    st.session_state["media_generation_key"] = None


def _analysis_hash(file_bytes: bytes, file_name: str, analyzer_id: str) -> str:
    digest = hashlib.sha256()
    digest.update(file_name.encode("utf-8"))
    digest.update(analyzer_id.encode("utf-8"))
    digest.update(file_bytes)
    return digest.hexdigest()


def _generation_key(analysis_hash: str | None, settings: AzureOpenAISettings) -> str:
    digest = hashlib.sha256()
    digest.update((analysis_hash or "").encode("utf-8"))
    digest.update(settings.endpoint.encode("utf-8"))
    digest.update(settings.deployment.encode("utf-8"))
    digest.update(settings.api_version.encode("utf-8"))
    return digest.hexdigest()


if __name__ == "__main__":
    main()
