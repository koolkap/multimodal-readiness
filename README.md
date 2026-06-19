# AI Course Builder & Learning Media Generator

A production-ready Streamlit application that turns educational PDFs or DOCX files into course outlines, lesson plans, quizzes, assignments, mini projects, and presentation outlines.

The app uses Azure AI Content Understanding first to analyze uploaded educational content. Only after Content Understanding produces extracted knowledge does Azure OpenAI generate learning media.

## Architecture

```text
Educational PDF or DOCX
    -> Azure AI Content Understanding
    -> Knowledge Extraction
    -> Azure OpenAI Course Builder Agent
    -> Learning Media Generation
    -> Streamlit UI, JSON export, Markdown export
```

## Features

- Upload PDF and DOCX educational files.
- Analyze content with Azure AI Content Understanding.
- Extract course title, topics, concepts, learning objectives, and key terms.
- Generate a 4 week course outline.
- Generate lesson plan, quiz, assignment, mini project, and 10 slide presentation outline.
- Show Plotly analytics for topic distribution, learning objective count, and difficulty breakdown.
- Store results in Streamlit session state to avoid unnecessary regeneration.
- Download course outline, lesson plan, and quiz.
- Export the complete result as JSON or Markdown.

## Installation

Use Python 3.11 or later.

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

On macOS or Linux:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Environment Variables

Copy `.env.example` to `.env` and set the values from Azure.

```bash
AZURE_OPENAI_ENDPOINT=
AZURE_OPENAI_API_KEY=
AZURE_OPENAI_DEPLOYMENT=
AZURE_OPENAI_API_VERSION=

CONTENTUNDERSTANDING_ENDPOINT=
CONTENTUNDERSTANDING_KEY=
CONTENTUNDERSTANDING_ANALYZER_ID=
CONTENTUNDERSTANDING_API_VERSION=
```

Do not commit `.env` files or API keys.

## Azure Setup

1. Create or use an Azure AI Foundry resource that supports Azure AI Content Understanding.
2. Deploy an Azure OpenAI model such as GPT-4.1 or GPT-4o.
3. Copy the Azure OpenAI endpoint, API key, deployment name, and API version into `.env`.
4. Ensure the deployed model name is the deployment name you provide in `AZURE_OPENAI_DEPLOYMENT`.

## Content Understanding Setup

1. Create or use a Microsoft Foundry resource with Content Understanding enabled.
2. Configure required model deployment defaults for Content Understanding in the Foundry resource.
3. Create or choose an analyzer and set `CONTENTUNDERSTANDING_ANALYZER_ID`.
4. For general educational documents, `prebuilt-documentSearch` can extract markdown and semantic document content.
5. For best structured extraction, use a custom analyzer with fields such as `CourseTitle`, `Topics`, `Concepts`, `LearningObjectives`, and `KeyTerms`.

The app also performs deterministic fallback extraction from Content Understanding markdown when custom fields are not available.

## Run Instructions

```bash
streamlit run app.py
```

Open the Streamlit URL, upload a PDF or DOCX educational document, click **Analyze Content**, review the Content Understanding result, then click **Generate Learning Media**.

## Project Structure

```text
app.py
services/
  content_understanding.py
  course_builder.py
  learning_media.py
models/
  course_models.py
utils/
  prompts.py
.env.example
requirements.txt
README.md
```
Workflow Agent 
