# Job Rejection Diagnostic Agent

An evidence-first rejection diagnosis copilot for students and new grads. The app analyzes a resume, a target job description, and optional rejection notes to explain why a candidate is getting rejected, what evidence is missing, how to patch the resume, and what to do this week to improve the next application.

Built with `Google ADK`, `Gemini`, `Arize Phoenix`, `Phoenix MCP`, `Firestore`, and `Streamlit`.

## Core workflow

1. Upload a resume and paste a job description.
2. Diagnose ATS gaps, evidence gaps, and level-fit gaps.
3. Generate exact bullet rewrites, summary edits, project reframes, and a one-week action plan.
4. Save the job packet, revisit it later, and prepare interview questions from the same analysis.
5. Trace each run to Phoenix, evaluate the output quality, and draft prompt improvements from low-scoring runs.

## Project layout

```text
app/streamlit_app.py
src/job_rejection_agent/
tests/
infra/
scripts/
```

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
streamlit run app/streamlit_app.py
```

## Environment variables

- `GOOGLE_API_KEY`: Gemini access for ADK and direct generation.
- `PHOENIX_API_KEY`: Phoenix Cloud API key.
- `PHOENIX_BASE_URL`: Phoenix Cloud base URL.
- `PHOENIX_PROJECT_NAME`: Trace project name.
- `FIRESTORE_PROJECT_ID`: Firestore project for saved job packets.
- `SESSION_DB_URL`: ADK durable session database URL.

## Cloud Run deployment

```bash
gcloud run services replace infra/cloudrun.yaml
```

Or build manually from the included Dockerfile:

```bash
gcloud run deploy job-rejection-agent \
  --source . \
  --region us-central1 \
  --allow-unauthenticated
```

## Demo flow

1. Diagnose a real rejection.
2. Open Phoenix and show the trace plus eval annotations.
3. Show the saved job packet inside the app.
4. Trigger prompt candidate generation and compare the draft against the baseline.

## Verification

- `python3 -m unittest discover -s tests`
- `python3 scripts/smoke_test.py`

