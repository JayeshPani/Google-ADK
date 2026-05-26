"""Streamlit UI for the Job Rejection Diagnostic Agent."""

from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import uuid

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import streamlit as st

from job_rejection_agent.agents import AgentRuntime
from job_rejection_agent.observability import PromptOptimizer
from job_rejection_agent.services import render_packet_markdown

RESUME_FIXTURES = ROOT / "tests" / "fixtures" / "resumes"
JD_FIXTURES = ROOT / "tests" / "fixtures" / "jds"
DEMO_CASES = {
    "ML Platform Fit": ("nisha_ml_newgrad.txt", "ml_platform_engineer.md"),
    "Backend New Grad Fit": ("arjun_backend_student.txt", "backend_newgrad.md"),
    "Data Analyst Fit": ("meera_data_analyst.txt", "data_analyst_rotational.md"),
    "AI Internship Fit": ("rahul_fullstack_intern.txt", "ai_products_intern.md"),
}


def inject_theme() -> None:
    st.markdown(
        """
        <style>
          @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;700&display=swap');
          :root {
            --bg: #09111f;
            --bg-soft: #10192e;
            --panel: rgba(11, 21, 39, 0.82);
            --stroke: rgba(169, 180, 208, 0.18);
            --text: #edf3ff;
            --muted: #aab7d4;
            --mint: #7ee787;
            --amber: #ffb454;
            --rose: #ff8f8f;
            --ice: #7dd3fc;
          }
          html, body, [class*="css"]  {
            font-family: 'IBM Plex Sans', sans-serif;
            color: var(--text);
          }
          .stApp {
            background:
              radial-gradient(circle at 10% 10%, rgba(125, 211, 252, 0.14), transparent 28%),
              radial-gradient(circle at 90% 0%, rgba(255, 180, 84, 0.12), transparent 24%),
              linear-gradient(180deg, #08111e 0%, #0e1830 62%, #08111e 100%);
          }
          h1, h2, h3 {
            font-family: 'Space Grotesk', sans-serif;
            letter-spacing: -0.03em;
          }
          .hero {
            padding: 1.5rem 1.7rem;
            border: 1px solid var(--stroke);
            border-radius: 24px;
            background:
              linear-gradient(135deg, rgba(125, 211, 252, 0.16), rgba(126, 231, 135, 0.08)),
              rgba(8, 17, 30, 0.88);
            box-shadow: 0 20px 50px rgba(0, 0, 0, 0.25);
            margin-bottom: 1rem;
          }
          .hero-eyebrow {
            color: var(--mint);
            text-transform: uppercase;
            letter-spacing: 0.16em;
            font-size: 0.76rem;
            font-weight: 700;
          }
          .hero-title {
            font-size: 2.1rem;
            line-height: 1.04;
            margin: 0.45rem 0;
            font-family: 'Space Grotesk', sans-serif;
          }
          .hero-copy {
            color: var(--muted);
            max-width: 58rem;
            font-size: 1rem;
          }
          .status-strip {
            display: flex;
            gap: 0.6rem;
            flex-wrap: wrap;
            margin-top: 1rem;
          }
          .chip {
            display: inline-flex;
            align-items: center;
            gap: 0.35rem;
            padding: 0.45rem 0.7rem;
            border-radius: 999px;
            background: rgba(255,255,255,0.05);
            border: 1px solid var(--stroke);
            color: var(--text);
            font-size: 0.82rem;
          }
          .metric-card, .panel-card {
            border: 1px solid var(--stroke);
            background: var(--panel);
            border-radius: 20px;
            padding: 1rem 1.1rem;
            backdrop-filter: blur(12px);
          }
          .metric-label {
            color: var(--muted);
            font-size: 0.82rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
          }
          .metric-value {
            font-family: 'Space Grotesk', sans-serif;
            font-size: 2rem;
            margin-top: 0.25rem;
          }
          .gap-card {
            border: 1px solid var(--stroke);
            border-left: 4px solid var(--amber);
            background: rgba(8, 17, 30, 0.55);
            border-radius: 18px;
            padding: 0.9rem 1rem;
            margin-bottom: 0.75rem;
          }
          .gap-title {
            font-weight: 700;
            font-family: 'Space Grotesk', sans-serif;
          }
          .gap-copy {
            color: var(--muted);
            margin-top: 0.3rem;
            font-size: 0.95rem;
          }
          .small-note {
            color: var(--muted);
            font-size: 0.88rem;
          }
          div[data-testid="stFileUploader"] section,
          div[data-testid="stTextArea"] textarea,
          div[data-testid="stTextInput"] input {
            background: rgba(8, 17, 30, 0.55) !important;
            color: var(--text) !important;
          }
          div[data-testid="stTabs"] button {
            font-family: 'Space Grotesk', sans-serif;
          }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource(show_spinner=False)
def get_runtime() -> AgentRuntime:
    return AgentRuntime()


@st.cache_resource(show_spinner=False)
def get_optimizer() -> PromptOptimizer:
    return PromptOptimizer()


def get_user_id() -> str:
    if "user_id" not in st.session_state:
        st.session_state["user_id"] = f"demo-{uuid.uuid4().hex[:10]}"
    return st.session_state["user_id"]


def render_hero(runtime: AgentRuntime) -> None:
    adk_state = "ADK live" if runtime.adk_available() else "Heuristic fallback"
    st.markdown(
        f"""
        <div class="hero">
          <div class="hero-eyebrow">Google ADK + Phoenix diagnostic copilot</div>
          <div class="hero-title">Diagnose why this application got rejected, then patch it fast.</div>
          <div class="hero-copy">
            This demo is optimized for students and new grads. It separates ATS gaps, evidence gaps, and level-fit gaps,
            then turns them into exact edits, recruiter-readable project reframes, and a one-week recovery plan.
          </div>
          <div class="status-strip">
            <span class="chip">{adk_state}</span>
            <span class="chip">Phoenix traces ready when keys are present</span>
            <span class="chip">Cloud Run-friendly persistence path</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_metrics(packet) -> None:
    metrics = [
        ("Overall match", f"{packet.report.score_overall:.1f}/10"),
        ("ATS fit", f"{packet.report.score_ats:.1f}/10"),
        ("Evidence fit", f"{packet.report.score_evidence:.1f}/10"),
        ("Level fit", f"{packet.report.score_level_fit:.1f}/10"),
    ]
    cols = st.columns(4)
    for col, (label, value) in zip(cols, metrics):
        col.markdown(
            f"""
            <div class="metric-card">
              <div class="metric-label">{label}</div>
              <div class="metric-value">{value}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_gap_cards(packet) -> None:
    for gap in packet.report.top_gaps[:4]:
        st.markdown(
            f"""
            <div class="gap-card">
              <div class="gap-title">{gap.title} <span class="small-note">[{gap.severity}]</span></div>
              <div class="gap-copy">{gap.details}</div>
              <div class="small-note" style="margin-top:0.5rem;">Fix: {gap.recommended_fix}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def load_demo_case(case_name: str) -> tuple[str, str, Path]:
    resume_name, jd_name = DEMO_CASES[case_name]
    resume_path = RESUME_FIXTURES / resume_name
    jd_text = (JD_FIXTURES / jd_name).read_text(encoding="utf-8")
    return resume_name, jd_text, resume_path


def persist_uploaded_file(uploaded_file) -> Path:
    suffix = Path(uploaded_file.name).suffix or ".txt"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as handle:
        handle.write(uploaded_file.getvalue())
        return Path(handle.name)


def resolve_resume_path(uploaded_file, demo_resume_path: Path | None) -> tuple[Path, str]:
    if uploaded_file is not None:
        return persist_uploaded_file(uploaded_file), "uploaded"
    if demo_resume_path is not None:
        return demo_resume_path, "demo"
    raise ValueError("No resume source provided.")


def sidebar(runtime: AgentRuntime) -> None:
    st.sidebar.markdown("### Session")
    st.sidebar.code(get_user_id())
    st.sidebar.markdown("### Runtime")
    st.sidebar.write(f"Primary model: `{runtime.settings.model_id}`")
    st.sidebar.write(f"Model failover chain: `{', '.join(runtime.settings.generation_model_candidates)}`")
    st.sidebar.write(f"ADK enabled: `{runtime.adk_available()}`")
    st.sidebar.write(f"Phoenix project: `{runtime.settings.phoenix_project_name}`")
    st.sidebar.write(f"Storage: `{runtime.settings.firestore_project_id or 'local json fallback'}`")


def main() -> None:
    st.set_page_config(page_title="Job Rejection Diagnostic Agent", layout="wide")
    inject_theme()
    runtime = get_runtime()
    optimizer = get_optimizer()
    sidebar(runtime)
    render_hero(runtime)

    if "last_result" not in st.session_state:
        st.session_state["last_result"] = None

    diagnose_tab, patch_tab, track_tab = st.tabs(["Diagnose", "Patch", "Track & Prep"])

    with diagnose_tab:
        left, right = st.columns([1.15, 0.85])
        with left:
            st.markdown("### Input packet")
            demo_case = st.selectbox("Start from a seeded demo case", ["None"] + list(DEMO_CASES.keys()))
            uploaded_file = st.file_uploader("Upload resume", type=["pdf", "docx", "txt", "md"])
            jd_text_default = ""
            demo_resume_path: Path | None = None
            if demo_case != "None":
                _, jd_text_default, demo_resume_path = load_demo_case(demo_case)
                st.caption(f"Loaded seeded case: `{demo_resume_path.name}`")
            if uploaded_file is not None and demo_resume_path is not None:
                st.info("Using the uploaded resume. The seeded demo case only prefills the job description.")
            jd_text = st.text_area("Paste job description", height=260, value=jd_text_default)
            rejection_notes = st.text_area("Optional recruiter or rejection notes", height=120)
            run_clicked = st.button("Run diagnostic", type="primary", use_container_width=True)
        with right:
            st.markdown("### What this app optimizes for")
            st.markdown(
                """
                <div class="panel-card">
                  <strong>Not another ATS scanner.</strong><br/><br/>
                  The report is built to answer four judge-visible questions:
                  <ul>
                    <li>What exact evidence is missing?</li>
                    <li>Is the role too senior for the current resume story?</li>
                    <li>Which edits can be made today without inventing facts?</li>
                    <li>Can the agent explain why it made each recommendation?</li>
                  </ul>
                </div>
                """,
                unsafe_allow_html=True,
            )

        if run_clicked:
            if not jd_text.strip():
                st.error("Paste a job description or choose a seeded demo case.")
            elif not uploaded_file and demo_resume_path is None:
                st.error("Upload a resume or choose a seeded demo case.")
            else:
                resume_path, _ = resolve_resume_path(uploaded_file, demo_resume_path)
                with st.spinner("Diagnosing rejection pattern..."):
                    result = runtime.run_diagnostic(
                        resume_path=str(resume_path),
                        jd_text=jd_text,
                        rejection_notes=rejection_notes,
                        user_id=get_user_id(),
                    )
                st.session_state["last_result"] = result
                st.success("Diagnosis complete.")

        result = st.session_state.get("last_result")
        if result:
            packet = result["packet"]
            st.caption(f"Analyzed resume: `{packet.resume_name}`")
            render_metrics(packet)
            st.markdown("### Rejection drivers")
            render_gap_cards(packet)
            st.markdown("### Full diagnostic")
            st.markdown(result["text"])
            if result.get("eval_scores"):
                st.markdown("### Eval overlay")
                eval_cols = st.columns(len(result["eval_scores"]))
                for col, (key, value) in zip(eval_cols, result["eval_scores"].items()):
                    col.markdown(
                        f"""
                        <div class="panel-card">
                          <div class="metric-label">{key.replace('_', ' ')}</div>
                          <div style="font-size:1.1rem;font-weight:700;">{value}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

    with patch_tab:
        result = st.session_state.get("last_result")
        packet = result["packet"] if result else None
        if not packet:
            st.info("Run a diagnostic first to unlock exact edits and project reframes.")
        else:
            st.markdown("### Exact edits")
            for edit in packet.report.exact_edits:
                st.markdown(
                    f"""
                    <div class="panel-card" style="margin-bottom:0.8rem;">
                      <div class="metric-label">{edit.section}</div>
                      <div style="margin:0.4rem 0;color:#aab7d4;"><strong>Before:</strong> {edit.original_text}</div>
                      <div><strong>After:</strong> {edit.rewritten_text}</div>
                      <div class="small-note" style="margin-top:0.45rem;">Why: {edit.reason}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("### Project reframes")
                for item in packet.report.project_reframes:
                    st.markdown(f"- {item}")
            with col2:
                st.markdown("### This week's action plan")
                for step in packet.report.action_plan:
                    st.markdown(f"- {step}")

    with track_tab:
        tracker = runtime.service.tracker
        entries = tracker.list_entries(get_user_id())
        left, right = st.columns([1.05, 0.95])
        with left:
            st.markdown("### Saved job packets")
            if not entries:
                st.info("Saved packets appear here after you run a diagnosis.")
            else:
                labels = {
                    entry.packet_id: f"{entry.company_name} · {entry.role_title} · {entry.score_overall:.1f}/10"
                    for entry in entries
                }
                selected_id = st.selectbox("Open packet", list(labels.keys()), format_func=lambda value: labels[value])
                saved_packet = tracker.get(selected_id)
                if saved_packet:
                    render_metrics(saved_packet)
                    st.markdown(render_packet_markdown(saved_packet))
        with right:
            st.markdown("### Prompt improvement loop")
            st.caption("Draft a candidate prompt from observed traces, then promote it only if live held-out ADK runs improve without regressions.")
            if st.button("Generate candidate prompt", use_container_width=True):
                candidate_prompt, improvement_run = optimizer.optimize()
                st.session_state["candidate_prompt"] = candidate_prompt
                st.session_state["improvement_run"] = improvement_run
            if st.session_state.get("improvement_run"):
                improvement_run = st.session_state["improvement_run"]
                st.markdown(
                    f"""
                    <div class="panel-card">
                      <div class="metric-label">Promotion</div>
                      <div style="font-size:1.15rem;font-weight:700;">{improvement_run.promoted}</div>
                      <div class="small-note" style="margin-top:0.5rem;">{improvement_run.analysis}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                st.json(
                    {
                        "baseline_scores": improvement_run.baseline_scores,
                        "candidate_scores": improvement_run.candidate_scores,
                    }
                )
                st.text_area("Candidate prompt preview", st.session_state.get("candidate_prompt", ""), height=260)


if __name__ == "__main__":
    main()
