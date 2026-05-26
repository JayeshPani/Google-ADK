"""FastAPI frontend for the Refine job rejection diagnostic agent."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any
import sys
import uuid

from fastapi import BackgroundTasks, FastAPI, File, Form, Request, UploadFile
from fastapi.responses import HTMLResponse, PlainTextResponse, RedirectResponse, Response
from fastapi.templating import Jinja2Templates

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from job_rejection_agent.agents import AgentRuntime
from job_rejection_agent.coaching import session_overview
from job_rejection_agent.config import Settings, get_settings
from job_rejection_agent.domain import (
    InterviewSimulationSession,
    MultiJDComparison,
    RewritePatch,
    SavedJobPacket,
    TrackerEntry,
)
from job_rejection_agent.ingestion import parse_resume_file
from job_rejection_agent.observability import PromptOptimizer
from job_rejection_agent.services import AuthError, AuthService, render_packet_markdown
from job_rejection_agent.services.resume_export import build_resume_docx_bytes, build_resume_pdf_bytes


ROOT = PROJECT_ROOT
TEMPLATES = Jinja2Templates(directory=str(ROOT / "app" / "templates"))
COOKIE_NAME = "refine_user_id"
SESSION_COOKIE_NAME = "refine_session"
GOOGLE_STATE_COOKIE_NAME = "refine_google_oauth_state"
COOKIE_MAX_AGE_SECONDS = 60 * 60 * 24 * 30

RESUME_FIXTURES = ROOT / "tests" / "fixtures" / "resumes"
JD_FIXTURES = ROOT / "tests" / "fixtures" / "jds"
DEMO_CASES = {
    "ml-platform-fit": {
        "label": "ML Platform Fit",
        "resume": "nisha_ml_newgrad.txt",
        "jd": "ml_platform_engineer.md",
    },
    "backend-newgrad-fit": {
        "label": "Backend New Grad Fit",
        "resume": "arjun_backend_student.txt",
        "jd": "backend_newgrad.md",
    },
    "data-analyst-fit": {
        "label": "Data Analyst Fit",
        "resume": "meera_data_analyst.txt",
        "jd": "data_analyst_rotational.md",
    },
    "ai-intern-fit": {
        "label": "AI Internship Fit",
        "resume": "rahul_fullstack_intern.txt",
        "jd": "ai_products_intern.md",
    },
}
WEEKDAY_LABELS = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")
PATCH_SECTION_ORDER = ("summary", "experience", "project", "skills", "education")
RESUME_PREVIEW_MAX_CHARS = 2200


@dataclass(slots=True)
class ViewerContext:
    user_id: str
    label: str
    authenticated: bool
    email: str | None
    guest_user_id: str | None
    should_set_guest_cookie: bool


def _decision_meta(decision: str) -> dict[str, str]:
    mapping = {
        "apply_now": {"label": "Apply Now", "badge_class": "badge-apply-now", "tone_class": "text-emerald-700"},
        "apply_after_patch": {"label": "Apply After Patch", "badge_class": "badge-patch", "tone_class": "text-amber-700"},
        "defer": {"label": "Defer", "badge_class": "badge-defer", "tone_class": "text-slate-600"},
        "not_fit": {"label": "Not Fit", "badge_class": "badge-not-fit", "tone_class": "text-rose-700"},
    }
    return mapping.get(decision, mapping["defer"])


def _user_identity(request: Request, auth_service: AuthService) -> ViewerContext:
    guest_user_id = request.cookies.get(COOKIE_NAME)
    auth_session = auth_service.verify_session_token(request.cookies.get(SESSION_COOKIE_NAME))
    if auth_session:
        return ViewerContext(
            user_id=auth_session.user_id,
            label=auth_session.email,
            authenticated=True,
            email=auth_session.email,
            guest_user_id=guest_user_id,
            should_set_guest_cookie=False,
        )
    if guest_user_id:
        return ViewerContext(
            user_id=guest_user_id,
            label=_user_label(guest_user_id),
            authenticated=False,
            email=None,
            guest_user_id=guest_user_id,
            should_set_guest_cookie=False,
        )
    new_guest_user_id = f"guest-{uuid.uuid4().hex[:10]}"
    return ViewerContext(
        user_id=new_guest_user_id,
        label=_user_label(new_guest_user_id),
        authenticated=False,
        email=None,
        guest_user_id=new_guest_user_id,
        should_set_guest_cookie=True,
    )


def _user_label(user_id: str) -> str:
    suffix = user_id.split("-")[-1][:6].upper()
    return f"Guest {suffix}"


def _set_guest_cookie(response: HTMLResponse | RedirectResponse, user_id: str) -> None:
    response.set_cookie(
        COOKIE_NAME,
        user_id,
        max_age=COOKIE_MAX_AGE_SECONDS,
        samesite="lax",
        httponly=True,
    )


def _clear_guest_cookie(response: HTMLResponse | RedirectResponse) -> None:
    response.delete_cookie(COOKIE_NAME)


def _set_session_cookie(
    request: Request,
    response: HTMLResponse | RedirectResponse,
    session_token: str,
) -> None:
    response.set_cookie(
        SESSION_COOKIE_NAME,
        session_token,
        max_age=COOKIE_MAX_AGE_SECONDS,
        samesite="lax",
        httponly=True,
        secure=request.url.scheme == "https",
    )


def _clear_session_cookie(response: HTMLResponse | RedirectResponse) -> None:
    response.delete_cookie(SESSION_COOKIE_NAME)


def _set_google_state_cookie(
    request: Request,
    response: HTMLResponse | RedirectResponse,
    state_token: str,
) -> None:
    response.set_cookie(
        GOOGLE_STATE_COOKIE_NAME,
        state_token,
        max_age=600,
        samesite="lax",
        httponly=True,
        secure=request.url.scheme == "https",
    )


def _clear_google_state_cookie(response: HTMLResponse | RedirectResponse) -> None:
    response.delete_cookie(GOOGLE_STATE_COOKIE_NAME)


def _sanitize_next_path(next_path: str | None) -> str:
    candidate = (next_path or "").strip()
    if not candidate.startswith("/"):
        return "/history"
    if candidate.startswith("//"):
        return "/history"
    return candidate or "/history"


def _google_redirect_uri(request: Request, settings: Settings) -> str:
    return settings.google_oauth_redirect_uri or str(request.url_for("google_oauth_callback"))


def _resolve_demo_case(demo_case_key: str | None) -> dict[str, str] | None:
    if not demo_case_key:
        return None
    return DEMO_CASES.get(demo_case_key)


def _load_demo_case(demo_case_key: str) -> tuple[Path, str]:
    demo = _resolve_demo_case(demo_case_key)
    if demo is None:
        raise ValueError("Unknown demo case.")
    resume_path = RESUME_FIXTURES / demo["resume"]
    jd_path = JD_FIXTURES / demo["jd"]
    return resume_path, jd_path.read_text(encoding="utf-8")


def _packet_sections(packet: SavedJobPacket) -> dict[str, list[RewritePatch]]:
    grouped: dict[str, list[RewritePatch]] = defaultdict(list)
    for patch in packet.report.exact_edits:
        grouped[patch.section].append(patch)
    ordered: dict[str, list[RewritePatch]] = {}
    for section in PATCH_SECTION_ORDER:
        if grouped.get(section):
            ordered[section] = grouped[section]
    for section, patches in grouped.items():
        if section not in ordered:
            ordered[section] = patches
    return ordered


def _recommended_skills(packet: SavedJobPacket) -> list[str]:
    values = packet.report.missing_skills + packet.report.under_evidenced_skills
    seen: set[str] = set()
    unique: list[str] = []
    for value in values:
        key = value.strip().lower()
        if not key or key in seen:
            continue
        seen.add(key)
        unique.append(value)
    return unique[:4]


def _learning_icon(skill: str) -> str:
    lowered = skill.lower()
    if "design system" in lowered:
        return "lucide:component"
    if "prototype" in lowered or "figma" in lowered:
        return "lucide:mouse-pointer-2"
    if "research" in lowered or "user" in lowered:
        return "lucide:search"
    if "data" in lowered or "metric" in lowered or "analysis" in lowered:
        return "lucide:bar-chart-2"
    if "docker" in lowered or "backend" in lowered or "api" in lowered:
        return "lucide:server"
    return "lucide:sparkles"


def _learning_cards(packet: SavedJobPacket) -> list[dict[str, str]]:
    fallback = ["Evidence Framing", "Outcome Metrics", "Interview Storytelling", "Role Targeting"]
    skills = _recommended_skills(packet) or fallback
    cards: list[dict[str, str]] = []
    for skill in skills[:4]:
        cards.append(
            {
                "title": skill,
                "icon": _learning_icon(skill),
                "description": f"Build recruiter-visible evidence for {skill.lower()} using a focused mini-project, metric rewrite, or case-study pass.",
                "effort": "1-3h sprint",
            }
        )
    return cards


def _action_plan_cards(packet: SavedJobPacket) -> list[dict[str, str]]:
    cards: list[dict[str, str]] = []
    for index, step in enumerate(packet.report.action_plan[:7]):
        title = step
        description = step
        if ":" in step:
            _, trailing = step.split(":", 1)
            description = trailing.strip()
            title = description.split(".")[0].strip() or description
        cards.append(
            {
                "day": WEEKDAY_LABELS[index],
                "priority": "High" if index < 2 else "Medium",
                "title": title[:70],
                "description": description,
                "duration": "Under 3 hours",
            }
        )
    return cards


def _interview_columns(packet: SavedJobPacket) -> list[list[dict[str, str]]]:
    labels = (
        "Design Process",
        "Cultural Fit",
        "Technical Execution",
        "Company Specific",
    )
    questions = packet.report.interview_questions or [
        "Walk me through the strongest project on your resume and explain what changed because of your work."
    ]
    cards: list[dict[str, str]] = []
    for index, question in enumerate(questions[:8]):
        guidance = packet.report.top_gaps[min(index, len(packet.report.top_gaps) - 1)].recommended_fix if packet.report.top_gaps else "Answer with concrete evidence from your actual resume."
        cards.append(
            {
                "category": labels[index % len(labels)],
                "question": question,
                "guidance": guidance,
            }
        )
    midpoint = (len(cards) + 1) // 2
    return [cards[:midpoint], cards[midpoint:]]


def _history_entries(
    entries: list[TrackerEntry],
    *,
    status_filter: str = "all",
    sort_order: str = "newest",
) -> list[TrackerEntry]:
    filtered = entries
    if status_filter != "all":
        filtered = [entry for entry in filtered if entry.status == status_filter]
    if sort_order == "highest":
        return sorted(filtered, key=lambda entry: entry.score_overall, reverse=True)
    if sort_order == "alphabetical":
        return sorted(filtered, key=lambda entry: (entry.company_name.lower(), entry.role_title.lower()))
    return sorted(filtered, key=lambda entry: entry.updated_at, reverse=True)


def _company_initials(packet: SavedJobPacket | TrackerEntry) -> str:
    name = getattr(packet, "company_name", "") or getattr(getattr(packet, "job_requirements", None), "company_name", "")
    if not name:
        name = getattr(getattr(packet, "job_requirements", None), "role_title", "R")
    parts = [part[0].upper() for part in name.split() if part]
    return "".join(parts[:2]) or "R"


def _get_user_packet(runtime: AgentRuntime, user_id: str, packet_id: str) -> SavedJobPacket | None:
    if not packet_id:
        return None
    packet = runtime.service.tracker.get(packet_id)
    if packet is None or packet.user_id != user_id:
        return None
    return packet


def _build_resume_preview(resume_path: Path, *, display_name: str | None = None) -> dict[str, Any] | None:
    try:
        parsed = parse_resume_file(resume_path)
    except Exception:
        return None

    preview_text = parsed.normalized_text[:RESUME_PREVIEW_MAX_CHARS].strip()
    return {
        "display_name": display_name or parsed.file_name,
        "file_type": parsed.file_type.upper(),
        "text": preview_text,
        "is_truncated": len(parsed.normalized_text) > len(preview_text),
        "line_count": len([line for line in preview_text.splitlines() if line.strip()]),
    }


def _diagnose_context(
    runtime: AgentRuntime,
    *,
    packet: SavedJobPacket | None,
    demo_case: str = "",
    jd_text: str = "",
    rejection_notes: str = "",
    error_message: str = "",
    resume_hint: str = "",
    resume_preview: dict[str, Any] | None = None,
) -> dict[str, Any]:
    demo_jd = jd_text
    if not demo_jd and demo_case:
        try:
            _, demo_jd = _load_demo_case(demo_case)
        except ValueError:
            demo_jd = ""
    return {
        "active_item": "diagnose",
        "packet": packet,
        "packet_meta": _decision_meta(packet.report.recommended_decision) if packet else None,
        "top_skill_chips": packet.report.missing_skills[:4] if packet else [],
        "preview_edits": packet.report.exact_edits[:2] if packet else [],
        "preview_plan": packet.report.action_plan[:3] if packet else [],
        "eval_scores": getattr(packet, "eval_scores", None) if packet else None,
        "demo_case_key": demo_case,
        "demo_jd": demo_jd,
        "error_message": error_message,
        "resume_hint": resume_hint,
        "resume_preview": resume_preview,
        "form_state": {
            "jd_text": jd_text,
            "rejection_notes": rejection_notes,
            "demo_case": demo_case,
        },
        "runtime_status": {
            "adk_enabled": runtime.adk_available(),
            "model_chain": ", ".join(runtime.settings.generation_model_candidates),
            "phoenix_project": runtime.settings.phoenix_project_name,
        },
    }


def _render(
    request: Request,
    template_name: str,
    *,
    viewer: ViewerContext,
    context: dict[str, Any],
) -> HTMLResponse:
    response = TEMPLATES.TemplateResponse(
        request=request,
        name=template_name,
        context={
            "request": request,
            "user_id": viewer.user_id,
            "user_label": viewer.label,
            "user_authenticated": viewer.authenticated,
            "user_email": viewer.email,
            "demo_cases": DEMO_CASES,
            **context,
        },
    )
    if viewer.should_set_guest_cookie:
        _set_guest_cookie(response, viewer.user_id)
    return response


def _download_text(packet: SavedJobPacket) -> str:
    lines = [render_packet_markdown(packet), "", "Exact Edits", ""]
    for patch in packet.report.exact_edits:
        lines.extend(
            [
                f"[{patch.section.upper()}]",
                f"Before: {patch.original_text}",
                f"After: {patch.rewritten_text}",
                f"Why: {patch.reason}",
                "",
            ]
        )
    return "\n".join(lines)


def _interview_session_lookup(packet: SavedJobPacket, session_id: str = "") -> InterviewSimulationSession | None:
    if session_id:
        for session in packet.interview_sessions:
            if session.session_id == session_id:
                return session
    return packet.interview_sessions[-1] if packet.interview_sessions else None


def _sorted_comparison_rows(comparison: MultiJDComparison, sort_key: str = "overall") -> list[dict[str, Any]]:
    key_map = {
        "overall": "score_overall",
        "ats": "score_ats",
        "evidence": "score_evidence",
        "level_fit": "score_level_fit",
    }
    attribute = key_map.get(sort_key, "score_overall")
    rows = sorted(comparison.rows, key=lambda item: getattr(item, attribute), reverse=True)
    return [
        {
            "packet_id": row.packet_id,
            "role_title": row.role_title,
            "company_name": row.company_name,
            "score_overall": row.score_overall,
            "score_ats": row.score_ats,
            "score_evidence": row.score_evidence,
            "score_level_fit": row.score_level_fit,
            "recommended_decision": row.recommended_decision,
            "top_gap_title": row.top_gap_title,
            "decision_meta": _decision_meta(row.recommended_decision),
        }
        for row in rows
    ]


def create_app(
    *,
    settings: Settings | None = None,
    runtime: AgentRuntime | None = None,
    optimizer: PromptOptimizer | None = None,
) -> FastAPI:
    settings = settings or get_settings()
    runtime = runtime or AgentRuntime(settings=settings)
    optimizer = optimizer or PromptOptimizer(settings=settings)
    auth_service = AuthService(settings=settings, tracker=runtime.service.tracker)
    app = FastAPI(title="Refine")
    app.state.settings = settings
    app.state.runtime = runtime
    app.state.optimizer = optimizer
    app.state.auth_service = auth_service

    @app.get("/auth/google/start")
    async def google_oauth_start(
        request: Request,
        next_path: str = "/history",
    ):
        viewer = _user_identity(request, auth_service)
        if viewer.authenticated:
            return RedirectResponse(url=_sanitize_next_path(next_path), status_code=303)
        if not auth_service.google_oauth_enabled:
            return RedirectResponse(url="/login?mode=signin", status_code=303)

        sanitized_next = _sanitize_next_path(next_path)
        state_token = auth_service.create_google_oauth_state_token(
            next_path=sanitized_next,
            guest_user_id=viewer.guest_user_id,
        )
        authorization_url = auth_service.build_google_oauth_authorize_url(
            redirect_uri=_google_redirect_uri(request, settings),
            state_token=state_token,
        )
        response = RedirectResponse(url=authorization_url, status_code=303)
        _set_google_state_cookie(request, response, state_token)
        if viewer.should_set_guest_cookie:
            _set_guest_cookie(response, viewer.user_id)
        return response

    @app.get("/auth/google/callback", name="google_oauth_callback")
    async def google_oauth_callback(
        request: Request,
        code: str = "",
        state: str = "",
        error: str = "",
    ):
        viewer = _user_identity(request, auth_service)
        if error:
            return _render(
                request,
                "login.html",
                viewer=viewer,
                context={
                    "active_item": "login",
                    "auth_mode": "signin",
                    "next_path": "/history",
                    "auth_error": f"Google sign-in failed: {error}",
                    "google_oauth_enabled": auth_service.google_oauth_enabled,
                },
            )
        cookie_state = request.cookies.get(GOOGLE_STATE_COOKIE_NAME)
        verified_state = auth_service.verify_google_oauth_state_token(cookie_state)
        if not state or not cookie_state or state != cookie_state or verified_state is None:
            response = _render(
                request,
                "login.html",
                viewer=viewer,
                context={
                    "active_item": "login",
                    "auth_mode": "signin",
                    "next_path": "/history",
                    "auth_error": "Google sign-in state could not be verified. Try again.",
                    "google_oauth_enabled": auth_service.google_oauth_enabled,
                },
            )
            _clear_google_state_cookie(response)
            return response
        if not code:
            response = _render(
                request,
                "login.html",
                viewer=viewer,
                context={
                    "active_item": "login",
                    "auth_mode": "signin",
                    "next_path": verified_state.get("next_path", "/history"),
                    "auth_error": "Google sign-in did not return an authorization code.",
                    "google_oauth_enabled": auth_service.google_oauth_enabled,
                },
            )
            _clear_google_state_cookie(response)
            return response
        try:
            user = await auth_service.authenticate_google_code(
                code=code,
                redirect_uri=_google_redirect_uri(request, settings),
                guest_user_id=verified_state.get("guest_user_id") or viewer.guest_user_id,
            )
        except AuthError as exc:
            response = _render(
                request,
                "login.html",
                viewer=viewer,
                context={
                    "active_item": "login",
                    "auth_mode": "signin",
                    "next_path": verified_state.get("next_path", "/history"),
                    "auth_error": str(exc),
                    "google_oauth_enabled": auth_service.google_oauth_enabled,
                },
            )
            _clear_google_state_cookie(response)
            return response

        response = RedirectResponse(url=_sanitize_next_path(verified_state.get("next_path")), status_code=303)
        _set_session_cookie(
            request,
            response,
            auth_service.create_session_token(user, ttl_seconds=COOKIE_MAX_AGE_SECONDS),
        )
        _clear_google_state_cookie(response)
        _clear_guest_cookie(response)
        return response

    @app.get("/login", response_class=HTMLResponse)
    async def login_page(
        request: Request,
        next_path: str = "/history",
        mode: str = "signin",
    ) -> HTMLResponse:
        viewer = _user_identity(request, auth_service)
        if viewer.authenticated:
            return RedirectResponse(url=_sanitize_next_path(next_path), status_code=303)
        return _render(
            request,
            "login.html",
            viewer=viewer,
            context={
                "active_item": "login",
                "auth_mode": "signup" if mode == "signup" else "signin",
                "next_path": _sanitize_next_path(next_path),
                "auth_error": "",
                "google_oauth_enabled": auth_service.google_oauth_enabled,
            },
        )

    @app.post("/login")
    async def login_submit(
        request: Request,
        email: str = Form(""),
        password: str = Form(""),
        next_path: str = Form("/history"),
    ):
        viewer = _user_identity(request, auth_service)
        try:
            user = auth_service.authenticate(
                email=email,
                password=password,
                guest_user_id=viewer.guest_user_id if not viewer.authenticated else None,
            )
        except AuthError as exc:
            return _render(
                request,
                "login.html",
                viewer=viewer,
                context={
                    "active_item": "login",
                    "auth_mode": "signin",
                    "next_path": _sanitize_next_path(next_path),
                    "auth_error": str(exc),
                    "google_oauth_enabled": auth_service.google_oauth_enabled,
                },
            )
        response = RedirectResponse(url=_sanitize_next_path(next_path), status_code=303)
        _set_session_cookie(
            request,
            response,
            auth_service.create_session_token(user, ttl_seconds=COOKIE_MAX_AGE_SECONDS),
        )
        _clear_guest_cookie(response)
        return response

    @app.post("/signup")
    async def signup_submit(
        request: Request,
        email: str = Form(""),
        password: str = Form(""),
        next_path: str = Form("/history"),
    ):
        viewer = _user_identity(request, auth_service)
        try:
            user = auth_service.register(
                email=email,
                password=password,
                guest_user_id=viewer.guest_user_id if not viewer.authenticated else None,
            )
        except AuthError as exc:
            return _render(
                request,
                "login.html",
                viewer=viewer,
                context={
                    "active_item": "login",
                    "auth_mode": "signup",
                    "next_path": _sanitize_next_path(next_path),
                    "auth_error": str(exc),
                    "google_oauth_enabled": auth_service.google_oauth_enabled,
                },
            )
        response = RedirectResponse(url=_sanitize_next_path(next_path), status_code=303)
        _set_session_cookie(
            request,
            response,
            auth_service.create_session_token(user, ttl_seconds=COOKIE_MAX_AGE_SECONDS),
        )
        _clear_guest_cookie(response)
        return response

    @app.post("/logout")
    async def logout_submit(request: Request):
        response = RedirectResponse(url="/", status_code=303)
        _clear_session_cookie(response)
        _clear_guest_cookie(response)
        return response

    @app.get("/", response_class=HTMLResponse)
    async def diagnose_page(
        request: Request,
        packet_id: str = "",
        demo_case: str = "",
    ) -> HTMLResponse:
        viewer = _user_identity(request, auth_service)
        packet = _get_user_packet(runtime, viewer.user_id, packet_id)
        return _render(
            request,
            "diagnose.html",
            viewer=viewer,
            context=_diagnose_context(runtime, packet=packet, demo_case=demo_case),
        )

    @app.post("/diagnose")
    async def run_diagnosis(
        request: Request,
        background_tasks: BackgroundTasks,
        jd_text: str = Form(""),
        rejection_notes: str = Form(""),
        demo_case: str = Form(""),
        resume: UploadFile | None = File(default=None),
    ):
        viewer = _user_identity(request, auth_service)
        temp_path: Path | None = None
        resume_path: Path | None = None
        resume_preview: dict[str, Any] | None = None
        effective_jd = jd_text
        effective_demo_case = demo_case.strip()
        resume_hint = resume.filename if resume is not None and resume.filename else ""
        try:
            if resume is not None and resume.filename:
                suffix = Path(resume.filename).suffix or ".txt"
                with NamedTemporaryFile(delete=False, suffix=suffix) as handle:
                    handle.write(await resume.read())
                    temp_path = Path(handle.name)
                resume_path = temp_path
                resume_preview = _build_resume_preview(resume_path, display_name=resume_hint)
            elif effective_demo_case:
                try:
                    resume_path, demo_jd = _load_demo_case(effective_demo_case)
                except ValueError as exc:
                    return _render(
                        request,
                        "diagnose.html",
                        viewer=viewer,
                        context=_diagnose_context(
                            runtime,
                            packet=None,
                            demo_case="",
                            jd_text=jd_text,
                            rejection_notes=rejection_notes,
                            error_message=str(exc),
                            resume_preview=resume_preview,
                        ),
                    )
                resume_preview = _build_resume_preview(resume_path)
                if not effective_jd.strip():
                    effective_jd = demo_jd

            if resume_path is None:
                return _render(
                    request,
                    "diagnose.html",
                    viewer=viewer,
                    context=_diagnose_context(
                        runtime,
                        packet=None,
                        demo_case=effective_demo_case,
                        jd_text=effective_jd,
                        rejection_notes=rejection_notes,
                        error_message="Upload a resume or select a demo case before running a diagnosis.",
                        resume_preview=resume_preview,
                    ),
                )
            if not effective_jd.strip():
                return _render(
                    request,
                    "diagnose.html",
                    viewer=viewer,
                    context=_diagnose_context(
                        runtime,
                        packet=None,
                        demo_case=effective_demo_case,
                        jd_text=effective_jd,
                        rejection_notes=rejection_notes,
                        error_message="Paste a job description before running a diagnosis.",
                        resume_hint=resume_hint,
                        resume_preview=resume_preview,
                    ),
                )

            try:
                result = await runtime.run_diagnostic_async(
                    resume_path=str(resume_path),
                    jd_text=effective_jd,
                    rejection_notes=rejection_notes,
                    user_id=viewer.user_id,
                )
                if resume_hint and result.get("packet") is not None:
                    result["packet"].resume_name = resume_hint
                    runtime.service.tracker.save(result["packet"])
            except Exception as exc:
                return _render(
                    request,
                    "diagnose.html",
                    viewer=viewer,
                    context=_diagnose_context(
                        runtime,
                        packet=None,
                        demo_case=effective_demo_case,
                        jd_text=effective_jd,
                        rejection_notes=rejection_notes,
                        error_message=f"Diagnosis failed: {exc}",
                        resume_hint=resume_hint,
                        resume_preview=resume_preview,
                    ),
                )
        finally:
            if temp_path is not None and temp_path.exists():
                temp_path.unlink(missing_ok=True)

        background_tasks.add_task(
            optimizer.record_successful_diagnosis,
            packet_id=str(result.get("packet_id", "")),
            session_id=str(result.get("session_id", "")),
        )
        response = RedirectResponse(url=f"/?packet_id={result['packet_id']}", status_code=303)
        response.background = background_tasks
        if viewer.should_set_guest_cookie:
            _set_guest_cookie(response, viewer.user_id)
        return response

    @app.get("/patch/{packet_id}", response_class=HTMLResponse)
    async def patch_page(request: Request, packet_id: str) -> HTMLResponse:
        viewer = _user_identity(request, auth_service)
        packet = _get_user_packet(runtime, viewer.user_id, packet_id)
        if packet is None:
            return _render(
                request,
                "diagnose.html",
                viewer=viewer,
                context=_diagnose_context(
                    runtime,
                    packet=None,
                    error_message="That analysis packet could not be found for this session.",
                ),
            )
        sections = _packet_sections(packet)
        return _render(
            request,
            "patch.html",
            viewer=viewer,
            context={
                "active_item": "diagnose",
                "packet": packet,
                "packet_meta": _decision_meta(packet.report.recommended_decision),
                "patch_sections": sections,
                "copy_blob": _download_text(packet),
            },
        )

    @app.get("/patch/{packet_id}/download")
    async def download_patch(request: Request, packet_id: str) -> PlainTextResponse:
        viewer = _user_identity(request, auth_service)
        packet = _get_user_packet(runtime, viewer.user_id, packet_id)
        if packet is None:
            return PlainTextResponse("Packet not found.", status_code=404)
        filename = f"refine-patch-{packet.packet_id[:8]}.txt"
        return PlainTextResponse(
            _download_text(packet),
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )

    @app.get("/resume/{packet_id}", response_class=HTMLResponse)
    async def rewritten_resume_page(request: Request, packet_id: str) -> HTMLResponse:
        viewer = _user_identity(request, auth_service)
        packet = _get_user_packet(runtime, viewer.user_id, packet_id)
        if packet is None or packet.report.rewritten_resume is None:
            return _render(
                request,
                "diagnose.html",
                viewer=viewer,
                context=_diagnose_context(
                    runtime,
                    packet=None,
                    error_message="The rewritten resume is not available for this packet.",
                ),
            )
        return _render(
            request,
            "resume.html",
            viewer=viewer,
            context={
                "active_item": "diagnose",
                "packet": packet,
                "packet_meta": _decision_meta(packet.report.recommended_decision),
                "rewritten_resume": packet.report.rewritten_resume,
            },
        )

    @app.get("/resume/{packet_id}/export.docx")
    async def rewritten_resume_docx(request: Request, packet_id: str) -> Response:
        viewer = _user_identity(request, auth_service)
        packet = _get_user_packet(runtime, viewer.user_id, packet_id)
        if packet is None or packet.report.rewritten_resume is None:
            return Response("Packet not found.", status_code=404)
        filename = f"refine-resume-{packet.packet_id[:8]}.docx"
        return Response(
            content=build_resume_docx_bytes(packet.report.rewritten_resume),
            media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )

    @app.get("/resume/{packet_id}/export.pdf")
    async def rewritten_resume_pdf(request: Request, packet_id: str) -> Response:
        viewer = _user_identity(request, auth_service)
        packet = _get_user_packet(runtime, viewer.user_id, packet_id)
        if packet is None or packet.report.rewritten_resume is None:
            return Response("Packet not found.", status_code=404)
        filename = f"refine-resume-{packet.packet_id[:8]}.pdf"
        return Response(
            content=build_resume_pdf_bytes(packet.report.rewritten_resume),
            media_type="application/pdf",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )

    @app.get("/plan/{packet_id}", response_class=HTMLResponse)
    async def plan_page(request: Request, packet_id: str) -> HTMLResponse:
        viewer = _user_identity(request, auth_service)
        packet = _get_user_packet(runtime, viewer.user_id, packet_id)
        if packet is None:
            return _render(
                request,
                "diagnose.html",
                viewer=viewer,
                context=_diagnose_context(
                    runtime,
                    packet=None,
                    error_message="That action plan is not available for this session.",
                ),
            )
        return _render(
            request,
            "plan.html",
            viewer=viewer,
            context={
                "active_item": "diagnose",
                "packet": packet,
                "packet_meta": _decision_meta(packet.report.recommended_decision),
                "plan_cards": _action_plan_cards(packet),
                "learning_cards": _learning_cards(packet),
                "interview_columns": _interview_columns(packet),
            },
        )

    @app.get("/interview/{packet_id}", response_class=HTMLResponse)
    async def interview_page(request: Request, packet_id: str, session_id: str = "") -> HTMLResponse:
        viewer = _user_identity(request, auth_service)
        packet = _get_user_packet(runtime, viewer.user_id, packet_id)
        if packet is None:
            return _render(
                request,
                "diagnose.html",
                viewer=viewer,
                context=_diagnose_context(
                    runtime,
                    packet=None,
                    error_message="That interview simulator packet is not available for this session.",
                ),
            )
        session = _interview_session_lookup(packet, session_id)
        current_question = ""
        if session is not None and session.status == "in_progress":
            current_question = session.questions[session.current_index]
        return _render(
            request,
            "interview.html",
            viewer=viewer,
            context={
                "active_item": "diagnose",
                "packet": packet,
                "packet_meta": _decision_meta(packet.report.recommended_decision),
                "session": session,
                "session_overview": session_overview(packet.report, session),
                "current_question": current_question,
            },
        )

    @app.post("/interview/{packet_id}/start")
    async def interview_start(request: Request, packet_id: str):
        viewer = _user_identity(request, auth_service)
        result = runtime.service.create_interview_session(packet_id=packet_id, user_id=viewer.user_id)
        if result is None:
            return RedirectResponse(url=f"/interview/{packet_id}", status_code=303)
        _, session = result
        return RedirectResponse(url=f"/interview/{packet_id}?session_id={session.session_id}", status_code=303)

    @app.post("/interview/{packet_id}/answer")
    async def interview_answer(
        request: Request,
        packet_id: str,
        session_id: str = Form(""),
        answer: str = Form(""),
    ):
        viewer = _user_identity(request, auth_service)
        if not answer.strip():
            return RedirectResponse(url=f"/interview/{packet_id}?session_id={session_id}", status_code=303)
        result = runtime.service.submit_interview_answer(
            packet_id=packet_id,
            session_id=session_id,
            user_id=viewer.user_id,
            answer=answer,
        )
        if result is None:
            return RedirectResponse(url=f"/interview/{packet_id}", status_code=303)
        _, session = result
        return RedirectResponse(url=f"/interview/{packet_id}?session_id={session.session_id}", status_code=303)

    @app.get("/compare", response_class=HTMLResponse)
    async def compare_page(request: Request) -> HTMLResponse:
        viewer = _user_identity(request, auth_service)
        return _render(
            request,
            "compare.html",
            viewer=viewer,
            context={
                "active_item": "compare",
                "error_message": "",
                "form_state": {"rejection_notes": "", "jd_texts": ["", "", "", "", ""]},
            },
        )

    @app.post("/compare")
    async def run_comparison(
        request: Request,
        rejection_notes: str = Form(""),
        jd_1: str = Form(""),
        jd_2: str = Form(""),
        jd_3: str = Form(""),
        jd_4: str = Form(""),
        jd_5: str = Form(""),
        resume: UploadFile | None = File(default=None),
    ):
        viewer = _user_identity(request, auth_service)
        jd_texts = [jd_1, jd_2, jd_3, jd_4, jd_5]
        if resume is None or not resume.filename:
            return _render(
                request,
                "compare.html",
                viewer=viewer,
                context={
                    "active_item": "compare",
                    "error_message": "Upload one resume before running a multi-JD comparison.",
                    "form_state": {"rejection_notes": rejection_notes, "jd_texts": jd_texts},
                },
            )
        non_empty = [item for item in jd_texts if item.strip()]
        if len(non_empty) < 2:
            return _render(
                request,
                "compare.html",
                viewer=viewer,
                context={
                    "active_item": "compare",
                    "error_message": "Paste at least two job descriptions to compare fit across roles.",
                    "form_state": {"rejection_notes": rejection_notes, "jd_texts": jd_texts},
                },
            )
        temp_path: Path | None = None
        try:
            suffix = Path(resume.filename).suffix or ".txt"
            with NamedTemporaryFile(delete=False, suffix=suffix) as handle:
                handle.write(await resume.read())
                temp_path = Path(handle.name)
            comparison = runtime.service.compare_job_descriptions(
                resume_path=str(temp_path),
                jd_texts=non_empty,
                rejection_notes=rejection_notes,
                user_id=viewer.user_id,
            )
        finally:
            if temp_path is not None and temp_path.exists():
                temp_path.unlink(missing_ok=True)
        return RedirectResponse(url=f"/compare/{comparison.comparison_id}", status_code=303)

    @app.get("/compare/{comparison_id}", response_class=HTMLResponse)
    async def comparison_results_page(request: Request, comparison_id: str, sort: str = "overall") -> HTMLResponse:
        viewer = _user_identity(request, auth_service)
        comparison = runtime.service.tracker.get_comparison(comparison_id)
        if comparison is None or comparison.user_id != viewer.user_id:
            return _render(
                request,
                "compare.html",
                viewer=viewer,
                context={
                    "active_item": "compare",
                    "error_message": "That comparison bundle could not be found for this session.",
                    "form_state": {"rejection_notes": "", "jd_texts": ["", "", "", "", ""]},
                },
            )
        return _render(
            request,
            "compare_results.html",
            viewer=viewer,
            context={
                "active_item": "compare",
                "comparison": comparison,
                "rows": _sorted_comparison_rows(comparison, sort),
                "sort_order": sort,
            },
        )

    @app.get("/history", response_class=HTMLResponse)
    async def history_page(
        request: Request,
        packet_id: str = "",
        status: str = "all",
        sort: str = "newest",
    ) -> HTMLResponse:
        viewer = _user_identity(request, auth_service)
        entries = _history_entries(runtime.service.tracker.list_entries(viewer.user_id), status_filter=status, sort_order=sort)
        selected_packet = _get_user_packet(runtime, viewer.user_id, packet_id)
        if selected_packet is None and entries:
            selected_packet = _get_user_packet(runtime, viewer.user_id, entries[0].packet_id)
        entry_packets = {
            entry.packet_id: _get_user_packet(runtime, viewer.user_id, entry.packet_id)
            for entry in entries
        }
        return _render(
            request,
            "history.html",
            viewer=viewer,
            context={
                "active_item": "history",
                "entries": entries,
                "selected_packet": selected_packet,
                "status_filter": status,
                "sort_order": sort,
                "entry_packets": entry_packets,
                "decision_meta": _decision_meta,
                "company_initials": _company_initials,
                "history_persistent": viewer.authenticated,
            },
        )

    @app.get("/settings", response_class=HTMLResponse)
    async def settings_page(request: Request) -> HTMLResponse:
        viewer = _user_identity(request, auth_service)
        improvement_snapshot = optimizer.latest_snapshot()
        return _render(
            request,
            "settings.html",
            viewer=viewer,
            context={
                "active_item": "settings",
                "improvement_run": improvement_snapshot["improvement_run"],
                "candidate_prompt": improvement_snapshot["candidate_prompt"],
                "improvement_status": improvement_snapshot,
                "runtime_status": {
                    "adk_enabled": runtime.adk_available(),
                    "model_chain": runtime.settings.generation_model_candidates,
                    "phoenix_project": runtime.settings.phoenix_project_name,
                    "collector": runtime.settings.phoenix_collector_endpoint,
                },
            },
        )

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    return app


app = create_app()
