"""ADK root agent facade with deterministic fallback."""

from __future__ import annotations

import asyncio
from contextlib import nullcontext
import hashlib
from pathlib import Path
from typing import Any
import uuid

from job_rejection_agent.config import Settings, get_settings
from job_rejection_agent.google_models import is_resource_exhausted_error
from job_rejection_agent.observability import (
    configure_tracing,
    evaluate_packet,
    format_span_id,
    format_trace_id,
)
from job_rejection_agent.observability.phoenix_mcp import build_phoenix_mcp_toolset
from job_rejection_agent.persistence import create_session_service
from job_rejection_agent.services import DiagnosticService, render_packet_markdown

from .subagents import compose_specialist_context


class AgentRuntime:
    def __init__(self, settings: Settings | None = None, service: DiagnosticService | None = None) -> None:
        self.settings = settings or get_settings()
        self.service = service or DiagnosticService(self.settings)

    def _toolkit(self):
        service = self.service

        def parse_resume_tool(file_path: str) -> dict[str, Any]:
            """Parse a resume and return structured facts."""
            return service.diagnose(resume_path=file_path, jd_text="Software engineer", persist=False).packet.resume_facts.to_dict()

        def analyze_job_description_tool(jd_text: str) -> dict[str, Any]:
            """Parse a job description into structured requirements."""
            from job_rejection_agent.ingestion import parse_job_description
            from job_rejection_agent.analysis import extract_job_requirements

            return extract_job_requirements(parse_job_description(jd_text)).to_dict()

        def diagnose_job_rejection_tool(
            resume_path: str,
            jd_text: str,
            rejection_notes: str = "",
            user_id: str = "anonymous",
            tool_context: Any | None = None,
        ) -> dict[str, Any]:
            """Run the full rejection diagnosis pipeline and return a saved packet."""
            session_id = getattr(getattr(tool_context, "session", None), "id", None)
            effective_user_id = getattr(tool_context, "user_id", None) or user_id
            result = service.diagnose(
                resume_path=resume_path,
                jd_text=jd_text,
                rejection_notes=rejection_notes,
                user_id=effective_user_id,
                session_id=session_id,
                persist=True,
            )
            return {
                "packet_id": result.packet.packet_id,
                "session_id": result.packet.session_id,
                "report": result.packet.report.to_dict(),
                "markdown": render_packet_markdown(result.packet),
            }

        def list_saved_packets_tool(user_id: str = "anonymous") -> list[dict[str, Any]]:
            """List saved job packets for a user."""
            return [entry.to_dict() for entry in service.tracker.list_entries(user_id)]

        tools: list[Any] = [
            parse_resume_tool,
            analyze_job_description_tool,
            diagnose_job_rejection_tool,
            list_saved_packets_tool,
        ]
        phoenix_toolset = build_phoenix_mcp_toolset(self.settings)
        if phoenix_toolset is not None:
            tools.append(phoenix_toolset)
        return tools

    def adk_available(self) -> bool:
        try:
            import google.adk  # noqa: F401
            import google.genai  # noqa: F401
            return bool(self.settings.google_api_key and self.settings.generation_model_candidates)
        except Exception:
            return False

    def build_agent(self, *, prompt_text_override: str | None = None, model_override: str | None = None):
        from google.adk.agents import LlmAgent

        prompt = prompt_text_override or Path(self.settings.prompt_path).read_text(encoding="utf-8")
        instruction = prompt.rstrip() + "\n\nSpecialist guidance:\n" + compose_specialist_context()
        return LlmAgent(
            model=model_override or self.settings.model_id,
            name="JobRejectionCoach",
            description="Diagnoses job rejections and generates exact patch plans.",
            instruction=instruction,
            tools=self._toolkit(),
        )

    def _record_request_attributes(
        self,
        span: Any,
        *,
        session_id: str,
        user_id: str,
        resume_path: str,
        jd_text: str,
        rejection_notes: str,
        prompt_text_override: str | None = None,
        model_candidates: tuple[str, ...] = (),
    ) -> None:
        span.set_attribute("session.id", session_id)
        span.set_attribute("user.id", user_id)
        span.set_attribute("job_rejection.session_id", session_id)
        span.set_attribute("job_rejection.user_id", user_id)
        span.set_attribute("job_rejection.resume_name", Path(resume_path).name)
        span.set_attribute("job_rejection.jd_length", len(jd_text))
        span.set_attribute("job_rejection.has_rejection_notes", bool(rejection_notes.strip()))
        span.set_attribute("job_rejection.prompt_source", "override" if prompt_text_override else "default")
        if model_candidates:
            span.set_attribute("job_rejection.model_candidates", " | ".join(model_candidates))
        if prompt_text_override:
            span.set_attribute(
                "job_rejection.prompt_fingerprint",
                hashlib.sha256(prompt_text_override.encode("utf-8")).hexdigest()[:16],
            )

    async def _reset_adk_session(
        self,
        *,
        session_service: Any,
        user_id: str,
        session_id: str,
    ) -> None:
        existing = await session_service.get_session(
            app_name=self.settings.app_name,
            user_id=user_id,
            session_id=session_id,
        )
        if existing is not None:
            await session_service.delete_session(
                app_name=self.settings.app_name,
                user_id=user_id,
                session_id=session_id,
            )
        await session_service.create_session(
            app_name=self.settings.app_name,
            user_id=user_id,
            session_id=session_id,
        )

    async def _run_adk_attempt(
        self,
        *,
        model_id: str,
        session_service: Any,
        session_id: str,
        user_id: str,
        user_message: Any,
        prompt_text_override: str | None = None,
    ) -> str:
        from google.adk.runners import Runner

        runner = Runner(
            app_name=self.settings.app_name,
            agent=self.build_agent(prompt_text_override=prompt_text_override, model_override=model_id),
            session_service=session_service,
        )
        final_text = ""
        async for event in runner.run_async(
            user_id=user_id,
            session_id=session_id,
            new_message=user_message,
        ):
            if hasattr(event, "is_final_response") and event.is_final_response():
                final_text = getattr(event, "stringify_content", lambda: "")() or final_text
        return final_text

    def _record_result_attributes(self, span: Any, *, packet: Any, output_text: str) -> None:
        span.set_attribute("job_rejection.packet_id", packet.packet_id)
        span.set_attribute("job_rejection.role_title", packet.job_requirements.role_title)
        span.set_attribute("job_rejection.company_name", packet.job_requirements.company_name)
        span.set_attribute("job_rejection.recommended_decision", packet.report.recommended_decision)
        span.set_attribute("job_rejection.score_overall", packet.report.score_overall)
        span.set_attribute(
            "job_rejection.top_gap_titles",
            " | ".join(gap.title for gap in packet.report.top_gaps[:3]),
        )
        span.set_attribute("job_rejection.output_preview", output_text[:600])

    async def run_diagnostic_async(
        self,
        *,
        resume_path: str,
        jd_text: str,
        rejection_notes: str = "",
        user_id: str = "anonymous",
        prompt_text_override: str | None = None,
    ) -> dict[str, Any]:
        session_id = str(uuid.uuid4())
        root_span_id = None
        trace_ids: dict[str, str] = {}
        tracing_enabled = configure_tracing(self.settings)
        model_candidates = self.settings.generation_model_candidates
        span_context_manager = nullcontext(None)
        if tracing_enabled:
            from opentelemetry import trace as otel_trace

            tracer = otel_trace.get_tracer(self.settings.app_name)
            span_context_manager = tracer.start_as_current_span("job_rejection_session")

        packet = None
        final_text = ""
        used_adk = self.adk_available()
        with span_context_manager as root_span:
            if root_span is not None:
                self._record_request_attributes(
                    root_span,
                    session_id=session_id,
                    user_id=user_id,
                    resume_path=resume_path,
                    jd_text=jd_text,
                    rejection_notes=rejection_notes,
                    prompt_text_override=prompt_text_override,
                    model_candidates=model_candidates,
                )
                span_context = root_span.get_span_context()
                root_span_id = format_span_id(span_context.span_id)
                trace_ids = {
                    "trace_id": format_trace_id(span_context.trace_id),
                    "root_span_id": root_span_id,
                }

            if not used_adk:
                result = self.service.diagnose(
                    resume_path=resume_path,
                    jd_text=jd_text,
                    rejection_notes=rejection_notes,
                    user_id=user_id,
                    session_id=session_id,
                    persist=True,
                )
                packet = result.packet
                final_text = render_packet_markdown(result.packet)
            else:
                from google.genai import types

                session_service = create_session_service(self.settings)
                user_message = types.Content(
                    role="user",
                    parts=[
                        types.Part(
                            text=(
                                f"User ID: {user_id}\n"
                                f"Resume path: {resume_path}\n"
                                f"Job description:\n{jd_text}\n\n"
                                f"Rejection notes:\n{rejection_notes}\n\n"
                                "Use the available tools to diagnose the rejection and produce the required format."
                            )
                        )
                    ],
                )
                selected_model = ""
                last_model_error: Exception | None = None
                for model_id in model_candidates:
                    try:
                        await self._reset_adk_session(
                            session_service=session_service,
                            user_id=user_id,
                            session_id=session_id,
                        )
                        final_text = await self._run_adk_attempt(
                            model_id=model_id,
                            session_service=session_service,
                            session_id=session_id,
                            user_id=user_id,
                            user_message=user_message,
                            prompt_text_override=prompt_text_override,
                        )
                        selected_model = model_id
                        break
                    except Exception as exc:
                        if is_resource_exhausted_error(exc):
                            last_model_error = exc
                            continue
                        raise

                if selected_model:
                    if root_span is not None:
                        root_span.set_attribute("job_rejection.selected_model", selected_model)
                    packet_candidates = self.service.tracker.list_entries(user_id)
                    packet_id = packet_candidates[0].packet_id if packet_candidates else ""
                    packet = self.service.tracker.get(packet_id) if packet_id else None
                else:
                    if root_span is not None:
                        root_span.set_attribute("job_rejection.selected_model", "deterministic-fallback")
                        root_span.set_attribute("job_rejection.adk_fallback_reason", "resource_exhausted")
                    used_adk = False
                    result = self.service.diagnose(
                        resume_path=resume_path,
                        jd_text=jd_text,
                        rejection_notes=rejection_notes,
                        user_id=user_id,
                        session_id=session_id,
                        persist=True,
                    )
                    packet = result.packet
                    final_text = render_packet_markdown(result.packet)
                    if last_model_error is not None and root_span is not None:
                        root_span.set_attribute("job_rejection.last_model_error", str(last_model_error)[:600])

            if root_span is not None and packet is not None:
                self._record_result_attributes(root_span, packet=packet, output_text=final_text)

        eval_scores = evaluate_packet(
            packet,
            span_id=root_span_id,
            output_text=final_text,
            settings=self.settings,
        ) if packet else {}
        return {
            "used_adk": used_adk,
            "session_id": packet.session_id if packet else session_id,
            "packet_id": packet.packet_id if packet else "",
            "text": final_text or (render_packet_markdown(packet) if packet else ""),
            "eval_scores": eval_scores,
            "packet": packet,
            **trace_ids,
        }

    def run_diagnostic(self, **kwargs: Any) -> dict[str, Any]:
        return asyncio.run(self.run_diagnostic_async(**kwargs))


def build_root_agent(settings: Settings | None = None):
    return AgentRuntime(settings=settings).build_agent()
