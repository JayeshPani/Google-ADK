"""Run a deterministic end-to-end diagnostic against a seeded fixture."""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from job_rejection_agent.services import DiagnosticService, render_packet_markdown, summarise_packet


def main() -> None:
    service = DiagnosticService()
    resume_path = ROOT / "tests" / "fixtures" / "resumes" / "nisha_ml_newgrad.txt"
    jd_path = ROOT / "tests" / "fixtures" / "jds" / "ml_platform_engineer.md"
    result = service.diagnose(
        resume_path=resume_path,
        jd_text=jd_path.read_text(encoding="utf-8"),
        rejection_notes="Recruiter said the profile felt promising but not yet production-ready.",
        persist=False,
    )
    print("Summary")
    print(summarise_packet(result.packet))
    print("\nReport\n")
    print(render_packet_markdown(result.packet))


if __name__ == "__main__":
    main()
