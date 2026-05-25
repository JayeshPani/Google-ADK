"""Seed saved job packets from the bundled demo fixtures."""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from job_rejection_agent.services import DiagnosticService
CASES = [
    ("nisha_ml_newgrad.txt", "ml_platform_engineer.md", "Recruiter asked for stronger production evidence."),
    ("arjun_backend_student.txt", "backend_newgrad.md", "Resume looked too project-heavy and light on APIs."),
    ("meera_data_analyst.txt", "data_analyst_rotational.md", "Feedback suggested stronger SQL storytelling was needed."),
    ("rahul_fullstack_intern.txt", "ai_products_intern.md", "The role wanted more visible LLM or agent work."),
]


def main() -> None:
    service = DiagnosticService()
    for resume_name, jd_name, note in CASES:
        resume_path = ROOT / "tests" / "fixtures" / "resumes" / resume_name
        jd_path = ROOT / "tests" / "fixtures" / "jds" / jd_name
        result = service.diagnose(
            resume_path=resume_path,
            jd_text=jd_path.read_text(encoding="utf-8"),
            rejection_notes=note,
            user_id="demo-seed",
            persist=True,
        )
        print(f"Seeded {result.packet.packet_id}: {result.packet.job_requirements.company_name} / {result.packet.job_requirements.role_title}")


if __name__ == "__main__":
    main()
