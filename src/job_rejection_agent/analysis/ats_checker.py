"""Deterministic ATS readiness checks."""

from __future__ import annotations

from job_rejection_agent.domain import ATSCheckResult, JobRequirements, ResumeFacts


def _status_rank(status: str) -> int:
    return {"fail": 0, "warn": 1, "pass": 2}[status]


def evaluate_ats_checks(resume_facts: ResumeFacts, requirements: JobRequirements) -> list[ATSCheckResult]:
    checks: list[ATSCheckResult] = []
    sections = resume_facts.section_map
    section_score = sum(1 for name in ("experience", "projects", "skills", "education") if sections.get(name))
    if section_score >= 3:
        section_check = ATSCheckResult(
            category="section_structure",
            status="pass",
            title="Core resume sections are present",
            details="The parser found clear section boundaries for most major resume sections.",
            recommendation="Keep section headings plain and recruiter-readable.",
        )
    elif section_score >= 2:
        section_check = ATSCheckResult(
            category="section_structure",
            status="warn",
            title="Section structure is only partially clear",
            details="Some expected sections are missing or may not be clearly labeled for ATS parsing.",
            recommendation="Use standard headings like Summary, Experience, Projects, Skills, and Education.",
        )
    else:
        section_check = ATSCheckResult(
            category="section_structure",
            status="fail",
            title="Section structure is weak",
            details="The parser could not confidently detect enough standard resume sections.",
            recommendation="Reformat the resume with explicit standard headings and distinct section blocks.",
        )
    checks.append(section_check)

    missing_contacts = []
    if not resume_facts.contact_signals.get("email"):
        missing_contacts.append("email")
    if not resume_facts.contact_signals.get("phone"):
        missing_contacts.append("phone")
    if missing_contacts:
        contact_status = "fail" if "email" in missing_contacts else "warn"
        checks.append(
            ATSCheckResult(
                category="contact_info",
                status=contact_status,
                title="Contact information is incomplete",
                details=f"Missing or unclear contact signals: {', '.join(missing_contacts)}.",
                recommendation="Place your full contact block in the header with email and phone on separate, readable tokens.",
            )
        )
    elif not resume_facts.contact_signals.get("linkedin"):
        checks.append(
            ATSCheckResult(
                category="contact_info",
                status="warn",
                title="LinkedIn signal is missing",
                details="A LinkedIn URL was not detected in the header area.",
                recommendation="Add a clean LinkedIn URL near your email and phone if you actively maintain it.",
            )
        )
    else:
        checks.append(
            ATSCheckResult(
                category="contact_info",
                status="pass",
                title="Contact information is ATS-readable",
                details="Email, phone, and a professional profile signal are present.",
                recommendation="Keep the contact header plain and text-based.",
            )
        )

    long_findings = [finding for finding in resume_facts.ats_findings if "long" in finding.lower() or "few explicit bullets" in finding.lower()]
    if long_findings or not resume_facts.metrics:
        checks.append(
            ATSCheckResult(
                category="readability",
                status="warn",
                title="Bullet readability can be stronger",
                details=long_findings[0] if long_findings else "The resume lacks measurable outcomes, which weakens scannability and recruiter confidence.",
                recommendation="Tighten long bullets and add verified outcome metrics to the strongest lines.",
            )
        )
    else:
        checks.append(
            ATSCheckResult(
                category="readability",
                status="pass",
                title="Bullets read clearly",
                details="The resume contains metric-bearing content and does not trigger major readability heuristics.",
                recommendation="Preserve short, high-signal bullets with clear outcomes.",
            )
        )

    formatting_findings = [
        finding for finding in resume_facts.ats_findings
        if any(token in finding.lower() for token in ("table", "decorative", "formatting", "parsed text"))
    ]
    if formatting_findings:
        checks.append(
            ATSCheckResult(
                category="formatting_risk",
                status="warn",
                title="Formatting choices may confuse ATS parsing",
                details=formatting_findings[0],
                recommendation="Avoid tables, decorative bullets, or dense layouts that flatten poorly in text extraction.",
            )
        )
    else:
        checks.append(
            ATSCheckResult(
                category="formatting_risk",
                status="pass",
                title="Formatting risk appears low",
                details="No major formatting heuristics were triggered by the parser.",
                recommendation="Keep the document single-column and text-first.",
            )
        )

    required_count = max(1, len(requirements.required_skills))
    missing_count = len([skill for skill in requirements.required_skills if skill not in resume_facts.skills])
    missing_ratio = missing_count / required_count
    if missing_ratio >= 0.5:
        keyword_status = "fail"
        keyword_details = "More than half of the job's required skills are missing from the resume text."
    elif missing_ratio >= 0.25:
        keyword_status = "warn"
        keyword_details = "Several required skills are absent or not explicit enough in the resume text."
    else:
        keyword_status = "pass"
        keyword_details = "Most required keywords are present or inferable from the resume."
    checks.append(
        ATSCheckResult(
            category="keyword_coverage",
            status=keyword_status,
            title="Keyword coverage",
            details=keyword_details,
            recommendation="Only add missing keywords if you can back them with real project, coursework, or internship evidence.",
        )
    )

    if resume_facts.source_file_type not in {"pdf", "docx", "txt", "md"}:
        hygiene_status = "fail"
        hygiene_details = f"Unsupported source format: {resume_facts.source_file_type}."
    elif len(resume_facts.normalized_text.split()) < 120:
        hygiene_status = "warn"
        hygiene_details = "Very little text was extracted from the uploaded resume, which may indicate formatting or content sparsity."
    else:
        hygiene_status = "pass"
        hygiene_details = f"The uploaded {resume_facts.source_file_type.upper()} file produced a stable text extraction."
    checks.append(
        ATSCheckResult(
            category="file_hygiene",
            status=hygiene_status,
            title="File hygiene",
            details=hygiene_details,
            recommendation="Use a text-readable PDF or DOCX and verify the exported text before applying widely.",
        )
    )

    return sorted(checks, key=lambda item: (_status_rank(item.status), item.category))
