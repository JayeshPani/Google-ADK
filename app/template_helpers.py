from __future__ import annotations

from markupsafe import Markup, escape


_STROKE_ICON_BODIES: dict[str, str] = {
    "lucide:layers": """
        <path d="M12 3 2 8l10 5 10-5-10-5Z"/>
        <path d="m2 13 10 5 10-5"/>
        <path d="m2 18 10 5 10-5"/>
    """,
    "lucide:sun": """
        <path d="M12 3v2.2"/>
        <path d="M12 18.8V21"/>
        <path d="m4.9 4.9 1.6 1.6"/>
        <path d="m17.5 17.5 1.6 1.6"/>
        <path d="M3 12h2.2"/>
        <path d="M18.8 12H21"/>
        <path d="m4.9 19.1 1.6-1.6"/>
        <path d="m17.5 6.5 1.6-1.6"/>
        <circle cx="12" cy="12" r="3.5"/>
    """,
    "lucide:moon": """
        <path d="M21 12.8A8.5 8.5 0 1 1 11.2 3a6.5 6.5 0 0 0 9.8 9.8Z"/>
    """,
    "lucide:upload-cloud": """
        <path d="M6 18a4 4 0 1 1 .6-7.96A6 6 0 0 1 18 9a4 4 0 1 1 0 8H6Z"/>
        <path d="M12 12v7"/>
        <path d="m8.5 15.5 3.5-3.5 3.5 3.5"/>
    """,
    "lucide:alert-circle": """
        <circle cx="12" cy="12" r="9"/>
        <path d="M12 8v5"/>
        <path d="M12 16h.01"/>
    """,
    "lucide:alert-triangle": """
        <path d="M10.3 3.8 1.8 18A2 2 0 0 0 3.5 21h17a2 2 0 0 0 1.7-3L13.7 3.8a2 2 0 0 0-3.4 0Z"/>
        <path d="M12 9v4"/>
        <path d="M12 17h.01"/>
    """,
    "lucide:arrow-right": """
        <path d="M5 12h14"/>
        <path d="m13 5 7 7-7 7"/>
    """,
    "lucide:chevron-right": """
        <path d="m9 18 6-6-6-6"/>
    """,
    "lucide:file-check": """
        <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8Z"/>
        <path d="M14 2v6h6"/>
        <path d="m9 15 2 2 4-4"/>
    """,
    "lucide:file-text": """
        <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8Z"/>
        <path d="M14 2v6h6"/>
        <path d="M8 13h8"/>
        <path d="M8 17h8"/>
        <path d="M8 9h3"/>
    """,
    "lucide:info": """
        <circle cx="12" cy="12" r="9"/>
        <path d="M12 10v6"/>
        <path d="M12 7h.01"/>
    """,
    "lucide:x-circle": """
        <circle cx="12" cy="12" r="9"/>
        <path d="m15 9-6 6"/>
        <path d="m9 9 6 6"/>
    """,
    "lucide:arrow-up-right": """
        <path d="M7 17 17 7"/>
        <path d="M7 7h10v10"/>
    """,
    "lucide:calendar": """
        <path d="M8 2v4"/>
        <path d="M16 2v4"/>
        <rect x="3" y="4" width="18" height="17" rx="2"/>
        <path d="M3 10h18"/>
    """,
    "lucide:chevron-down": """
        <path d="m6 9 6 6 6-6"/>
    """,
    "lucide:filter": """
        <path d="M4 6h16"/>
        <path d="M7 12h10"/>
        <path d="M10 18h4"/>
    """,
    "lucide:play-circle": """
        <circle cx="12" cy="12" r="9"/>
        <path d="m10 8 6 4-6 4Z"/>
    """,
    "lucide:check": """
        <path d="m5 12 5 5L20 7"/>
    """,
    "lucide:check-circle": """
        <circle cx="12" cy="12" r="9"/>
        <path d="m8.5 12.5 2.5 2.5 4.5-5"/>
    """,
    "lucide:copy": """
        <rect x="9" y="9" width="10" height="10" rx="2"/>
        <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/>
    """,
    "lucide:download": """
        <path d="M12 3v12"/>
        <path d="m7 10 5 5 5-5"/>
        <path d="M4 21h16"/>
    """,
    "lucide:help-circle": """
        <circle cx="12" cy="12" r="9"/>
        <path d="M9.1 9a3 3 0 1 1 5.8 1c0 2-3 2-3 4"/>
        <path d="M12 17h.01"/>
    """,
    "lucide:trash-2": """
        <path d="M3 6h18"/>
        <path d="M8 6V4h8v2"/>
        <path d="M19 6l-1 14H6L5 6"/>
        <path d="M10 11v6"/>
        <path d="M14 11v6"/>
    """,
    "lucide:bookmark": """
        <path d="M7 3h10a2 2 0 0 1 2 2v16l-7-4-7 4V5a2 2 0 0 1 2-2Z"/>
    """,
    "lucide:clock": """
        <circle cx="12" cy="12" r="9"/>
        <path d="M12 7v5l3 2"/>
    """,
    "lucide:file-pen-line": """
        <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8Z"/>
        <path d="M14 2v6h6"/>
        <path d="m10 17 5.5-5.5 2 2L12 19H10Z"/>
    """,
    "lucide:mic": """
        <rect x="9" y="3" width="6" height="11" rx="3"/>
        <path d="M5 11a7 7 0 0 0 14 0"/>
        <path d="M12 18v3"/>
        <path d="M9 21h6"/>
    """,
    "lucide:file-down": """
        <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8Z"/>
        <path d="M14 2v6h6"/>
        <path d="M12 11v6"/>
        <path d="m9.5 14.5 2.5 2.5 2.5-2.5"/>
    """,
    "lucide:file-output": """
        <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8Z"/>
        <path d="M14 2v6h6"/>
        <path d="M8 15h6"/>
        <path d="m12 12 3 3-3 3"/>
    """,
    "lucide:component": """
        <rect x="3" y="3" width="7" height="7" rx="1.5"/>
        <rect x="14" y="3" width="7" height="7" rx="1.5"/>
        <rect x="3" y="14" width="7" height="7" rx="1.5"/>
        <path d="M14 17h7"/>
        <path d="M17.5 14v7"/>
    """,
    "lucide:mouse-pointer-2": """
        <path d="m4 3 7.5 16 1.7-6 6-1.7L4 3Z"/>
        <path d="m12 12 4 4"/>
    """,
    "lucide:search": """
        <circle cx="11" cy="11" r="7"/>
        <path d="m20 20-3.5-3.5"/>
    """,
    "lucide:search-check": """
        <circle cx="11" cy="11" r="7"/>
        <path d="m20 20-3.5-3.5"/>
        <path d="m8.5 11.5 1.5 1.5 3.5-4"/>
    """,
    "lucide:scale": """
        <path d="m16 16 3-8 3 8c-.9.7-1.8 1-3 1s-2.1-.3-3-1Z"/>
        <path d="m2 16 3-8 3 8c-.9.7-1.8 1-3 1s-2.1-.3-3-1Z"/>
        <path d="M7 21h10"/>
        <path d="M12 3v18"/>
        <path d="M3 8h18"/>
    """,
    "lucide:map": """
        <path d="M14.5 5.5 9.5 3 3 6.5v14l6.5-3.5 5 2.5 6.5-3.5v-14l-6.5 3.5Z"/>
        <path d="M9.5 3v14"/>
        <path d="M14.5 5.5v14"/>
    """,
    "lucide:messages-square": """
        <path d="M21 14a2 2 0 0 1-2 2H8l-5 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2Z"/>
        <path d="M8 9h8"/>
        <path d="M8 12h5"/>
    """,
    "lucide:bookmark-check": """
        <path d="M7 3h10a2 2 0 0 1 2 2v16l-7-4-7 4V5a2 2 0 0 1 2-2Z"/>
        <path d="m9 10 2 2 4-4"/>
    """,
    "lucide:bar-chart-2": """
        <path d="M4 20V10"/>
        <path d="M10 20V4"/>
        <path d="M16 20v-7"/>
        <path d="M22 20v-3"/>
    """,
    "lucide:server": """
        <rect x="3" y="4" width="18" height="6" rx="2"/>
        <rect x="3" y="14" width="18" height="6" rx="2"/>
        <path d="M7 7h.01"/>
        <path d="M7 17h.01"/>
    """,
    "lucide:sparkles": """
        <path d="m12 3 1.7 4.3L18 9l-4.3 1.7L12 15l-1.7-4.3L6 9l4.3-1.7L12 3Z"/>
        <path d="m5 16 .8 2.2L8 19l-2.2.8L5 22l-.8-2.2L2 19l2.2-.8L5 16Z"/>
        <path d="m19 14 1 2.5L22.5 17 20 18l-1 2.5L18 18l-2.5-1 2.5-.5L19 14Z"/>
    """,
}

_FILL_ICON_BODIES: dict[str, tuple[str, str]] = {
    "logos:google-icon": (
        "0 0 24 24",
        """
            <path fill="#4285F4" d="M21.6 12.23c0-.64-.06-1.25-.16-1.83H12v3.46h5.39a4.6 4.6 0 0 1-2 3.02v2.5h3.24c1.9-1.74 2.97-4.31 2.97-7.15Z"/>
            <path fill="#34A853" d="M12 22c2.7 0 4.97-.9 6.63-2.43l-3.24-2.5c-.9.6-2.05.96-3.39.96-2.6 0-4.8-1.76-5.59-4.12H3.06v2.59A9.99 9.99 0 0 0 12 22Z"/>
            <path fill="#FBBC05" d="M6.41 13.91A5.99 5.99 0 0 1 6.1 12c0-.66.11-1.3.31-1.91V7.5H3.06A9.99 9.99 0 0 0 2 12c0 1.61.39 3.13 1.06 4.5l3.35-2.59Z"/>
            <path fill="#EA4335" d="M12 5.97c1.47 0 2.78.5 3.81 1.48l2.85-2.85C16.96 3 14.7 2 12 2 8.1 2 4.74 4.24 3.06 7.5l3.35 2.59C7.2 7.73 9.4 5.97 12 5.97Z"/>
        """,
    ),
}


def icon_svg(name: str, class_name: str = "", title: str | None = None) -> Markup:
    escaped_class = escape(class_name)
    accessibility = (
        f'role="img" aria-label="{escape(title)}"'
        if title
        else 'aria-hidden="true"'
    )

    if name in _FILL_ICON_BODIES:
        view_box, body = _FILL_ICON_BODIES[name]
        return Markup(
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="{view_box}" class="app-icon {escaped_class}" {accessibility}>{body}</svg>'
        )

    body = _STROKE_ICON_BODIES.get(name)
    if body is None:
        body = '<circle cx="12" cy="12" r="8"/><path d="M12 8v8"/><path d="M8 12h8"/>'
    return Markup(
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" '
        f'class="app-icon {escaped_class}" fill="none" stroke="currentColor" stroke-width="2" '
        f'stroke-linecap="round" stroke-linejoin="round" {accessibility}>{body}</svg>'
    )
