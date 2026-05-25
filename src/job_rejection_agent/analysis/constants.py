"""Shared analysis constants."""

from __future__ import annotations

TECH_SKILL_LEXICON = (
    "python",
    "java",
    "c++",
    "sql",
    "pandas",
    "numpy",
    "scikit-learn",
    "tensorflow",
    "pytorch",
    "machine learning",
    "deep learning",
    "nlp",
    "computer vision",
    "llm",
    "large language models",
    "rag",
    "retrieval augmented generation",
    "langchain",
    "google adk",
    "agentic ai",
    "gemini",
    "vertex ai",
    "docker",
    "kubernetes",
    "gcp",
    "google cloud",
    "aws",
    "azure",
    "firebase",
    "firestore",
    "bigquery",
    "airflow",
    "spark",
    "etl",
    "fastapi",
    "flask",
    "streamlit",
    "react",
    "node.js",
    "git",
    "linux",
    "postgresql",
    "mongodb",
    "redis",
    "rest api",
    "grpc",
    "data analysis",
    "data visualization",
    "tableau",
    "power bi",
    "ci/cd",
    "terraform",
)

SKILL_ALIASES = {
    "google cloud platform": "gcp",
    "google cloud": "gcp",
    "large language model": "llm",
    "large language models": "llm",
    "retrieval augmented generation": "rag",
    "machine learning": "ml",
    "deep learning": "dl",
    "natural language processing": "nlp",
}

LEVEL_RANK = {
    "student": 0,
    "new_grad": 1,
    "entry": 1,
    "junior": 2,
    "mid": 3,
    "senior": 4,
    "staff": 5,
}


def normalize_skill_name(skill: str) -> str:
    lowered = skill.strip().lower()
    return SKILL_ALIASES.get(lowered, lowered)

