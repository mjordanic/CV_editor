"""Deterministic check for tools and skills a cheap model added without evidence.

A light writer model chases the job ad's keywords and will happily list "AWS, Azure, GCP" or
"Airflow" when the candidate never mentioned them; the critic, running on the same light model,
does not always catch it. This module checks a fixed list of well-known tools, platforms and
languages: a term counts as unsupported when it appears in a generated document but in none of
the candidate's own material (original CV, original cover letter, portfolio, the user's messages).

The list is deliberately limited to unambiguous product names, so ordinary English words never
trigger it. Tools outside the list are left to the LLM critic.
"""
import re

# Canonical name -> regex alternatives. Matched case-insensitively unless listed in _CASE_SENSITIVE
# (names that are also everyday English words, e.g. "react", "excel").
_TERMS: dict[str, list[str]] = {
    # Cloud platforms
    "AWS": [r"aws", r"amazon web services"],
    "Azure": [r"azure"],
    "GCP": [r"gcp", r"google cloud(?: platform)?"],
    "S3": [r"s3"],
    "EC2": [r"ec2"],
    "Redshift": [r"redshift"],
    "BigQuery": [r"bigquery", r"big query"],
    "Databricks": [r"databricks"],
    "Snowflake": [r"snowflake"],
    "Heroku": [r"heroku"],
    # Workflow orchestration and data processing
    "Airflow": [r"airflow"],
    "Prefect": [r"prefect"],
    "Dagster": [r"dagster"],
    "Luigi": [r"luigi"],
    "dbt": [r"dbt"],
    "Spark": [r"spark", r"pyspark"],
    "Hadoop": [r"hadoop"],
    "Kafka": [r"kafka"],
    "Flink": [r"flink"],
    "RabbitMQ": [r"rabbitmq"],
    "Celery": [r"celery"],
    # Containers, infrastructure, CI
    "Docker": [r"docker"],
    "Kubernetes": [r"kubernetes", r"k8s"],
    "Terraform": [r"terraform"],
    "Ansible": [r"ansible"],
    "Helm": [r"Helm"],
    "Jenkins": [r"jenkins"],
    "GitHub Actions": [r"github actions"],
    "GitLab CI": [r"gitlab ci"],
    # Databases and storage
    "PostgreSQL": [r"postgresql", r"postgres"],
    "MySQL": [r"mysql"],
    "MongoDB": [r"mongodb", r"mongo"],
    "Redis": [r"redis"],
    "Elasticsearch": [r"elasticsearch"],
    "Cassandra": [r"cassandra"],
    "DynamoDB": [r"dynamodb"],
    "Oracle": [r"oracle"],
    # Languages
    "Python": [r"python"],
    "Java": [r"java"],
    "JavaScript": [r"javascript"],
    "TypeScript": [r"typescript"],
    "Scala": [r"scala"],
    "Kotlin": [r"kotlin"],
    "C++": [r"c\+\+"],
    "C#": [r"c#"],
    "Golang": [r"golang"],
    "Rust": [r"Rust"],
    "MATLAB": [r"matlab"],
    # Frameworks and libraries
    "FastAPI": [r"fastapi"],
    "Flask": [r"flask"],
    "Django": [r"django"],
    "React": [r"React"],
    "Node.js": [r"node\.?js"],
    "Spring": [r"Spring Boot"],
    "TensorFlow": [r"tensorflow"],
    "PyTorch": [r"pytorch"],
    "scikit-learn": [r"scikit-learn", r"sklearn"],
    "Pandas": [r"pandas"],
    "XGBoost": [r"xgboost"],
    # BI and analytics
    "Tableau": [r"tableau"],
    "Power BI": [r"power ?bi"],
    "Looker": [r"looker"],
    "Excel": [r"Excel"],
}
_CASE_SENSITIVE = {"Rust", "React", "Spring", "Excel", "Helm"}

_PATTERNS = {
    name: re.compile(
        r"(?<![\w])(?:" + "|".join(alternatives) + r")(?![\w+#])",
        0 if name in _CASE_SENSITIVE else re.IGNORECASE,
    )
    for name, alternatives in _TERMS.items()
}


def terms_in(text: str | None) -> set[str]:
    """Canonical names of the listed tools and skills mentioned in text."""
    if not text:
        return set()
    return {name for name, pattern in _PATTERNS.items() if pattern.search(text)}


def _message_text(msg) -> tuple[str, str]:
    """(role, content) of a message dict or LangChain message."""
    if isinstance(msg, dict):
        return msg.get("role", ""), msg.get("content", "") or ""
    role = {"human": "user"}.get(getattr(msg, "type", ""), getattr(msg, "type", ""))
    content = getattr(msg, "content", "")
    return role, content if isinstance(content, str) else str(content)


def candidate_evidence(state) -> str:
    """
    Everything the candidate supplied about themselves: original CV and cover letter, the full
    portfolio, the portfolio excerpt the writers saw, and the user's own messages (so a skill the
    user asked to add counts as supported). The job ad and generated drafts are not evidence.
    """
    candidate_text = state.get("candidate_text") or {}
    parts = [
        candidate_text.get("cv") or "",
        candidate_text.get("cover_letter") or "",
        state.get("portfolio_text") or "",
        state.get("relevant_experience") or "",
    ]
    for msg in state.get("messages") or []:
        role, content = _message_text(msg)
        if role == "user":
            parts.append(content)
    return "\n".join(parts)


def unsupported_terms(document: str | None, state) -> list[str]:
    """Listed tools and skills that the document mentions but the candidate's material does not."""
    return sorted(terms_in(document) - terms_in(candidate_evidence(state)))


def unsupported_job_terms(state, extra_text: str = "") -> list[str]:
    """
    Listed tools and skills the job ad (or extra_text, e.g. a critique) mentions that the candidate's
    material does not: the ones a writer is tempted to add. Used to tell the writer not to add them.
    """
    job = state.get("job_description_info") or {}
    job_text = "\n".join(str(job.get(k) or "") for k in ("candidate_minimal_requirements", "job_description"))
    return sorted(terms_in(job_text + "\n" + (extra_text or "")) - terms_in(candidate_evidence(state)))
