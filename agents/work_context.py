"""Work arrangement, culture signals and the candidate's own work-arrangement preference.

The arrangement (remote / hybrid / on-site, plus location or country limits) and culture signals
come from one model call over the job ad and the company search results (SearchAgent). Anything
neither source states stays "not stated". The candidate's preference is only what the candidate
said (in the chat, or --work-preference in testing mode); without it nothing is assumed.
"""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field

NOT_STATED = "not stated"
PREFERENCES = ("remote", "hybrid", "on-site", "no preference")

WORK_CONTEXT_PROMPT = (
    "From the job ad and the company web search results below, extract how and where the job is done "
    "and what the company says about its culture. Use only what these texts state; for anything they do "
    "not state, answer 'not stated' (or an empty list). If the ad and the company pages differ, the ad wins "
    "for this job; mention the difference in 'notes'.\n\n"
    "**Job ad:**\n{job_ad}\n\n**Company web search results:**\n{search_content}"
)


class WorkContext(BaseModel):
    """How and where the job is done, and culture signals, as stated by the ad or company pages."""
    arrangement: Literal["remote", "hybrid", "on-site", "not stated"] = Field(
        ..., description="Work arrangement for this job: remote, hybrid, on-site, or 'not stated'")
    location_limits: str = Field(
        NOT_STATED, description="Office location, required days in office, countries or time zones the job is "
                                "open to, relocation or right-to-work limits; 'not stated' if none are given")
    culture_signals: list[str] = Field(
        default_factory=list, description="Short phrases the texts use about values, culture or ways of working; "
                                          "empty if none")
    notes: str = Field("", description="Where the facts came from, or a difference between ad and company pages")


def remote_work_from(arrangement: str) -> str:
    """The older yes / no / NA remote flag, kept for existing readers of company_info."""
    return {"remote": "yes", "hybrid": "no", "on-site": "no"}.get(arrangement, "NA")


def preference_conflict(preference: Optional[str], arrangement: Optional[str]) -> Optional[str]:
    """A sentence for the user when the stated preference does not match the job, else None."""
    if not preference or preference == "no preference" or not arrangement or arrangement == NOT_STATED:
        return None
    if preference == arrangement:
        return None
    return (f"You said you prefer {preference} work, but this job is {arrangement}. The CV and cover letter will "
            f"not claim otherwise; consider whether to apply, or tell me if your preference has changed.")


def describe(company_info: Optional[dict], preference: Optional[str]) -> str:
    """Lines for a writer, critic or router prompt: arrangement, limits, culture, preference, conflict."""
    ctx = (company_info or {}).get("work_context") or {}
    lines = []
    if ctx:
        lines.append(f"Work arrangement (from the ad and company pages): {ctx.get('arrangement') or NOT_STATED}")
        lines.append(f"Location or country limits: {ctx.get('location_limits') or NOT_STATED}")
        culture = ctx.get("culture_signals") or []
        lines.append(f"Culture signals: {'; '.join(culture) if culture else NOT_STATED}")
    elif (company_info or {}).get("remote_work"):
        lines.append(f"Remote Work Policy: {company_info['remote_work']}")
    if preference:
        lines.append(f"Candidate's own work-arrangement preference (stated by the candidate): {preference}")
    else:
        lines.append("Candidate's work-arrangement preference: not given. Do not state any preference for the candidate.")
    conflict = preference_conflict(preference, ctx.get("arrangement"))
    if conflict:
        lines.append(f"Preference conflict: {conflict} Do not hide or misstate the job's arrangement or the candidate's preference.")
    lines.append("Use these only where relevant; never invent an arrangement, location limit or culture trait that is "
                 "'not stated', and never claim a preference the candidate did not give.")
    return "\n".join(lines)


def with_preference(company_info: Optional[dict], preference: Optional[str]) -> Optional[dict]:
    """company_info plus the candidate's preference, for the prompt formatters."""
    if company_info is None and not preference:
        return company_info
    return {**(company_info or {}), "work_preference": preference}
