"""Work arrangement, culture signals and the candidate's own work-arrangement preference.

The arrangement (remote / hybrid / on-site, plus location or country limits) and culture signals
come from one model call over the job ad and the company search results (SearchAgent). Anything
neither source states stays "not stated". The candidate's preference is only what the candidate
said (in the chat, or --work-preference in testing mode). Without one (or with "no preference")
the default is inclusive: the candidate is open to the job's arrangement, so the writers never present
remote, hybrid, on-site or relocation as a conflict or weakness and never narrow the candidate's
options, but they also never invent a specific commitment (such as relocating to a named place).
"""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field

NOT_STATED = "not stated"
PREFERENCES = ("remote", "hybrid", "on-site", "no preference")

OPEN_PREFERENCE_RULE = (
    "Treat the candidate as open to this job's arrangement (remote, hybrid, on-site or relocation). Never present the arrangement or location as a conflict, gap or "
    "weakness, and never narrow the candidate's options (e.g. by saying they only want remote work or are tied to "
    "their current city). Do not mention the arrangement, location or relocation at all unless the ad asks for "
    "something specific (e.g. on-site work in a named city, or a work permit); then state only what the candidate "
    "said about it, or nothing. Never invent a commitment the candidate did not make, such as 'willing to relocate "
    "to <place>' or a number of office days, and do not add filler such as 'open to the role's working arrangement'."
)

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
    if preference and preference != "no preference":
        lines.append(f"Candidate's own work-arrangement preference (stated by the candidate): {preference}")
    elif preference:
        lines.append("Candidate's own work-arrangement preference (stated by the candidate): no preference. "
                     + OPEN_PREFERENCE_RULE)
    else:
        lines.append("Candidate's work-arrangement preference: not given. " + OPEN_PREFERENCE_RULE)
    conflict = preference_conflict(preference, ctx.get("arrangement"))
    if conflict:
        lines.append(f"Preference conflict: {conflict} Do not hide or misstate the job's arrangement or the candidate's preference.")
    lines.append("Use these only where relevant; never invent an arrangement, location limit or culture trait that is "
                 "'not stated', and never claim a specific preference or commitment the candidate did not give.")
    return "\n".join(lines)


def with_preference(company_info: Optional[dict], preference: Optional[str]) -> dict:
    """company_info plus the candidate's preference, for the prompt formatters. Always a dict, so the
    preference rule (including the inclusive default when none was given) reaches every prompt even
    when there is no company research."""
    return {**(company_info or {}), "work_preference": preference}
