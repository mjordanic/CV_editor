"""What the CV writer and the cover letter writer share: how a draft is started, redrafted and finalized.

Each writer node does one of four things, decided from state:

- fresh first draft (no generated document yet and no request from the user): a new document
  tailored to this ad, with the candidate's existing CV or earlier letter used only as source
  material ("creative liberty");
- the user's own request (first draft or later): the request is applied to the document as the
  user asked, preserving everything else, so the user steers their own CV and letter;
- critique-driven redraft: the critic's suggestions, filtered against the user's earlier requests,
  plus the removal of every claim the grounding check could not support (never filtered);
- finalize: the best draft of the current request; if it still has unsupported claims, one
  removal-only pass and a fixed notice listing any claims that remain. A cover letter whose every
  draft repeats the CV has the CV cut out without a model call, with a fixed notice.

The writers always see the candidate's original material (CV, earlier letter, notes, portfolio
and what the user said in the chat) as source facts next to the document they edit, so a fact a
first draft dropped can come back and the critic's "restore X from the original" can be followed.
"""
from __future__ import annotations

import logging
import os
from typing import Callable, Optional

from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel

from config.config_loader import local_model_kwargs
from debug_utils import write_to_debug
from agents.critique import (
    NO_NEW_CLAIMS_RULE,
    _clean_text,
    _user_requests,
    candidate_material,
    employer_evidence,
    format_job_description,
)
from agents.job_description import output_language_instruction
from agents.letter_cv_check import session_cvs, strip_cv
from agents.notices import NOTICES_KEY, set_notices
from agents.pdf_text_check import find_placeholders
from agents.refinement import best_draft
from agents.work_context import describe as describe_work_context, with_preference

logger = logging.getLogger(__name__)


# Shared by both writers. The CV writer had these rules; the cover letter writer had none, and a
# letter is where invented motives, achievements and commitments show up.
TRUTHFULNESS_PROTOCOL = (
    "\n\nTRUTHFULNESS PROTOCOL (STRICT ENFORCEMENT):\n"
    "1. **NO HALLUCINATIONS**: State only facts about the candidate that the candidate's own material supports: "
    "the original CV, the earlier cover letter, the notes, the portfolio, and what the user said in this chat. "
    "Do NOT invent skills, certifications, qualifications, experiences, achievements, figures or anecdotes.\n"
    "2. **REFRAMING VS LYING**:\n"
    "   - **ALLOWED (Reframing)**: Changing 'Coded in Python' to 'Developed robust backend solutions using Python' (if true).\n"
    "   - **ALLOWED (Inference)**: If candidate says 'Built React app', you can add 'JavaScript' and 'Frontend Development' as skills.\n"
    "   - **FORBIDDEN (Lying)**: If job requires 'Pilot License' and candidate doesn't have it, do NOT add it. Do NOT add 'AWS' if candidate never mentioned cloud experience.\n"
    "3. **MISSING REQUIREMENTS**: If the candidate lacks a hard requirement (e.g., specific degree, license, years of experience), do NOT fake it. "
    "Focus on highlighting their closest relevant transferable skills instead.\n"
    "4. **NO INVENTED COMMITMENTS OR PREFERENCES**: Do not state availability, a start date, willingness to relocate or travel, "
    "salary expectations, plans to obtain something, or personal reasons for wanting the job unless the candidate's material "
    "or the user states them.\n"
    "5. **FACTS ABOUT THE EMPLOYER**: Say about the employer only what the job description and the company information state."
)

USER_STATEMENTS_LABEL = (
    "**What the User Said in This Chat (the user's own words; facts they state about themselves are the candidate's own material, like the CV):**"
)


def removal_instructions(claims: list[str] | None) -> str:
    """The instruction to remove the claims the grounding check could not support ("" if none)."""
    claims = [c for c in (claims or []) if c and c.strip()]
    if not claims:
        return ""
    listed = "\n".join(f"- {c}" for c in claims)
    return (
        "Remove these claims, which the sources do not support (the candidate's own material: CV, cover letter, notes, "
        "portfolio and what the user said; for the employer: the job ad and company information). Do not replace them "
        f"with other claims the candidate cannot back:\n{listed}"
    )


class DocumentWriterAgent:
    """Base class of CVWriterAgent and CoverLetterWriterAgent (see the module docstring)."""

    kind: str = ""            # state key prefix: "cv" or "cover_letter"
    label: str = ""           # "CV" or "cover letter", as the prompts say it
    file_stem: str = ""       # "generated_CV" or "generated_cover_letter"
    content_field: str = ""   # field of the response model that holds the document
    response_model: type[BaseModel] = BaseModel
    system_prompt: str = ""
    human_prompt: str = ""

    def __init__(
        self,
        output_folder: str = "data/generated_CVs",
        model: str = "openai:gpt-5-mini",
        temperature: float = 0.2,
        filter_model: str = "openai:gpt-5-nano",
        filter_temperature: float = 0.0,
        grounding_checker: Optional[Callable] = None,
        draft_comparer: Optional[Callable] = None,
    ):
        """
        Args:
            output_folder: Folder where drafts and the final text are saved
            model: The LLM model identifier for writing
            temperature: Temperature setting for the main LLM
            filter_model: The LLM model that filters critique suggestions against the user's requests
            filter_temperature: Temperature for the filter LLM
            grounding_checker: CritiqueAgent.check_grounding, used by the removal pass in finalize
            draft_comparer: CritiqueAgent.compare_drafts, used to choose between equally honest drafts
        """
        logger.info(f"Initializing {type(self).__name__} with output_folder: {output_folder}")
        self.llm = init_chat_model(model, temperature=temperature, **local_model_kwargs(model))
        self.filter_llm = init_chat_model(filter_model, temperature=filter_temperature, **local_model_kwargs(filter_model))
        self.output_folder = output_folder
        self.grounding_checker = grounding_checker
        self.draft_comparer = draft_comparer
        os.makedirs(self.output_folder, exist_ok=True)

    # ------------------------------------------------------------------ hooks for the subclasses
    def _original_document(self, candidate_text: dict) -> Optional[str]:
        """The candidate's own version of this document (the uploaded CV or earlier letter)."""
        raise NotImplementedError

    def _no_document_text(self) -> str:
        """What the prompt says when there is no document to start from."""
        raise NotImplementedError

    def _fresh_draft_instructions(self, has_document: bool) -> str:
        """Instructions for the first draft of a request-free cycle: a new, tailored document."""
        raise NotImplementedError

    def _base_document_note(self, mode: str, is_original: bool) -> str:
        """A line under the base document saying what it is and how to use it."""
        return ""

    def _extra_prompt_inputs(self, state) -> dict:
        """Further prompt variables a subclass needs."""
        return {}

    # ------------------------------------------------------------------ formatting
    def _format_company_info(self, company_info: Optional[dict]) -> str:
        if not company_info:
            return "No additional company information available."
        info_parts = []
        if company_info.get("company_description"):
            info_parts.append(f"Description: {company_info['company_description']}")
        info_parts.append(describe_work_context(company_info, company_info.get("work_preference")))
        return "\n".join(info_parts) if info_parts else "No additional company information available."

    def _format_job_description(self, job_description_info: Optional[dict]) -> str:
        # Includes the location, so the writer can follow the job country's conventions.
        return format_job_description(job_description_info)

    @staticmethod
    def _format_previous_requests(requests: list[str] | None) -> str:
        formatted = [f"{i}. {r.strip()}" for i, r in enumerate([r for r in (requests or []) if r and r.strip()], 1)]
        return "\n".join(formatted) if formatted else "No previous modification instructions."

    def _user_request_instructions(self, user_feedback: Optional[str], has_document: bool) -> str:
        """The user's own request, applied as asked (the user steers their own document)."""
        label = self.label
        if user_feedback and user_feedback.strip():
            if has_document:
                return (f"**CRITICAL: Apply ONLY these specific changes to the existing {label}. Preserve everything else exactly as-is:**\n"
                        f"{user_feedback}\n\nYou MUST:\n- Make ONLY the changes explicitly requested above\n"
                        "- Preserve all other content, structure, and formatting\n"
                        "- Do NOT add, remove, or modify anything not mentioned in the instructions\n"
                        f"- Do NOT rewrite or restructure the {label} unless explicitly requested")
            return f"**User Feedback/Modification Request:**\n{user_feedback}\n\nPlease incorporate these changes into the {label}."
        if has_document:
            return f"Please review the existing {label} and make improvements based on the job description and company information."
        return ""

    def _request_on_original_note(self) -> str:
        """
        Added to the user's request when it applies to the candidate's original document (first
        draft). Empty for the CV: the user's own CV belongs to them for any job, so their wish is
        applied to it as asked (D2).
        """
        return ""

    def _refinement_instructions(self, suggestions: str, claims: list[str] | None) -> str:
        """Critique-driven redraft: the (filtered) suggestions, then removals and the no-new-claims rule."""
        parts = [p for p in (suggestions, removal_instructions(claims)) if p]
        body = "\n\n".join(parts)
        return (f"**CRITICAL: Apply ONLY these specific improvements to the existing {self.label}. Preserve everything else exactly as-is:**\n"
                f"{body}\n\nYou MUST:\n- Make ONLY the changes explicitly requested above\n"
                "- Preserve all other content, structure, and formatting\n"
                "- Do NOT add, remove, or modify anything not mentioned in the instructions\n"
                f"- Do NOT rewrite or restructure the {self.label} unless explicitly requested\n"
                f"- {NO_NEW_CLAIMS_RULE}")

    # ------------------------------------------------------------------ the preference filter
    def _filter_critique_instructions_by_user_preferences(self, critique_instructions: str, previous_requests: list[str] | None) -> str:
        """
        Drop critique suggestions that conflict with the user's earlier requests for this document
        (e.g. "don't include education"). Only the critic's suggestions go through this filter; the
        removals of unsupported claims and the no-new-claims rule are added afterwards.
        """
        if not critique_instructions or not critique_instructions.strip():
            return critique_instructions
        user_preferences_text = self._format_previous_requests(previous_requests)
        if user_preferences_text == "No previous modification instructions.":
            return critique_instructions

        filter_prompt = f"""You are filtering critique improvement instructions based on user preferences.

**User Preferences (from previous feedback):**
{user_preferences_text}

**Critique Improvement Instructions:**
{critique_instructions}

**Task:**
Filter the critique instructions to respect user preferences.
- If critique suggests something that conflicts with user preferences, OMIT that suggestion
- If critique suggests something that aligns with user preferences, KEEP it
- If user explicitly stated they don't want something (e.g., "don't include education", "no education section"),
  remove any critique suggestions that would add or modify that element
- Keep all other valid critique suggestions that don't conflict
- If a critique instruction contains multiple parts and only some conflict, keep the non-conflicting parts and remove only the conflicting parts

Return ONLY the filtered critique instructions. If all suggestions conflict, return empty string.
Do NOT add explanations or notes - just return the filtered instructions.
CRITICAL: You MUST NOT add any new content, suggestions, or changes that are not explicitly requested in the critique instructions.
You can ONLY remove parts of the critique instructions that conflict with user preferences. Do not add, modify, or rephrase anything beyond removing conflicting parts."""

        try:
            logger.info("Filtering critique instructions against user preferences...")
            response = self.filter_llm.invoke(filter_prompt)
            raw = response.content if hasattr(response, "content") else str(response)
            # A light model answers "nothing" as "" (the two quote characters) or "None".
            filtered = _clean_text(raw if isinstance(raw, str) else str(raw))
            if filtered:
                logger.info(f"Filtered critique instructions - original length: {len(critique_instructions)}, filtered length: {len(filtered)}")
            else:
                logger.info("All critique suggestions were filtered out due to user preferences")
            return filtered
        except Exception as e:
            logger.warning(f"Error filtering critique instructions: {e}. Using original instructions.")
            return critique_instructions

    # ------------------------------------------------------------------ generation
    def generate_document(
        self,
        document: Optional[str],
        modification_instructions: str,
        state,
        base_document_note: str = "",
    ) -> tuple[str, str]:
        """One writer call: (document text, notes for the user)."""
        candidate_text = state.get("candidate_text") or {}
        same = "(The same as the existing document above.)"
        cv, letter = candidate_text.get("cv"), candidate_text.get("cover_letter")
        job_description_info = state.get("job_description_info")
        company_info = with_preference(state.get("company_info"), state.get("work_preference"))
        llm_input = {
            "base_document": document or self._no_document_text(),
            "base_document_note": base_document_note,
            # The candidate's own documents as source facts; not repeated when they are the base document.
            "original_cv": same if (cv and document == cv) else (cv or "No CV provided."),
            "original_cover_letter": same if (letter and document == letter) else (letter or "No earlier cover letter provided."),
            "candidate_notes": candidate_text.get("notes") or "No notes provided.",
            "relevant_experience_context": state.get("relevant_experience") or "No portfolio provided.",
            "user_statements": _user_requests(state.get("messages")) or "Nothing yet.",
            "job_description": self._format_job_description(job_description_info),
            "company_info": self._format_company_info(company_info),
            "modification_instructions": modification_instructions or "",
            "previous_modification_instructions": self._format_previous_requests(state.get(f"{self.kind}_user_requests")),
            "output_language": output_language_instruction(job_description_info, self.label),
            **self._extra_prompt_inputs(state),
        }
        prompt = ChatPromptTemplate.from_messages([("system", self.system_prompt), ("human", self.human_prompt)])
        chain = prompt | self.llm.with_structured_output(self.response_model)
        logger.info(f"Calling LLM to write the {self.label} - has_modification_instructions: {bool(modification_instructions)}")
        # Retried: a parse error or rate limit must not end the session and lose the user's inputs.
        response = chain.with_retry(stop_after_attempt=3).invoke(llm_input)
        text = getattr(response, self.content_field)
        notes = response.notes
        logger.info(f"LLM response received - {self.label} length: {len(text)}, notes length: {len(notes)}")
        return text, notes

    def save_document(self, text: str, filename: str) -> str:
        file_path = os.path.join(self.output_folder, filename)
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(text)
        logger.info(f"Saved {self.label} text to {file_path}")
        return file_path

    # ------------------------------------------------------------------ the writer node
    def run(self, state):
        """Write or redraft the document (see the module docstring for the four modes)."""
        kind, label = self.kind, self.label
        logger.info(f"{type(self).__name__}.run() called")

        candidate_text = state.get("candidate_text") or {}
        generated = state.get(f"generated_{kind}")
        original = self._original_document(candidate_text)
        document = generated or original
        user_feedback = state.get(f"{kind}_content_feedback")
        previous_requests = list(state.get(f"{kind}_user_requests") or [])
        suggestions = state.get(f"{kind}_critique_improvement_instructions")
        claims = state.get(f"{kind}_claims_to_remove") or []
        is_refinement = bool(state.get(f"{kind}_needs_refinement")) and bool(suggestions or claims) and bool(generated)

        if is_refinement:
            latest = (state.get(f"{kind}_history") or [{}])[-1]
            if latest.get("repeats_cv"):
                # Not a critic suggestion but the fixed instruction to delete the CV: the preference
                # filter must not drop it.
                filtered = suggestions
            else:
                filtered = self._filter_critique_instructions_by_user_preferences(suggestions, previous_requests) if suggestions else ""
            if not filtered and not claims:
                # Every critique suggestion conflicts with the user's earlier requests and nothing has
                # to be removed: keep the draft as it is.
                logger.info("All critique suggestions filtered out due to user preferences - skipping refinement")
                return {
                    f"generated_{kind}": generated,
                    f"{kind}_needs_critique": False,
                    f"{kind}_needs_refinement": False,
                    # Count the skipped pass: the graph still goes on to critique the unchanged
                    # draft, and without this the critique -> redraft loop would never use up its budget.
                    f"{kind}_refinement_count": (state.get(f"{kind}_refinement_count") or 0) + 1,
                    "current_node": f"{kind}_writer",
                    "messages": state.get("messages", []) + [{
                        "role": "assistant",
                        "content": f"{label[0].upper() + label[1:]} refinement skipped: All critique suggestions were filtered out based on your previous preferences."
                    }],
                }
            modification_instructions = self._refinement_instructions(filtered, claims)
            mode = "refinement"
        elif generated is None and not (user_feedback and user_feedback.strip()):
            # First draft without a request from the user: a new document tailored to this ad.
            modification_instructions = self._fresh_draft_instructions(bool(document))
            mode = "fresh"
            logger.info(f"First draft without a user request - writing a fresh {label} tailored to the ad")
        else:
            # The user's own request steers the document (also on the first draft).
            modification_instructions = self._user_request_instructions(user_feedback, bool(document))
            mode = "request"
            if generated is None and document:
                note = self._request_on_original_note()
                if note:
                    modification_instructions += "\n\n" + note

        is_original = generated is None
        text, notes = self.generate_document(
            document,
            modification_instructions,
            state,
            base_document_note=self._base_document_note(mode, is_original and bool(document)),
        )

        updates = {}
        if mode == "request" and user_feedback and user_feedback.strip():
            # The user's requests for this document, kept in state (traces and final_state.json show them).
            updates[f"{kind}_user_requests"] = previous_requests + [user_feedback.strip()]
            # The writer's notes on this request (e.g. a part it did not apply, or applied
            # differently): the router passes them on in its next message to the user.
            updates[f"{kind}_request_notes"] = (notes or "").strip() or None

        history = state.get(f"{kind}_history", []) or []
        iteration = len(history) + 1
        file_path = self.save_document(text, f"{self.file_stem}_draft_{iteration}.txt")
        notes_path = self.save_document(notes, f"{self.file_stem}_notes_draft_{iteration}.txt")

        debug_content = f"GENERATED {label.upper()} (saved to: {file_path}):\n" + "-" * 80 + "\n"
        if mode == "refinement":
            debug_content += "REFINEMENT MODE: Applying critique-based improvements\n"
        debug_content += f"MODIFICATION INSTRUCTIONS:\n{modification_instructions}\n\n"
        debug_content += f"PREVIOUS MODIFICATION INSTRUCTIONS:\n{self._format_previous_requests(previous_requests)}\n\n"
        debug_content += "-" * 80 + "\n" + text + "\n\n"
        debug_content += f"{label.upper()} NOTES (saved to: {notes_path}):\n" + "-" * 80 + "\n" + notes + "\n\n"
        write_to_debug(debug_content, f"{label.upper()} WRITER DEBUG INFO")

        refinement_count = state.get(f"{kind}_refinement_count", 0) or 0
        if mode == "refinement":
            refinement_count += 1
        message = f"{label[0].upper() + label[1:]} has been generated and saved to {file_path}.\n\n**Notes and Explanations:**\n{notes}\n\n"
        message += (f"{label[0].upper() + label[1:]} has been refined based on critique feedback." if mode == "refinement"
                    else f"Please review the {label} and let me know if you'd like any modifications.")
        return {
            **updates,
            f"generated_{kind}": text,
            f"{kind}_needs_critique": True,
            f"{kind}_needs_refinement": False,
            f"{kind}_refinement_count": refinement_count,
            "current_node": f"{kind}_writer",
            "messages": state.get("messages", []) + [{"role": "assistant", "content": message}],
        }

    # ------------------------------------------------------------------ finalize
    def _removal_pass(self, state, text: str, claims: list[str]) -> Optional[str]:
        """One writer call that only deletes the listed claims from the chosen draft."""
        instructions = (
            f"**REMOVAL ONLY: delete these exact claims from the existing {self.label}, and change nothing else.** "
            "Where a deleted claim leaves a sentence or list item broken, shorten it so it still reads well, "
            "without adding anything new.\n"
            + "\n".join(f"- {c}" for c in claims)
            + f"\n\n{NO_NEW_CLAIMS_RULE}"
        )
        try:
            new_text, _ = self.generate_document(text, instructions, state)
        except Exception as e:  # finalize must still produce the document
            logger.warning(f"Removal pass failed ({type(e).__name__}: {e}) - keeping the chosen draft")
            return None
        return new_text

    def finalize_best_version(self, state):
        """
        Keep the best draft of the current request, run one removal-only pass if it still has
        unsupported claims, and save it as the final text.
        """
        kind, label = self.kind, self.label
        logger.info(f"{type(self).__name__}.finalize_best_version() called")
        history = list(state.get(f"{kind}_history", []) or [])
        if not history:
            logger.warning(f"No {label} history found. Cannot finalize best version.")
            return {"current_node": f"finalize_{kind}"}

        # Pick the best draft of the current cycle only (drafts since the router last sent a
        # request here). An older draft from an earlier request must not win, or the user's latest
        # change would be thrown away. Fewer unsupported claims first, then fewer requirements left
        # unshown, then the critic's side-by-side comparison (see best_draft).
        cycle_start = state.get(f"{kind}_cycle_start") or 0
        candidates = history[cycle_start:] or history
        comparer = getattr(self, "draft_comparer", None)
        compare = None
        if comparer is not None:
            def compare(a, b):
                return comparer(label, a.get("content", ""), b.get("content", ""), state.get("job_description_info"))
        best = best_draft(candidates, compare=compare)
        logger.info(f"Selected best version: Iteration {best.get('iteration')} with score {best.get('score')}")

        updates = {}
        cv_repeat_notice = None
        if best.get("repeats_cv"):
            # Every draft of this request repeats the CV (a cover letter; see agents/letter_cv_check.py):
            # cut the CV out of the latest draft where that works, without a model call.
            cvs = session_cvs(state)
            stripped = next(((d, text) for d in reversed(candidates) if (text := strip_cv(d.get("content", ""), cvs))), None)
            if stripped:
                source, text = stripped
                best = {
                    "iteration": len(history) + 1,
                    "content": text,
                    "score": None,
                    "critique": None,
                    "unsupported_claims": [],
                    "grounding_checked": False,  # checked below, as for any unchecked draft
                    "defects": None,
                    "cv_cut_from": source.get("iteration"),
                }
                history.append(best)
                updates[f"{kind}_history"] = history
                logger.info(f"Every {label} draft repeats the CV - cut it out of iteration {source.get('iteration')}")
                cv_repeat_notice = (f"Note: every draft of the {label} repeated your CV, so the CV part was cut out "
                                    f"of it automatically. Read the {label} before you send it.")
            else:
                logger.info(f"Every {label} draft repeats the CV and it could not be cut out - keeping the chosen draft")
                cv_repeat_notice = (f"Warning: the final {label} repeats your CV, and the CV part could not be cut out "
                                    f"automatically. Delete it before you send the {label}, or ask me to redraft it.")
        claims = list(best.get("unsupported_claims") or [])
        checked = best.get("grounding_checked") is not False
        checker = getattr(self, "grounding_checker", None)
        if not checked and checker is not None:
            # The check could not run on this draft during the critique: try once more before it ships.
            rechecked = checker(label, best.get("content", ""), candidate_material(state), employer_info=employer_evidence(state))
            if rechecked is not None:
                claims, checked = list(rechecked), True
                best = {**best, "unsupported_claims": claims, "grounding_checked": True}
                history = [best if h.get("iteration") == best.get("iteration") else h for h in history]
                updates[f"{kind}_history"] = history
                logger.info(f"Grounding check of the chosen {label} ran at finalize: {len(claims)} unsupported claim(s)")
        if claims and checker is not None:
            logger.info(f"The chosen {label} still has {len(claims)} unsupported claim(s) - running one removal-only pass")
            new_text = self._removal_pass(state, best.get("content", ""), claims)
            if new_text and new_text.strip() and new_text != best.get("content"):
                remaining = checker(label, new_text, candidate_material(state), employer_info=employer_evidence(state))
                if remaining is None:
                    # An unchecked removal pass is not known to be cleaner: keep the chosen draft and its claims.
                    logger.info("The grounding check of the removal pass could not run - keeping the chosen draft and its claims")
                elif len(remaining) < len(claims):
                    entry = {
                        "iteration": len(history) + 1,
                        "content": new_text,
                        "score": best.get("score"),
                        "critique": None,
                        "unsupported_claims": remaining,
                        "grounding_checked": True,
                        "defects": best.get("defects"),
                        "removal_pass_of": best.get("iteration"),
                    }
                    history.append(entry)
                    updates[f"{kind}_history"] = history
                    best = entry
                    claims = list(remaining)
                    logger.info(f"Removal pass kept as iteration {entry['iteration']}: {len(remaining)} claim(s) remain")
                else:
                    logger.info("Removal pass did not reduce the unsupported claims - keeping the chosen draft")

        final_text = best.get("content")
        final_path = self.save_document(final_text, f"{self.file_stem}.txt")
        message = (f"**Finalization Complete:**\nSelected version from iteration {best.get('iteration')} "
                   f"(Score: {best.get('score')}) as the best version.\nSaved to {final_path}")
        # Fixed text, not the router model's words: the user must see exactly which claims remain,
        # or that they could not be checked. Set or cleared on every finalize of this document, so a
        # warning about an earlier draft does not outlive it; its PDF notice is cleared too, because
        # a new PDF follows.
        notice = None
        if claims:
            listed = "\n".join(f"- {c}" for c in claims)
            notice = (f"Warning: the final {label} still contains claims that your CV, notes, portfolio and messages "
                      f"(or, for the employer, the job ad and company research) do not support:\n{listed}\n"
                      f"Check them before you send it, and tell me if they should be changed or removed.")
        elif not checked:
            notice = (f"Warning: the check for invented claims could not run on the final {label}. "
                      f"Read it before you send it.")
        if notice:
            message += "\n\n" + notice
        changes = {f"{kind}_claims": notice, f"{kind}_pdf": None}
        if kind == "cover_letter":
            changes[f"{kind}_cv_repeat"] = cv_repeat_notice
        updates[NOTICES_KEY] = set_notices(state, updates, changes)

        placeholders = find_placeholders(final_text)
        if placeholders:
            logger.info(f"The final {label} has placeholders for missing details: {placeholders}")
        return {
            **updates,
            f"generated_{kind}": final_text,
            f"{kind}_final_iteration": best.get("iteration"),
            f"{kind}_placeholders": placeholders,
            "current_node": f"finalize_{kind}",
            # Clear the request after finalization to prevent infinite loops
            f"{kind}_content_feedback": None,
            "messages": state.get("messages", []) + [{"role": "assistant", "content": message}],
        }
