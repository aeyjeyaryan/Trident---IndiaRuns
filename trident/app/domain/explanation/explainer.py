from __future__ import annotations

import logging
from typing import Any

from app.domain.models import RankedCandidate, CandidateProfile, JobPosting

log = logging.getLogger(__name__)

JD_KEY_SKILLS = [
    "sentence-transformers", "embeddings", "retrieval", "ranking", "vector database",
    "faiss", "pinecone", "weaviate", "qdrant", "milvus", "elasticsearch",
    "opensearch", "hybrid search", "python", "pytorch", "tensorflow",
    "nlp", "llm", "fine-tuning", "lora", "qlora", "peft",
    "mlops", "kubernetes", "ndcg", "mrr", "map", "learning to rank",
    "information retrieval", "evaluation frameworks",
]

JD_DISQUALIFIER_INDUSTRIES = [
    "it services", "it services & consulting", "outsourcing",
]

JD_PREFERRED_CITIES = [
    "pune", "noida", "delhi", "gurgaon", "gurugram", "bangalore", "bengaluru",
    "hyderabad", "mumbai", "ahmedabad", "chennai", "kolkata",
]


class Explainer:
    def __init__(self, llm_client: Any | None = None):
        self._llm = llm_client

    def explain(
        self,
        candidate: RankedCandidate,
        job: JobPosting | str = "",
        profile: CandidateProfile | None = None,
    ) -> str:
        if self._llm is not None:
            try:
                return self._llm_explain(candidate, job, profile)
            except Exception as e:
                log.warning("LLM explanation failed (%s), using template fallback", e)
        return self._template_explain(candidate, job, profile)

    def _template_explain(
        self,
        candidate: RankedCandidate,
        job: JobPosting | str = "",
        profile: CandidateProfile | None = None,
    ) -> str:
        if profile is None:
            return self._fallback_explain(candidate, job)

        job_title = job.title if isinstance(job, JobPosting) else (job or "the role")

        parts = []

        # Core profile
        title = profile.current_title or "Professional"
        company = profile.current_company or "Unknown"
        yrs = profile.years_of_experience
        parts.append(f"{title} with {yrs:.1f} yrs at {company}")

        # JD-relevant skills (match against JD key skills)
        all_skills = [s.name for s in profile.skills]
        jd_matches = [s.name for s in profile.skills if s.name.lower() in JD_KEY_SKILLS]
        if jd_matches:
            parts.append(f"JD-relevant skills: {', '.join(jd_matches[:5])}")
        else:
            top_skills = [s.name for s in profile.skills[:4]]
            parts.append(f"skills: {', '.join(top_skills)}" if top_skills else "")

        # Redrob behavioral signals
        rs = profile.redrob_signals
        if rs.last_active_date:
            parts.append(f"active {rs.last_active_date}")
        if rs.recruiter_response_rate > 0:
            label = "high" if rs.recruiter_response_rate >= 0.7 else "moderate" if rs.recruiter_response_rate >= 0.4 else "low"
            parts.append(f"{label} engagement ({rs.recruiter_response_rate:.0%} response rate)")
        if rs.saved_by_recruiters_30d and rs.saved_by_recruiters_30d > 5:
            parts.append(f"saved by {rs.saved_by_recruiters_30d} recruiters")
        if rs.profile_views_received_30d and rs.profile_views_received_30d > 20:
            parts.append(f"{rs.profile_views_received_30d} profile views")

        # Location
        loc = profile.location or profile.country or ""
        if loc:
            city_lower = loc.lower().split(",")[0].strip()
            in_preferred = any(c in city_lower for c in JD_PREFERRED_CITIES)
            if in_preferred:
                parts.append(f"based in {loc} (preferred location)")
            else:
                parts.append(f"based in {loc}")

        # Concerns
        concerns = self._assess_concern(profile, job)

        result = "; ".join(p for p in parts if p)
        if concerns:
            result += f". Concerns: {concerns}"
        return result

    def _assess_concern(
        self,
        profile: CandidateProfile,
        job: JobPosting | str = "",
    ) -> str:
        rs = profile.redrob_signals
        concerns = []

        # JD disqualifier: IT services/consulting background
        ind = (profile.current_industry or "").lower()
        for di in JD_DISQUALIFIER_INDUSTRIES:
            if di in ind:
                concerns.append("IT services background (JD disprefers)")
                break

        # Long notice period
        notice = rs.notice_period_days
        if notice and notice > 90:
            concerns.append(f"long notice ({notice}d)")
        elif notice and notice > 60:
            concerns.append(f"{notice}d notice")

        # Not open to work
        if rs.open_to_work_flag is False:
            concerns.append("not open to work")

        # Low engagement
        if rs.recruiter_response_rate < 0.3 and rs.recruiter_response_rate > 0:
            concerns.append("low response rate")
        if rs.last_active_date:
            from datetime import datetime, date
            try:
                last_active = datetime.strptime(rs.last_active_date, "%Y-%m-%d").date()
                days_since = (date.today() - last_active).days
                if days_since > 90:
                    concerns.append(f"inactive {days_since}d")
            except ValueError:
                pass

        # No GitHub signal
        if rs.github_activity_score is not None and rs.github_activity_score < 0:
            concerns.append("no GitHub")

        # International (JD prefers India)
        if profile.country and profile.country.lower() != "india":
            concerns.append("non-India based")

        # Skill gaps relative to JD
        job_skills_lower = (
            [s.lower() for s in job.required_skills]
            if isinstance(job, JobPosting) else []
        )
        profile_skill_names = [s.name.lower() for s in profile.skills]
        missing = [s for s in job_skills_lower if s not in profile_skill_names]
        if missing and len(missing) >= 4:
            concerns.append(f"missing JD skills: {', '.join(missing[:4])}")

        return "; ".join(concerns) if concerns else ""

    def _fallback_explain(
        self,
        candidate: RankedCandidate,
        job: JobPosting | str = "",
    ) -> str:
        w = candidate.gate_weights
        w_str = f"gate=[sem={w[0]:.2f}, car={w[1]:.2f}, beh={w[2]:.2f}]"
        parts = [
            f"sem={candidate.semantic_score:.2f}",
            f"car={candidate.career_score:.2f}",
            f"beh={candidate.behavioral_score:.2f}",
        ]
        low_conf = " [low confidence]" if candidate.low_confidence else ""
        return f"Score {candidate.score:.3f} ({', '.join(parts)}, {w_str}){low_conf}"

    def _llm_explain(
        self,
        candidate: RankedCandidate,
        job: JobPosting | str = "",
        profile: CandidateProfile | None = None,
    ) -> str:
        job_title = job.title if isinstance(job, JobPosting) else (job or "the role")
        profile_summary = ""
        if profile:
            skills = [s.name for s in profile.skills[:8]]
            rs = profile.redrob_signals
            profile_summary = (
                f"Profile: {profile.current_title} @ {profile.current_company}, "
                f"{profile.years_of_experience:.1f} yrs, "
                f"skills: {', '.join(skills)}, "
                f"location: {profile.location}, "
                f"response rate: {rs.recruiter_response_rate}, "
                f"last active: {rs.last_active_date}"
            )

        prompt = (
            f"Candidate rank {candidate.rank} (score {candidate.score:.3f}) for '{job_title}'. "
            f"Semantic fit: {candidate.semantic_score:.2f}, "
            f"Career fit: {candidate.career_score:.2f}, "
            f"Behavioral: {candidate.behavioral_score:.2f}. "
            f"{profile_summary} "
            f"JD requires: embeddings/retrieval/ranking experience, strong Python, "
            f"evaluation frameworks (NDCG/MRR), vector databases, startup fit. "
            f"{'Low confidence due to sparse data.' if candidate.low_confidence else ''} "
            "Write one short sentence summarizing this candidate's fit for this role. "
            "Mention specific facts (title, company, skills, location, response rate) and flag any concerns. "
            "Do not invent facts."
        )
        if hasattr(self._llm, "generate_content"):
            resp = self._llm.generate_content(prompt)
            if hasattr(resp, "text"):
                return resp.text[:300]
        return self._template_explain(candidate, job, profile)
