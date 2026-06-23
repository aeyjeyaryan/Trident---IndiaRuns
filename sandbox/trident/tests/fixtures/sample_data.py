from app.domain.models import (
    CandidateProfile, JobPosting, CareerEntry, Education, Skill, RedrobSignals,
    TextView, CareerView, BehavioralView, RoleTuple, ExpertScore, FusedScore, RankedCandidate,
)


def make_job() -> JobPosting:
    return JobPosting(
        job_id="test_job_001",
        title="Senior Machine Learning Engineer",
        description="We need an ML engineer with experience in NLP, deep learning, and production ML systems. "
                    "Must have strong Python skills and experience with PyTorch or TensorFlow. "
                    "Responsible for building and deploying recommendation systems at scale.",
        required_skills=["Python", "PyTorch", "NLP", "Deep Learning", "MLOps"],
        preferred_skills=["Kubernetes", "Spark", "AWS"],
        seniority_band="senior",
        role_family="data_science",
        min_years_experience=3,
        max_years_experience=12,
    )


def make_candidate_strong() -> CandidateProfile:
    return CandidateProfile(
        candidate_id="CAND_TEST_001",
        anonymized_name="Test Candidate Strong",
        headline="Senior ML Engineer | NLP Specialist",
        summary="Senior ML engineer with 6 years of experience building NLP systems and recommendation "
                "engines at scale. Proficient in PyTorch, Python, and MLOps. "
                "Led development of production ML pipelines serving 10M+ users.",
        years_of_experience=6.0,
        current_title="Senior ML Engineer",
        current_company="TechCorp",
        current_company_size="1001-5000",
        current_industry="Technology",
        career_history=[
            CareerEntry(
                company="TechCorp",
                title="Senior ML Engineer",
                start_date="2022-01-01",
                end_date=None,
                duration_months=30,
                is_current=True,
                industry="Technology",
                company_size="1001-5000",
                description="Built production ML pipelines for recommendation systems.",
            ),
            CareerEntry(
                company="StartupAI",
                title="ML Engineer",
                start_date="2019-03-01",
                end_date="2021-12-31",
                duration_months=34,
                is_current=False,
                industry="Technology",
                company_size="51-200",
                description="Developed NLP models for text classification and entity extraction.",
            ),
        ],
        education=[
            Education(
                institution="MIT",
                degree="M.S.",
                field_of_study="Computer Science",
                start_year=2017,
                end_year=2019,
                grade="4.0 GPA",
                tier="tier_1",
            ),
        ],
        skills=[
            Skill(name="Python", proficiency="expert", endorsements=50, duration_months=72),
            Skill(name="PyTorch", proficiency="advanced", endorsements=40, duration_months=48),
            Skill(name="NLP", proficiency="advanced", endorsements=35, duration_months=48),
            Skill(name="MLOps", proficiency="intermediate", endorsements=20, duration_months=24),
            Skill(name="Kubernetes", proficiency="intermediate", endorsements=15, duration_months=18),
        ],
        redrob_signals=RedrobSignals(
            profile_completeness_score=92.0,
            signup_date="2024-01-15",
            last_active_date="2026-06-20",
            open_to_work_flag=True,
            profile_views_received_30d=45,
            applications_submitted_30d=3,
            recruiter_response_rate=0.85,
            avg_response_time_hours=12.5,
            connection_count=450,
            endorsements_received=60,
            search_appearance_30d=180,
            saved_by_recruiters_30d=15,
            interview_completion_rate=0.95,
            offer_acceptance_rate=0.60,
            verified_email=True,
            verified_phone=True,
        ),
    )


def make_candidate_weak() -> CandidateProfile:
    return CandidateProfile(
        candidate_id="CAND_TEST_002",
        anonymized_name="Test Candidate Weak",
        headline="Accountant | Finance Specialist",
        summary="Professional accountant with 2 years of experience in finance and bookkeeping.",
        years_of_experience=2.0,
        current_title="Junior Accountant",
        current_company="FinanceCorp",
        current_company_size="201-500",
        current_industry="Finance",
        career_history=[
            CareerEntry(
                company="FinanceCorp",
                title="Junior Accountant",
                start_date="2024-06-01",
                end_date=None,
                duration_months=12,
                is_current=True,
                industry="Finance",
                company_size="201-500",
                description="Handled accounts payable and receivable.",
            ),
        ],
        education=[
            Education(
                institution="State University",
                degree="B.Com",
                field_of_study="Commerce",
                start_year=2019,
                end_year=2023,
                grade="75%",
                tier="tier_3",
            ),
        ],
        skills=[
            Skill(name="Excel", proficiency="intermediate", endorsements=5, duration_months=24),
            Skill(name="Bookkeeping", proficiency="intermediate", endorsements=3, duration_months=18),
        ],
        redrob_signals=RedrobSignals(
            profile_completeness_score=45.0,
            signup_date="2025-06-01",
            last_active_date="2026-01-10",
            open_to_work_flag=False,
            profile_views_received_30d=2,
            applications_submitted_30d=0,
            recruiter_response_rate=0.10,
            avg_response_time_hours=240.0,
            connection_count=50,
            endorsements_received=3,
            search_appearance_30d=5,
            saved_by_recruiters_30d=0,
            interview_completion_rate=0.30,
            offer_acceptance_rate=-1.0,
            verified_email=True,
            verified_phone=False,
        ),
    )


def make_candidate_empty_behavioral() -> CandidateProfile:
    return CandidateProfile(
        candidate_id="CAND_TEST_003",
        anonymized_name="No Signals Candidate",
        headline="Generic Professional",
        summary="Professional with some experience.",
        years_of_experience=3.0,
        current_title="Generalist",
        current_company="GenericCorp",
        current_company_size="201-500",
        current_industry="Services",
        career_history=[
            CareerEntry(
                company="GenericCorp",
                title="Generalist",
                start_date="2023-01-01",
                end_date=None,
                duration_months=30,
                is_current=True,
                industry="Services",
                company_size="201-500",
                description="General professional work.",
            ),
        ],
        education=[],
        skills=[],
        redrob_signals=RedrobSignals(),
    )


def make_text_view(candidate: CandidateProfile) -> TextView:
    skill_text = " ".join(s.name for s in candidate.skills)
    return TextView(
        candidate_id=candidate.candidate_id,
        text=f"{candidate.headline} {candidate.summary} Skills: {skill_text}",
    )


def make_career_view(candidate: CandidateProfile) -> CareerView:
    roles = [
        RoleTuple(
            title=r.title,
            seniority="mid",
            employer=r.company,
            industry=r.industry or "",
            tenure_months=r.duration_months,
            is_current=r.is_current,
        )
        for r in candidate.career_history
    ]
    return CareerView(
        candidate_id=candidate.candidate_id,
        role_sequence=roles,
        total_experience_years=candidate.years_of_experience,
        current_title=candidate.current_title,
        current_industry=candidate.current_industry,
        skills=[s.name for s in candidate.skills],
        education=candidate.education,
    )


def make_behavioral_view(candidate: CandidateProfile) -> BehavioralView:
    rs = candidate.redrob_signals
    return BehavioralView(
        candidate_id=candidate.candidate_id,
        profile_views=rs.profile_views_received_30d,
        applications_submitted=rs.applications_submitted_30d,
        recruiter_response_rate=rs.recruiter_response_rate,
        avg_response_time_hours=rs.avg_response_time_hours,
        connection_count=rs.connection_count,
        search_appearance_30d=rs.search_appearance_30d,
        saved_by_recruiters_30d=rs.saved_by_recruiters_30d,
        interview_completion_rate=rs.interview_completion_rate,
        offer_acceptance_rate=rs.offer_acceptance_rate,
        github_activity_score=rs.github_activity_score,
        profile_completeness=rs.profile_completeness_score,
        open_to_work=rs.open_to_work_flag,
        signup_date=rs.signup_date,
        last_active_date=rs.last_active_date,
        verified_email=rs.verified_email,
        verified_phone=rs.verified_phone,
    )
