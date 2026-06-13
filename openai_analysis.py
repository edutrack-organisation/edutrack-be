import json
import logging
from pydantic import BaseModel
from typing import List, Optional

# Import the OpenAI async client based on your openai_parse.py structure
from config import async_client
import models

logger = logging.getLogger(__name__)

# 1. Define the Pydantic schema for OpenAI's structured output


class OpenAIAIAnalysisPayload(BaseModel):
    patterns: List[str]
    distractor_analysis: List[str]
    cognitive_depth: List[str]
    assessment_quality: List[str]
    quartile_analysis: List[str]
    recommendations: List[str]


async def generate_openai_analysis(paper: models.Paper) -> dict:
    """
    Takes a database Paper object, constructs a prompt with student performance data, 
    and calls the OpenAI API to generate an analysis.
    """

    # 2. Extract the relevant paper AND performance data (Exact match to Gemini logic)
    questions_data = []
    for q in paper.questions:
        questions_data.append({
            "question_id": q.id,
            "number": q.question_number,
            "type": getattr(q, "type", "MCQ"),
            "description": q.description,
            "difficulty": q.difficulty,
            "mark": q.mark,
            "level": q.level,  # Maps to Bloom's cognitive depth
            "topics": [t.title for t in q.topics] if q.topics else [],
            "overview": q.overview,  # Contains distractor/misconception data
            "student_choice_statistics": q.student_choice_statistics,
            "individual_choice_statistics": getattr(q, "individual_choice_statistics", None),
            "grade_statistics": q.grade_statistics,
            "quartile_statistics": getattr(q, "quartile_statistics", {
                "top_25": {},    # e.g., {"correct_rate": 0.9, "choices": {"A": 10, "B": 90}}
                "bottom_25": {}  # e.g., {"correct_rate": 0.2, "choices": {"A": 60, "B": 20}}
            })
        })

    # Include overall grade distribution if available
    grade_dist = None
    if getattr(paper, "grade_distribution", None):
        grade_dist = {
            "bins": paper.grade_distribution.bins,
            "counts": paper.grade_distribution.counts,
            "average_score": paper.grade_distribution.average_score
        }

    paper_context = {
        "metadata": {
            "title": paper.title,
            "module_code": paper.module_code,
            "academic_year": paper.academic_year,
            "semester": paper.semester,
            "overall_difficulty": paper.overall_difficulty,
        },
        "grade_distribution": grade_dist,
        "questions": questions_data
    }

    # 3. Insert the EXACT prompt used in Gemini
    prompt = f"""
    Role & Persona
    You are an expert educational data analyst and an experienced Computer Science professor. Your goal is to analyse exam results to uncover deep pedagogical insights, evaluate assessment quality, and provide actionable feedback for curriculum improvement.

    Task
    Analyse the provided JSON payload, which contains exam metadata (if available) and per-question performance statistics. Do not invent module codes or academic years if they are missing from the metadata.
    
    JSON Payload:
    {json.dumps(paper_context, indent=2)}

    Analytical Directives:
    - Synthesize Conceptual Gaps (`patterns`): Avoid question-by-question statistical summaries. Identify broad, underlying conceptual misunderstandings that span multiple topics. Look for cascading effects—how do failures in foundational concepts impact complex problem-solving later in the exam?
    - Diagnose Root Misconceptions (`distractor_analysis`): The `description` field contains the question text and the options. You MUST read this to understand the distractors. Formulate hypotheses about the flawed mental models, algorithmic misconceptions, or systemic teaching gaps that led students to choose specific incorrect options over the correct one.
    - Evaluate Cognitive Progression (`cognitive_depth`): Analyze the cohort's transition from foundational knowledge to complex application based on the `level` field. Pinpoint the exact cognitive threshold where student comprehension breaks down.
    - Critique Instrument Validity (`assessment_quality`): Read the `description` carefully. Assess if the exam effectively and fairly measures the intended learning outcomes. Identify structural flaws in the assessment design—such as unintended trick questions, ambiguous wording, or misaligned difficulty curves implied by the stats.
    - Analyze Performance Discrepancies (`quartile_analysis`): Define "item discrimination" (a metric indicating how well a question differentiates between the top 25% and bottom 25% of students) in your first point. Then, analyze the `quartile_statistics`. Identify "negative discrimination" (where top 25% students perform worse or choose a specific distractor at higher rates than the bottom 25%, indicating ambiguous wording or trick questions). Highlight questions that most effectively separate the cohorts, and explain the conceptual threshold the bottom 25% are failing to cross.
    - Strategic Pedagogical Interventions (`recommendations`): Provide structural recommendations for the curriculum. In addition to overarching curriculum adjustments, you MUST include differentiated, targeted action plans for both the lower 25% cohort (e.g., remedial scaffolding, alternative mental models) and the upper 25% cohort (e.g., extension tasks, advanced application). Suggest specific shifts in teaching methodologies based ONLY on the evidence-backed findings. Avoid generic advice.

    Formatting Directives (CRITICAL):
    - Output your analysis in strictly valid JSON format matching the schema exactly.
    - For EVERY string inside the lists (`patterns`, `distractor_analysis`, `cognitive_depth`, `assessment_quality`, `quartile_analysis`, `recommendations`), you MUST use a strict bulleted "POINT: Explanation" structure.
    - Extract the core theme as a concise, bolded font, followed by a colon, and then provide the detailed text as the explanation.
    - Example string format: "**Cascading Foundational Failure**: The failure to grasp basic syntax in Question 2 directly correlates with..."
    
    Constraints:
    - STRICT EVIDENCE REQUIREMENT: Every single claim, hypothesis, or recommendation you make MUST be explicitly backed up by a valid, logical explanation tied directly to the provided data. Connect the qualitative text in `description` with the quantitative `student_choice_statistics` or grade distributions to justify your insights. Do not make unfounded assumptions or state generic educational platitudes.
    - STRICTLY FORBIDDEN: Do not output mere descriptive statistics (e.g., "Question 1 had a 45% success rate"). Use quantitative data purely as supporting evidence for higher-level pedagogical claims.
    - Terminology: When referencing specific questions in your analysis, ALWAYS use the human-readable `number` field (e.g., "Question 1", "Q3"). NEVER output the raw `question_id`.
    - If a data point is ambiguous, state your uncertainty.
    - Output your analysis in clear, well-structured JSON format exactly matching these keys: patterns, distractor_analysis, assessment_quality, cognitive_depth, quartile_analysis, and recommendations. Each key must contain a list of strings detailing your advanced insights.
    """

    # 4. Call OpenAI Using Structured Outputs
    try:
        messages = [
            {"role": "user", "content": prompt}
        ]

        completion = await async_client.beta.chat.completions.parse(
            model="gpt-4o-2024-08-06",  # Using the same model from your openai_parse.py
            messages=messages,
            temperature=0.2,
            response_format=OpenAIAIAnalysisPayload,
        )

        parsed_response = completion.choices[0].message.parsed

        # Depending on Pydantic version, fallback to .dict() if .model_dump() isn't available
        try:
            return parsed_response.model_dump()
        except AttributeError:
            return parsed_response.dict()

    except Exception as e:
        logger.error(f"Failed to generate analysis from OpenAI: {e}")
        raise Exception("Error generating AI analysis via OpenAI")
