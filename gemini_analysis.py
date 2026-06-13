import json
import logging
from pydantic import BaseModel
from google.genai import types
from typing import List, Optional

# Import your existing initialized client
from config import gemini_client
import models
from schemas import PaperComparisonPayload

logger = logging.getLogger(__name__)

# 1. Update the schema to match the EXACT keys requested in your prompt


class GeminiAIAnalysisPayload(BaseModel):
    patterns: List[str]
    distractor_analysis: List[str]
    cognitive_depth: List[str]
    assessment_quality: List[str]
    quartile_analysis: List[str]
    recommendations: List[str]
    

class TopicInsight(BaseModel):
    topic_name: str
    better_performing_cohort: str
    analysis: str


class PaperComparisonPayload(BaseModel):
    executive_summary: str
    performance_shift: str
    topic_insights: List[TopicInsight]
    cognitive_depth_analysis: str
    actionable_recommendations: List[str]


async def generate_gemini_analysis(paper: models.Paper) -> dict:
    """
    Takes a database Paper object, constructs a prompt with student performance data, 
    and calls the Gemini API.
    """

    # 2. Extract the relevant paper AND performance data
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

    # 3. Insert your exact prompt
    prompt = f"""
    Role & Persona
    You are an expert educational data analyst and an experienced Computer Science professor. Your goal is to analyse exam results to uncover deep pedagogical insights, evaluate assessment quality, and provide actionable feedback for curriculum improvement.

    Task
    Analyse the provided JSON payload, which contains exam metadata (if available) and per-question performance statistics. Do not invent module codes or academic years if they are missing from the metadata.
    
    JSON Payload:
    {json.dumps(paper_context, indent=2)}

    Analytical Directives:
    - Synthesize Conceptual Gaps (`patterns`): Avoid question-by-question statistical summaries. Identify broad, underlying conceptual misunderstandings that span multiple topics. Look for cascading effects—how do failures in foundational concepts impact complex problem-solving later in the exam?
    - Diagnose Root Misconceptions (`distractor_analysis`): The `description` field contains the question text and the options. You MUST read this to understand the distractors. 
      * For MCQs: Analyze `student_choice_statistics`. Formulate hypotheses about the flawed mental models or algorithmic misconceptions that led students to choose specific incorrect options over the single correct one.
      * For MRQs (Multiple Response Questions): Analyze `individual_choice_statistics`. Note that in MRQs, each choice is evaluated independently. You must analyze both "Errors of Commission" (why a high percentage of students selected a specific incorrect option) and "Errors of Omission" (why a significant percentage failed to select a valid correct option). Detail the specific misunderstanding behind each highly problematic choice.
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
    - STRICT EVIDENCE REQUIREMENT: Every single claim, hypothesis, or recommendation you make MUST be explicitly backed up by a valid, logical explanation tied directly to the provided data. Connect the qualitative text in `description` with the quantitative `student_choice_statistics`, `individual_choice_statistics`, or grade distributions to justify your insights.
    - STRICTLY FORBIDDEN: Do not output mere descriptive statistics (e.g., "Question 1 had a 45% success rate"). Use quantitative data purely as supporting evidence for higher-level pedagogical claims.
    - Terminology: When referencing specific questions in your analysis, ALWAYS use the human-readable `number` field (e.g., "Question 1", "Q3"). NEVER output the raw `question_id`.
    - If a data point is ambiguous, state your uncertainty.
    - Output your analysis in clear, well-structured JSON format exactly matching these keys: patterns, distractor_analysis, assessment_quality, cognitive_depth, quartile_analysis, and recommendations. Each key must contain a list of strings detailing your advanced insights.
    """
    
    # 4. Call Gemini
    try:
        response = await gemini_client.aio.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=GeminiAIAnalysisPayload,  # Enforces your new keys
                temperature=0.2,
            )
        )

        if response.parsed:
            parsed_dict = response.parsed.model_dump()
        else:
            parsed_dict = json.loads(response.text)

        return parsed_dict

    except Exception as e:
        logger.error(f"Failed to generate analysis from Gemini: {e}")
        raise Exception("Error generating AI analysis")


async def format_analysis_for_faculty(analysis_data: dict) -> str:
    """
    Takes the structured JSON analysis and uses Gemini to format it into 
    a highly readable Markdown report using a strict "POINT: Explanation" layout.
    """

    prompt = f"""
    You are an expert educational consultant presenting findings to a university faculty. 
    I have provided a raw JSON payload containing an advanced pedagogical analysis of a recent exam.
    
    Your task is to transform this structured data into a polished, easily scannable Markdown report.
    
    Raw Analysis Data:
    {json.dumps(analysis_data, indent=2)}
    
    Formatting Directives:
    - You MUST create exactly 5 major headings (##) corresponding to the keys in the JSON: 
      Conceptual Gaps, Root Misconceptions, Cognitive Progression, Instrument Validity, and Pedagogical Interventions.
    - Within each section, format every insight from the lists strictly using a bulleted "POINT: Explanation" structure.
    - To do this, extract the core theme or finding as a concise, bolded **POINT**, followed by a colon, and then provide the detailed text as the explanation.
    - Example format:
      * **Cascading Foundational Failure**: The failure to grasp basic syntax in Question 2 directly correlates with...
    - Do not add any extra commentary outside of this structure. 
    - Maintain the exact insights, facts, and tone of the original data; do not invent new findings.
    """

    try:
        response = await gemini_client.aio.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.3,  # Kept slightly lower to ensure strict adherence to the formatting rules
            )
        )

        return response.text

    except Exception as e:
        logger.error(f"Failed to format readable analysis from Gemini: {e}")
        raise Exception("Error formatting AI analysis text")


async def process_paper_analysis(paper: models.Paper):
    # Step 1: Get the strict, structured JSON data
    structured_analysis_dict = await generate_gemini_analysis(paper)

    # (Optional) Save structured_analysis_dict to your database here
    # db.save_analysis(paper.id, structured_analysis_dict)

    # Step 2: Pass that dictionary to the formatter
    readable_markdown_report = await format_analysis_for_faculty(structured_analysis_dict)

    # Return or display the final formatted text
    return {
        "raw_data": structured_analysis_dict,
        "report": readable_markdown_report
    }


async def generate_gemini_paper_comparison(paper1: models.Paper, paper2: models.Paper) -> dict:
    """Extracts metrics from two papers and generates a structured AI comparison."""

    def extract_comparative_data(paper: models.Paper):
        grade_dist = None
        if getattr(paper, "grade_distribution", None):
            grade_dist = {
                "bins": paper.grade_distribution.bins,
                "counts": paper.grade_distribution.counts,
                "average_score": paper.grade_distribution.average_score,
                "standard_deviation": paper.grade_distribution.standard_deviation
            }

        questions_data = []
        for q in paper.questions:
            questions_data.append({
                "question_id": q.id,
                "number": q.question_number,
                "type": getattr(q, "type", "MCQ"),
                "description": q.description,
                "difficulty": q.difficulty,
                "mark": q.mark,
                "level": q.level,
                "topics": [t.title for t in q.topics] if q.topics else [],
                "overview": q.overview,
                "student_choice_statistics": getattr(q, "student_choice_statistics", None),
                "individual_choice_statistics": getattr(q, "individual_choice_statistics", None),
                "grade_statistics": getattr(q, "grade_statistics", None),
                "quartile_statistics": getattr(q, "quartile_statistics", {
                    "top_25": {},
                    "bottom_25": {}
                })
            })

        return {
            "metadata": {
                "academic_year": paper.academic_year,
                "semester": paper.semester,
                "overall_difficulty": paper.overall_difficulty,
            },
            "grade_distribution": grade_dist,
            "questions": questions_data
        }

    comparison_context = {
        "baseline_cohort": extract_comparative_data(paper1),
        "comparison_cohort": extract_comparative_data(paper2)
    }

    prompt = f"""
    Role & Persona
    You are an expert educational data analyst comparing the performance of two student cohorts taking exams for the same module.

    Task
    Analyse the provided JSON payload, which contains comprehensive performance metrics, question descriptions, and distractor statistics for a Baseline Cohort and a Comparison Cohort.

    JSON Payload:
    {json.dumps(comparison_context, indent=2)}
    
    Evidence & Logic Constraints CRITICAL:
    - STRICT EVIDENCE REQUIREMENT: Every single claim, hypothesis, or recommendation you make MUST be explicitly backed up by a valid, logical explanation tied directly to the provided data. Connect the qualitative text in `description` with the quantitative `student_choice_statistics`, `individual_choice_statistics`, or grade distributions to justify your insights.
    - STRICTLY FORBIDDEN: Do not output mere descriptive statistics (e.g., "Question 1 had a 45% success rate"). Use quantitative data purely as supporting evidence for higher-level pedagogical claims.
    - Terminology: When referencing specific questions in your analysis, ALWAYS use the human-readable `number` field (e.g., "Question 1", "Q3"). NEVER output the raw `question_id`.
    - Ambiguity: If a data point is ambiguous or insufficient to draw a definitive conclusion, explicitly state your uncertainty.
    - Write in percentage (%)

    Analytical Directives:
    Throughout your analysis, you MUST consistently frame all observations by keeping the Comparison Cohort as the active subject (e.g., "The Comparison Cohort [improved/declined/showed X] compared to the Baseline Cohort").
    - Executive Summary: Provide a 2-3 sentence high-level overview detailing how the Comparison Cohort's overall performance and conceptual mastery shifted compared to the Baseline Cohort.
    - Performance Shift: State whether the Comparison Cohort improved or declined compared to the Baseline Cohort, and hypothesize why based on overall difficulty, average scores, and quartile shifts.
    - Topic Insights: Compare mastery of overlapping topics. Look beyond basic correct rates and analyze specific distractor trends to identify if the Comparison Cohort resolved or worsened in specific misconceptions compared to the Baseline Cohort.
    - Cognitive Depth: Analyze how effectively the Comparison Cohort handled different Bloom's Taxonomy levels compared to the Baseline Cohort based on the question level data.
    - Recommendations: Provide actionable, comparative advice intended strictly for the Comparison Cohort. Suggest specific pedagogical interventions to address their gaps based on how their errors differ compared to the Baseline Cohort.

    Formatting Directives CRITICAL:
    - Output your analysis in strictly valid JSON format matching the schema exactly.
    """

    try:
        response = await gemini_client.aio.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=PaperComparisonPayload,
                temperature=0.2,
            )
        )

        if response.parsed:
            return response.parsed.model_dump()
        else:
            return json.loads(response.text)

    except Exception as e:
        logger.error(f"Failed to generate comparison from Gemini: {e}")
        raise Exception("Error generating AI comparison")
