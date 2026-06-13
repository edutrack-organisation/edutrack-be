"""
Main FastAPI application file that defines all API endpoints and their handlers.

This file serves as the core of the EduTrack API, containing the endpoints and config files. 
"""

from typing import List, Optional
import os
from fastapi import Depends, FastAPI, HTTPException, File, UploadFile, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
import crud, schemas
from database import SessionLocal, engine, get_db
from fastapi.middleware.cors import CORSMiddleware
import shutil
from openai_parse import parse_PDF_OpenAI
from gemini_parse import parse_PDF_Gemini
from dotenv import load_dotenv
from generate_question import (
    generate_question_from_prompt,
    select_random_questions_for_topic_with_limit_marks,
)
from gemini_analysis import generate_gemini_analysis
from openai_analysis import generate_openai_analysis
from pydantic import ValidationError
from topics import load_topics
from shared_state import topics_cache
from sqlalchemy import select
from database import get_db
from schemas import PaperComparisonPayload
from gemini_analysis import generate_gemini_paper_comparison

app = FastAPI(
    title="EduTrack API",
    description="API for managing educational content and assessments",
    version="1.0.0",
)


# cache the topics in excel
@app.on_event("startup")
async def startup_event():
    topics_cache.extend(load_topics())


# CORS configuration
origins = [
    "*",  # Replace with actual frontend URL in production
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mock data for papers, remove it later
MOCK_AVERAGE_SCORES = {
    1: {"average_score": 78.5},
    2: {"average_score": 85.2},
    3: {"average_score": 65.0},
    4: {"average_score": 91.0},
    5: {"average_score": 72.3},
}


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    print("Validation error:", exc)
    return JSONResponse(
        status_code=422,
        content={"detail": "Validation error. Please check that your input satisfy the requirement format."},
    )


# GET /papers
# Retrieves a list of all papers
# Parameters:
#   - skip: int (optional) - Number of records to skip for pagination
#   - limit: int (optional) - Maximum number of records to return
# Returns: List of Paper objects
@app.get("/papers", response_model=List[schemas.Paper])
def get_papers(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    try:
        papers = crud.get_papers(db, skip=skip, limit=limit)
        return papers
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve papers: {str(e)}")

# 9 oct
# Add this endpoint. It provides the ID and title of every paper.
@app.get("/papers/summary", response_model=List[schemas.PaperSummary])
def get_all_paper_summaries(db: Session = Depends(get_db)):
    try:
        papers = crud.get_papers(db)
        return papers
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to retrieve paper summaries: {str(e)}")





# GET /topics
# Retrieves a list of all topics that is saved in database
# Parameters:
#   - skip: int (optional) - Number of records to skip for pagination
#   - limit: int (optional) - Maximum number of records to return
# Returns: List of Topic objects
@app.get("/topics", response_model=List[schemas.Topic])
def get_topics(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    try:
        topics = crud.get_topics(db, skip=skip, limit=limit)
        return topics
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve topics: {str(e)}")


# GET /topics/available
# Retrieves the complete list of predefined topics from Excel file
# Returns: List of topic strings
@app.get("/topics/available", response_model=List[str])
def get_available_topics(db: Session = Depends(get_db)):
    try:
        # 1. Get pre-loaded topics from Excel cache
        excel_topics = set(topics_cache)

        # 2. Get user-created topics from Database
        # (Fetching only titles to keep it lightweight)
        db_topics = crud.get_topics(db, limit=10000)
        db_topic_titles = set(t.title for t in db_topics)

        # 3. Merge both sets to remove duplicates
        all_topics = list(excel_topics.union(db_topic_titles))

        # 4. Sort alphabetically for better UX
        all_topics.sort()

        return all_topics
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve available topics list: {str(e)}")


# GET /topics/{topic_id}
# Retrieves a specific topic by its ID
# Parameters:
#   - topic_id: int - The unique identifier of the topic
# Returns: Topic object if found, 404 if not found
@app.get("/topics/{topic_id}", response_model=schemas.Topic)
def get_topic_by_id(topic_id: int, db: Session = Depends(get_db)):
    try:
        topic = crud.get_topic_by_id(db, topic_id=topic_id)
        if topic is None:
            raise HTTPException(status_code=404, detail="Topic not found")
        return topic
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve topic: {str(e)}")


# POST /topics
# Creates a new topic
# Parameters:
#   - topic: TopicCreate - Topic data in request body
# Returns: Created Topic object
# Errors: 409 if topic already exists
@app.post("/topics", response_model=schemas.Topic)
def create_topic(topic: schemas.TopicCreate, db: Session = Depends(get_db)):
    try:
        db_topic = crud.get_topic_by_title(db, title=topic.title)
        if db_topic:
            raise HTTPException(status_code=409, detail="Topic already exists")
        return crud.upsert_topic(db=db, topic=topic)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create topic: {str(e)}")


# POST /papers/parse
# Parses a PDF file and extracts paper information
# Parameters:
#   - file: UploadFile - PDF file to parse
# Returns: Parsed paper data with questions and available topics
@app.post("/papers/parse", response_model=schemas.PaperParseResponse)
async def parse_paper_pdf(file: UploadFile = File(...)):
    temp_file_path = f"temp_{file.filename}"  # Temporary file path for saving the uploaded file
    try:
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Parse the PDF file using OpenAI GPT4-o
        #parsed_json = await parse_PDF_OpenAI(temp_file_path)
        
        parsed_json = await parse_PDF_Gemini(temp_file_path)
        # Combine the parsed_json with all_topics list before sending back to frontend
        parsed_json["all_topics"] = topics_cache

        return parsed_json
    except Exception as e:
        raise HTTPException(status_code=500, detail="Error parsing PDF")
    finally:
        # Clean up the temporary file
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)


# POST /papers
# Creates a new paper with associated questions
# Parameters:
#   - parsed_json: dict - Paper data with questions in request body
# Returns: Created Paper object
# Errors: 409 if paper title already exists
# edit on 16 oct
# In main.py

@app.post("/papers", response_model=schemas.Paper)
def create_paper(paper: schemas.PaperCreate, db: Session = Depends(get_db)):
    db_paper = crud.get_paper_by_title(db, title=paper.title)
    if db_paper:
        raise HTTPException(
            status_code=409, detail="Paper with this title already exists")
    try:
        # UPDATE THIS FUNCTION CALL to pass the dashboard data
        created_paper = crud.create_paper_with_associated_items(
            db,
            paper_data=paper
        )
        return created_paper
    except Exception as e:
        # It's better to log the actual error for debugging
        print(f"Error creating paper: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/papers/{paper_id}/dashboard", response_model=schemas.Dashboard)
def get_dashboard_data(paper_id: int, db: Session = Depends(get_db)):
    dashboard_data = crud.get_dashboard_data_by_paper_id(db, paper_id=paper_id)
    if not dashboard_data:
        raise HTTPException(
            status_code=404, detail="Dashboard data not found for this paper")
    return dashboard_data

# main.py


@app.put("/papers/{paper_id}", response_model=schemas.Paper)
def update_paper_metadata(paper_id: int, paper_update: schemas.PaperUpdate, db: Session = Depends(get_db)):
    try:
        updated_paper = crud.update_paper_metadata(db, paper_id, paper_update)
        if not updated_paper:
            raise HTTPException(status_code=404, detail="Paper not found")
        return updated_paper
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to update paper: {str(e)}")


# GET /questions
# Retrieves a filtered list of questions
# Parameters:
#   - skip: int (optional) - Number of records to skip
#   - limit: int (optional) - Maximum number of records to return
#   - topic_id: int (optional) - Filter questions by topic
# Returns: List of Question objects
@app.get("/questions", response_model=List[schemas.Question])
def get_questions(
    skip: int = 0,
    limit: int = 100,
    topic_id: Optional[int] = None,
    db: Session = Depends(get_db),
):
    try:
        if topic_id is not None:
            questions = crud.get_questions_with_topic(db, topic_id=topic_id, skip=skip, limit=limit)
        else:
            questions = crud.get_questions(db, skip=skip, limit=limit)
        return questions
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve questions: {str(e)}")

# for metrics


# In main.py

@app.get("/papers/metrics", response_model=List[schemas.Paper])
def get_paper_metrics(db: Session = Depends(get_db)):
    """
    Retrieves all papers, calculates metrics using real average scores from the database,
    and returns them as Pydantic schema objects.
    """
    try:
        db_papers = crud.get_papers(db)
        response_papers = []

        for paper in db_papers:
            questions_with_difficulty = [
                q for q in paper.questions if q.difficulty is not None]
            avg_int_difficulty = 1.0
            if questions_with_difficulty:
                total_difficulty = sum(
                    q.difficulty for q in questions_with_difficulty)
                avg_int_difficulty = total_difficulty / \
                    len(questions_with_difficulty)

            # UPDATED: Changed from .from_attributes() back to the correct .from_orm()
            paper_data = schemas.Paper.from_orm(paper)
            paper_data.overall_difficulty = round(avg_int_difficulty, 2)

            difficulty_factor = 0.75 + ((avg_int_difficulty-1) / 4) * 0.5
            grade_dist = crud.get_grade_distribution_by_paper_id(
                db, paper_id=paper.id)

            if grade_dist:
                avg_score = grade_dist.average_score
                normalized_score = avg_score * difficulty_factor
                std = grade_dist.standard_deviation or 0
                cv = std / avg_score
                normalized_score = normalized_score / (1 + cv)

                stats_obj = schemas.Statistic(
                    id=paper.id,
                    normalised_average_marks=min(
                        100.0, round(normalized_score, 2)),
                    normalised_mean_marks=0,
                    normalised_median_marks=0,
                    normalised_min_marks=0,
                    normalised_max_marks=0,
                )
                paper_data.statistics = stats_obj

            response_papers.append(paper_data)

        return response_papers

    except Exception as e:
        print(f"An error occurred in get_paper_metrics: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to retrieve paper metrics: {str(e)}")
        
# GET /papers/{paper_id}
# Retrieves a specific paper by its ID
# Parameters:
#   - paper_id: int - The unique identifier of the paper
# Returns: Paper object if found, 404 if not found


@app.get("/papers/{paper_id}", response_model=schemas.Paper)
def get_paper_by_id(paper_id: int, db: Session = Depends(get_db)):
    try:
        paper = crud.get_paper_by_id(db, paper_id=paper_id)
        print([q.overview for q in paper.questions])
        if paper is None:
            raise HTTPException(status_code=404, detail="Paper not found")
        return paper
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to retrieve paper: {str(e)}")

# GET /questions/{question_id}
# Retrieves a specific question by its ID
# Parameters:
#   - question_id: int - The unique identifier of the question
# Returns: Question object if found, 404 if not found
@app.get("/questions/{question_id}", response_model=schemas.Question)
def get_question_by_id(question_id: int, db: Session = Depends(get_db)):
    try:
        question = crud.get_question_by_id(db, question_id=question_id)
        if question is None:
            raise HTTPException(status_code=404, detail="Question not found")
        return question
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve question: {str(e)}")


# PUT /questions/{question_id}
# Updates an existing question
# Parameters:
#   - question_id: int - The unique identifier of the question
#   - question: QuestionUpdate - Updated question data
# Returns: Updated Question object
@app.put("/questions/{question_id}", response_model=schemas.Question)
def update_question_by_id(question_id: int, question: schemas.QuestionUpdate, db: Session = Depends(get_db)):
    try:
        db_question = crud.update_question_by_id(db, question_id=question_id, question=question)
        if db_question is None:
            raise HTTPException(status_code=404, detail="Question not found")
        return db_question
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update question: {str(e)}")


# DELETE /questions/{question_id}
# Deletes a specific question
# Parameters:
#   - question_id: int - The unique identifier of the question
# Returns: Deleted Question object
@app.delete("/questions/{question_id}", response_model=schemas.Question)
def delete_question_by_id(question_id: int, db: Session = Depends(get_db)):
    try:
        db_question = crud.delete_question_by_id(db, question_id=question_id)
        if db_question is None:
            raise HTTPException(status_code=404, detail="Question not found")
        return db_question
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete question: {str(e)}")


# POST /questions/generate
# Generates a new question using GPT
# Parameters:
#   - req: GenerateQuestionGPTRequest - Prompt for question generation
# Returns: Generated Question object
@app.post("/generate-gpt", response_model=schemas.GenerateQuestionGPTResponse)
def generate_question_using_gpt(req: schemas.GenerateQuestionGPTRequest, db: Session = Depends(get_db)):
    try:
        if not req.prompt:
            raise HTTPException(status_code=400, detail="Prompt is required")
        return generate_question_from_prompt(db, req.prompt)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate question using GPT: {str(e)}")


# POST /questions/quick-generate
# Generates a set of questions based on topics and mark allocation
# Parameters:
#   - req: QuickGenerateQuestionsRequest - List of topics and their mark allocations
# Returns: List of selected Question objects
@app.post("/questions/quick-generate", response_model=List[schemas.Question])
def quick_generate_questions(req: schemas.QuickGenerateQuestionsRequest, db: Session = Depends(get_db)):
    try:
        if not req.topics:
            raise HTTPException(status_code=400, detail="At least one topic is required")

        questions = []
        selected_questions_id = set()

        # req.topics is [TopicMark(topic_id=107, max_allocated_marks=4)]
        # for each of the topic in req.topics, i want to retrive questions belonging to that topic up to max_allocated_marks
        for topic in req.topics:
            if topic.max_allocated_marks <= 0:
                raise HTTPException(status_code=400, detail="Max allocated marks must be greater than 0")

            topic_questions = select_random_questions_for_topic_with_limit_marks(
                db,
                topic_id=topic.topic_id,
                max_allocated_marks=topic.max_allocated_marks,
            )

            # only add questions that haven't been selected yet as one question can belong to multiple topics
            for q in topic_questions:
                if q.id not in selected_questions_id:
                    questions.append(q)
                    selected_questions_id.add(q.id)

        if not questions:
            raise HTTPException(status_code=404, detail="No questions found for the specified topics")
        return questions
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to quick generate questions: {str(e)}")


@app.post("/papers/{paper_id}/grades", response_model=schemas.GradeDistribution)
def save_paper_grades(paper_id: int, distribution: schemas.GradeDistributionCreate, db: Session = Depends(get_db)):
    """Saves or updates the student grade distribution for a specific paper."""
    db_paper = crud.get_paper_by_id(db, paper_id=paper_id)
    if db_paper is None:
        raise HTTPException(status_code=404, detail="Paper not found")
    try:
        return crud.create_or_update_grade_distribution(db, paper_id=paper_id, distribution=distribution)
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to save grade distribution: {str(e)}")


@app.get("/papers/{paper_id}/grades", response_model=schemas.GradeDistribution)
def get_paper_grades(paper_id: int, db: Session = Depends(get_db)):
    """Retrieves the student grade distribution for a specific paper."""
    db_distribution = crud.get_grade_distribution_by_paper_id(
        db, paper_id=paper_id)
    if db_distribution is None:
        raise HTTPException(
            status_code=404, detail="Grade distribution not found for this paper")
    return db_distribution

# 16 oct

@app.post("/papers/{paper_id}/dashboard", response_model=schemas.Dashboard)
def save_paper_dashboard(paper_id: int, data: schemas.DashboardCreate, db: Session = Depends(get_db)):
    """
    Upserts a JSON payload representing dashboard metrics for a paper.
    """
    # Check if the paper exists first
    paper = crud.get_paper_by_id(db, paper_id=paper_id)
    if paper is None:
        raise HTTPException(status_code=404, detail="Paper not found")

    # The logic is now handled by the CRUD function
    try:
        return crud.upsert_dashboard_data(db, paper_id=paper_id, data=data)
    except Exception as e:
        print(f"Error upserting dashboard data: {e}")
        raise HTTPException(
            status_code=500, detail="Could not save dashboard data.")


@app.get("/papers/{paper_id}/dashboard", response_model=schemas.Dashboard)
def get_paper_dashboard(paper_id: int, db: Session = Depends(get_db)):
    """
    Retrieve the saved dashboard payload for a paper.
    """
    # The logic is now handled by the CRUD function
    dashboard_data = crud.get_dashboard_data_by_paper_id(db, paper_id=paper_id)
    if not dashboard_data:
        raise HTTPException(
            status_code=404, detail="Dashboard data not found for this paper")
    return dashboard_data

# 30 oct


@app.post("/questions/statistics", status_code=204)
def save_student_choice_statistics(data: schemas.StudentChoiceData, db: Session = Depends(get_db)):
    """
    Receives and saves the calculated student choice statistics for multiple questions.
    """
    try:
        crud.update_question_statistics_bulk(db, updates=data.questions)
        return
    except Exception as e:
        print(f"Error bulk updating statistics: {e}")
        raise HTTPException(
            status_code=500, detail="Failed to save student choice statistics.")
        
# 23 feb


@app.post("/papers/{paper_id}/ai-analysis", response_model=schemas.PaperAnalysis)
async def create_paper_analysis(
    paper_id: int,
    db: Session = Depends(get_db)
):
    """
    Calls Gemini to generate a paper analysis, validates the payload, 
    and saves it using CRUD operations.
    """
    # 1. Use your existing CRUD function to find the paper
    db_paper = crud.get_paper_by_id(db, paper_id)
    if not db_paper:
        raise HTTPException(status_code=404, detail="Paper not found")

    # 2. Call the Gemini AI service
    try:
        ai_json_dict = await generate_gemini_analysis(db_paper)
    except Exception as e:
        raise HTTPException(
            status_code=502, detail="Failed to generate AI analysis.")

    # 3. Pydantic validation firewall
    try:
        # Validate the raw AI dictionary into our structured schema
        ai_payload = schemas.AIAnalysisPayload(**ai_json_dict)
        # Wrap it in the Create schema expected by the CRUD function
        create_schema = schemas.PaperAnalysisCreate(analysis_data=ai_payload)
    except ValidationError as e:
        raise HTTPException(
            status_code=422, detail="AI returned malformed data.")

    # 4. Use your new CRUD function to save to the database
    try:
        saved_analysis = crud.upsert_paper_analysis(
            db, paper_id, create_schema)
        return saved_analysis
    except Exception as e:
        raise HTTPException(
            status_code=500, detail="Database transaction failed")
        

@app.get("/papers/{paper_id}/ai-analysis", response_model=schemas.PaperAnalysis)
async def get_paper_analysis(paper_id: int, db: Session = Depends(get_db)):
    """Fetches existing AI analysis for a paper using CRUD operations."""

    db_analysis = crud.get_paper_analysis_by_paper_id(db, paper_id=paper_id)

    if not db_analysis:
        raise HTTPException(status_code=404, detail="Analysis not found")

    return db_analysis


@app.post("/papers/compare/{baseline_id}/{comparison_id}", response_model=schemas.PaperComparisonDB)
async def generate_and_save_comparison(
    baseline_id: int,
    comparison_id: int,
    db: Session = Depends(get_db)
):
    """Generates a new AI comparison between two papers and saves it to the database."""
    paper1 = crud.get_paper_by_id(db, baseline_id)
    paper2 = crud.get_paper_by_id(db, comparison_id)

    if not paper1 or not paper2:
        raise HTTPException(
            status_code=404, detail="One or both papers not found.")

    try:
        # 1. Call Gemini to get the raw dict
        ai_json_dict = await generate_gemini_paper_comparison(paper1, paper2)
    except Exception as e:
        raise HTTPException(
            status_code=502, detail="Failed to generate AI comparison.")

    try:
        # 2. Validate with Pydantic firewall
        ai_payload = schemas.PaperComparisonPayload(**ai_json_dict)
        create_schema = schemas.PaperComparisonCreate(
            comparison_data=ai_payload)
    except ValidationError as e:
        raise HTTPException(
            status_code=422, detail="AI returned malformed data.")

    try:
        # 3. Save to database using CRUD
        saved_comparison = crud.upsert_paper_comparison(
            db, baseline_id, comparison_id, create_schema)
        return saved_comparison
    except Exception as e:
        raise HTTPException(
            status_code=500, detail="Database transaction failed")


@app.get("/papers/compare/{baseline_id}/{comparison_id}", response_model=schemas.PaperComparisonDB)
def get_paper_comparison(baseline_id: int, comparison_id: int, db: Session = Depends(get_db)):
    """Fetches an existing comparison from the database."""
    db_comparison = crud.get_paper_comparison(db, baseline_id, comparison_id)

    if not db_comparison:
        raise HTTPException(
            status_code=404, detail="Comparison not found. Please generate it first.")

    return db_comparison


@app.get("/papers/comparisons/all", response_model=List[schemas.PaperComparisonDB])
def get_all_comparisons(db: Session = Depends(get_db)):
    """Fetches all saved AI comparisons."""
    try:
        return crud.get_all_paper_comparisons(db)
    except Exception as e:
        raise HTTPException(
            status_code=500, detail="Failed to fetch comparisons.")
