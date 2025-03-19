from typing import List, Optional
import os

from fastapi import Depends, FastAPI, HTTPException, File, UploadFile
from sqlalchemy.orm import Session
import crud, schemas
from database import SessionLocal, engine
from fastapi.middleware.cors import CORSMiddleware
import shutil
from parse import parse_pdf
from openai_parse import parse_PDF_OpenAI
from dotenv import load_dotenv
from topics_data import all_topics
from generate_question import (
    generate_question_from_prompt,
    select_random_questions_for_topic_with_limit_marks,
)

# Load environment variables from .env file
load_dotenv()

app = FastAPI(
    title="EduTrack API",
    description="API for managing educational content and assessments",
    version="1.0.0",
)

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


# Dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# Endpoints for papers
@app.get("/papers", response_model=List[schemas.Paper])
def get_papers(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    try:
        papers = crud.get_papers(db, skip=skip, limit=limit)
        return papers
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve papers: {str(e)}")


# Endpoint to get paper by id
@app.get("/papers/{paper_id}", response_model=schemas.Paper)
def get_paper_by_id(paper_id: int, db: Session = Depends(get_db)):
    try:
        paper = crud.get_paper_by_id(db, paper_id=paper_id)
        if paper is None:
            raise HTTPException(status_code=404, detail="Paper not found")
        return paper
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve paper: {str(e)}")


@app.get("/topics", response_model=List[schemas.Topic])
def get_topics(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    try:
        topics = crud.get_topics(db, skip=skip, limit=limit)
        return topics
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve topics: {str(e)}")


# Endpoint to get topic by id
@app.get("/topics/{topic_id}", response_model=schemas.Topic)
def get_topic_by_id(topic_id: int, db: Session = Depends(get_db)):
    try:
        topic = crud.get_topic_by_id(db, topic_id=topic_id)
        if topic is None:
            raise HTTPException(status_code=404, detail="Topic not found")
        return topic
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve topic: {str(e)}")


@app.post("/topics", response_model=schemas.Topic)
def create_topic(topic: schemas.TopicCreate, db: Session = Depends(get_db)):
    try:
        db_topic = crud.get_topic_by_title(db, title=topic.title)
        if db_topic:
            raise HTTPException(status_code=409, detail="Topic already exists")
        return crud.create_topic(db=db, topic=topic)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create topic: {str(e)}")


# POST endpoint that takes in a PDF file
@app.post("/papers/parse", response_model=schemas.PaperParseResponse)
async def parse_paper_pdf(file: UploadFile = File(...)):
    temp_file_path = f"temp_{file.filename}"  # Temporary file path for saving the uploaded file
    try:
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Parse the PDF file using LlamaParse
        # parsed_json = parse_pdf(temp_file_path)

        # Parse the PDF file using OpenAI GPT4-o
        parsed_json = await parse_PDF_OpenAI(temp_file_path)
        # Combine the parsed_json with all_topics list before sending back to frontend
        parsed_json["all_topics"] = all_topics
        return parsed_json
    except Exception as e:
        raise HTTPException(status_code=500, detail="Error parsing PDF")
    finally:
        # Clean up the temporary file
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)


# POST endpoint that takes in the parsed JSON file and stores it in the database
@app.post("/papers", status_code=201, response_model=schemas.Paper)
def create_paper(parsed_json: dict, db: Session = Depends(get_db)):
    try:
        title = parsed_json.get("title")
        questions = parsed_json.get("questions", [])

        if not title:
            raise HTTPException(status_code=400, detail="Paper title is required")

        existing_paper = crud.get_paper_by_title(db, title)
        if existing_paper:
            raise HTTPException(status_code=409, detail="Paper with this title already exists")

        result = crud.create_paper_with_associated_items(db=db, title=title, questions=questions)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create paper: {str(e)}")


# Get questions endpoint
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


# Get question by id endpoint
@app.get("/questions/{question_id}", response_model=schemas.Question)
def get_question_by_id(question_id: int, db: Session = Depends(get_db)):
    try:
        question = crud.get_question_by_id(db, question_id=question_id)
        if question is None:
            raise HTTPException(status_code=404, detail="Question not found")
        return question
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve question: {str(e)}")


# Update question by id endpoint
@app.put("/questions/{question_id}", response_model=schemas.Question)
def update_question_by_id(question_id: int, question: schemas.QuestionUpdate, db: Session = Depends(get_db)):
    try:
        db_question = crud.update_question_by_id(db, question_id=question_id, question=question)
        if db_question is None:
            raise HTTPException(status_code=404, detail="Question not found")
        return db_question
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update question: {str(e)}")


# Delete question by id endpoint
@app.delete("/questions/{question_id}", response_model=schemas.Question)
def delete_question_by_id(question_id: int, db: Session = Depends(get_db)):
    try:
        db_question = crud.delete_question_by_id(db, question_id=question_id)
        if db_question is None:
            raise HTTPException(status_code=404, detail="Question not found")
        return db_question
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete question: {str(e)}")


# POST endpoint that takes in the prompt for generate question using chatgpt and return the generate question
@app.post("/generate-gpt", response_model=schemas.Question)
def generate_question_using_gpt(req: schemas.GenerateQuestion):
    try:
        if not req.prompt:
            raise HTTPException(status_code=400, detail="Prompt is required")
        return generate_question_from_prompt(req.prompt)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate question using GPT: {str(e)}")


# A post endpoint that is /quick-generate/
# takes in a list of [{topic_id, max score}]
@app.post("/questions/quick-generate", response_model=List[schemas.Question])
def quick_generate_questions(req: schemas.QuickGenerateQuestions, db: Session = Depends(get_db)):
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
