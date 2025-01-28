from typing import List
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

# Load environment variables from .env file
load_dotenv()

app = FastAPI()

# TODO: CORS ORIGIN AlLOWED -> To revisit for security concerns
origins = [
    "*",
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

# Endpoints for courses
@app.get("/courses", response_model=List[schemas.Paper])
async def get_courses(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    courses = crud.get_courses(db, skip=skip, limit=limit)
    return courses

# Endpoint to get course by id
@app.get("/courses/{course_id}", response_model=schemas.Course)
async def get_course_by_id(course_id: int, db: Session = Depends(get_db)):
    course = crud.get_course_by_id(db, course_id=course_id)
    print(course)
    if course is None:
        raise HTTPException(status_code=404, detail="Course not found")
    return course

@app.post("/courses", response_model=schemas.Course)
async def create_course(parsed_json: dict, db: Session = Depends(get_db)):
    db_course = crud.create_course(db, course_title=parsed_json.get("course_title", "Test"))
    return db_course

# Endpoints for papers
@app.get("/papers", response_model=List[schemas.Paper])
async def get_papers(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    papers = crud.get_papers(db, skip=skip, limit=limit)
    return papers

# Endpoint to get paper by id
@app.get("/papers/{paper_id}", response_model=schemas.Paper)
async def get_paper_by_id(paper_id: int, db: Session = Depends(get_db)):
    paper = crud.get_paper_by_id(db, paper_id=paper_id)
    if paper is None:
        raise HTTPException(status_code=404, detail="Paper not found")
    return paper

# Endpoint to update a paper
# @app.put("/papers/{paper_id}", response_model=schemas.Paper)
# async def update_paper(paper_id: int, paper: schemas.PaperCreate, db: Session = Depends(get_db)):

@app.get("/topics/", response_model=List[schemas.Topic]) 
async def get_topics(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    topics = crud.get_topics(db, skip=skip, limit=limit)
    return topics

@app.post("/topics/", response_model=schemas.Topic)
def create_topic(topic: schemas.TopicCreate, db: Session = Depends(get_db)):
    db_topic = crud.get_topic_by_title(db, title=topic.title)
    if db_topic:
        raise HTTPException(status_code=400, detail="Topic already exists")
    return crud.create_topic(db=db, topic=topic)

# POST endpoint that takes in a PDF file
@app.post("/parsePDF/")
async def create_upload_file(file: UploadFile = File(...)):
    temp_file_path = f"temp_{file.filename}"
    with open(temp_file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Parse the PDF file using LlamaParse
    # parsed_json = parse_pdf(temp_file_path)

    # Parse the PDF file using OpenAI GPT4-o
    try:
        parsed_json = parse_PDF_OpenAI(temp_file_path)
    except Exception as e:
        raise HTTPException(status_code=400, detail="Error parsing PDF")

    # Clean up the file that storing the pdf
    os.remove(temp_file_path)

    return parsed_json

# POST endpoint that takes in the parsed JSON file and stores it in the database
@app.post("/saveParsedPDF/")
async def save_parsed_pdf(parsed_json: dict, db: Session = Depends(get_db)):   
    title = parsed_json.get("title")
    questions = parsed_json.get("questions", [])  

    existing_paper = crud.get_paper_by_title(db, title)
    if existing_paper:
        raise HTTPException(status_code=400, detail="Paper with this title already exists")
    
    #TODO: try catch error handling?
    crud.create_paper_with_associated_items(db=db, title=title, questions=questions)
    
    return {"message": "Paper and questions saved successfully"}

# Get questions endpoint
@app.get("/questions/", response_model=List[schemas.Question])
async def get_questions(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    questions = crud.get_questions(db, skip=skip, limit=limit)
    return questions

# Get question by id endpoint
@app.get("/questions/{question_id}", response_model=schemas.Question)
async def get_question_by_id(question_id: int, db: Session = Depends(get_db)):
    question = crud.get_question_by_id(db, question_id=question_id)
    if question is None:
        raise HTTPException(status_code=404, detail="Question not found")
    return question

# Update question by id endpoint
@app.put("/questions/{question_id}", response_model=schemas.Question)
async def update_question_by_id(question_id: int, question: schemas.QuestionUpdate, db: Session = Depends(get_db)):
    db_question = crud.update_question_by_id(db, question_id=question_id, question=question)
    if db_question is None:
        raise HTTPException(status_code=404, detail="Question not found")
    return db_question

# Delete question by id endpoint
@app.delete("/questions/{question_id}", response_model=schemas.Question)
async def delete_question_by_id(question_id: int, db: Session = Depends(get_db)):
    db_question = crud.delete_question_by_id(db, question_id=question_id)
    if db_question is None:
        raise HTTPException(status_code=404, detail="Question not found")
    return db_question