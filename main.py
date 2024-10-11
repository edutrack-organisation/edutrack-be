from typing import List

from fastapi import Depends, FastAPI, HTTPException, File, UploadFile
from sqlalchemy.orm import Session
import crud, models, schemas
from database import SessionLocal, engine
from fastapi.middleware.cors import CORSMiddleware
import shutil
from parse import parse_pdf

models.Base.metadata.create_all(bind=engine)

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

# @app.get("/")
# async def read_root():
#     return {"Hello": "World"}

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
# NOTE: possibly clean up the file after parsing?
@app.post("/parsePDF/")
async def create_upload_file(file: UploadFile = File(...)):
    temp_file_path = f"temp_{file.filename}"
    with open(temp_file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Parse the PDF file (currently mark down for testing)
    # parsed_json  = parsePDF(temp_file_path)
    # parsed_json = parse_markdown_to_json(temp_file_path)
    parsed_json = parse_pdf(temp_file_path)

    # optional: clean up the file that storing the pdf
    # os.remove(temp_file_path)

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