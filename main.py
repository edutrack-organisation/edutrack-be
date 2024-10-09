from typing import List
import os
import time

from fastapi import Depends, FastAPI, HTTPException, File, UploadFile
from sqlalchemy.orm import Session
import crud, models, schemas
from database import SessionLocal, engine
from fastapi.middleware.cors import CORSMiddleware
import shutil
from parse import parse_pdf


models.Base.metadata.create_all(bind=engine)

import json
import pathlib

app = FastAPI()

# CORS ORIGIN AlLOWED -> To revisit for security concerns
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

@app.get("/")
async def read_root():
    return {"Hello": "World"}


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


@app.get("/questions/") 
async def get_questions():
    # Read the JSON file
    json_file_path = pathlib.Path('questions.json')  # Replace with your JSON file path
    with open(json_file_path, 'r', encoding='utf-8') as file:
        questions = json.load(file)
    
    return questions



# @app.get("/items/{item_id}")
# async def read_item(item_id: int, q: Union[str, None] = None):
#     return {"item_id": item_id, "q": q}


# @app.post("/users/", response_model=schemas.User)
# def create_user(user: schemas.UserCreate, db: Session = Depends(get_db)):
#     db_user = crud.get_user_by_email(db, email=user.email)
#     if db_user:
#         raise HTTPException(status_code=400, detail="Email already registered")
#     return crud.create_user(db=db, user=user)


# @app.get("/users/", response_model=List[schemas.User])
# def read_users(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
#     users = crud.get_users(db, skip=skip, limit=limit)
#     return users


# @app.get("/users/{user_id}", response_model=schemas.User)
# def read_user(user_id: int, db: Session = Depends(get_db)):
#     db_user = crud.get_user(db, user_id=user_id)
#     if db_user is None:
#         raise HTTPException(status_code=404, detail="User not found")
#     return db_user


# @app.post("/users/{user_id}/items/", response_model=schemas.Item)
# def create_item_for_user(
#     user_id: int, item: schemas.ItemCreate, db: Session = Depends(get_db)
# ):
#     return crud.create_user_item(db=db, item=item, user_id=user_id)


# @app.get("/items/", response_model=List[schemas.Item])
# def read_items(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
#     items = crud.get_items(db, skip=skip, limit=limit)
#     return items