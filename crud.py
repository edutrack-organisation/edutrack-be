"""
This module provides CRUD (Create, Read, Update, Delete) operations for the EduTrack application.
It includes functions to manage papers, questions, and topics within the database using SQLAlchemy ORM.
"""

from sqlalchemy.orm import Session
import models, schemas

# CRUD operations for paper
def get_papers(db: Session, skip: int = 0, limit: int = 100):
    return db.query(models.Paper).offset(skip).limit(limit).all()

def get_paper_by_id(db: Session, paper_id: int):
    return db.query(models.Paper).filter(models.Paper.id == paper_id).first()

def get_paper_by_title(db: Session, title: str):
    return db.query(models.Paper).filter(models.Paper.title == title).first()

def create_paper(db: Session, paper: schemas.PaperCreate):
    db_paper = models.Paper(title=paper.title)
    db.add(db_paper)
    db.commit()
    db.refresh(db_paper)
    return db_paper

# We want to create paper and its associated questions topics in one transaction
def create_paper_with_associated_items(db: Session, title: str, questions: list):
    # Create a new paper entry
    paper_create = schemas.PaperCreate(title=title)
    db_paper = models.Paper(title=paper_create.title)
    db.add(db_paper)

    # Create a new question entry
    for i, question in enumerate(questions):
        question_create = schemas.QuestionCreate(description=question.get("description"), question_number=i + 1, mark=question.get("mark", 1),difficulty=question.get("difficulty", 1), topics_str=question.get("topics", []))
        
        db_question = models.Question(
        question_number=question_create.question_number,
        mark=question_create.mark,
        difficulty=question_create.difficulty,
        description=question_create.description)
    
        # Add topics to question
        for topic_s in question_create.topics_str:
            # Upsert topic
            topic = schemas.TopicCreate(title=topic_s)
            db_topic = get_topic_by_title(db, topic.title)
            if not db_topic:
                # create topic
                db_topic = models.Topic(title=topic.title)
                db.add(db_topic)
                db.flush() # Ensure the topic is added to the session before appending, handle duplicates within same paper
            db_question.topics.append(db_topic)
        
        db.add(db_question)
        db_paper.questions.append(db_question)
    # one shot commit 
    db.commit()
    db.refresh(db_paper)
    return db_paper      


# CRUD operations for topic
def get_topics(db: Session, skip: int = 0, limit: int = 100):
    return db.query(models.Topic).offset(skip).limit(limit).all()

def get_topic_by_id(db: Session, topic_id: int):
    return db.query(models.Topic).filter(models.Topic.id == topic_id).first()

def get_topic_by_title(db: Session, title: str):
    return db.query(models.Topic).filter(models.Topic.title == title).first()

def create_topic(db: Session, topic: schemas.TopicCreate):
    db_topic = models.Topic(title=topic.title)
    db.add(db_topic)
    db.commit()
    db.refresh(db_topic)
    return db_topic

def upsert_topic(db: Session, topic: schemas.TopicCreate):
    # check if topic exists
    db_topic = get_topic_by_title(db, topic.title)
    if db_topic:
        return db_topic
    else:
        return create_topic(db, topic)
    

# CRUD operations for questions
def get_questions(db: Session, skip: int = 0, limit: int = 100):
    return db.query(models.Question).offset(skip).limit(limit).all()

def get_questions_with_topic(db: Session, topic_id: int, skip: int = 0, limit: int = 100):
    return db.query(models.Question).filter(models.Question.topics.any(id=topic_id)).all()


def create_question(db: Session, question: schemas.QuestionCreate):
    db_question = models.Question(
        question_number=question.question_number,
        mark=question.mark,
        difficulty=question.difficulty,
        description=question.description)
    
    # Add topics to question
    for topic_s in question.topics_str:
        # create the topic from string
        topic = schemas.TopicCreate(title=topic_s)
        db_topic = upsert_topic(db, topic)
        db_question.topics.append(db_topic)
        
    db.add(db_question)
    db.commit()
    db.refresh(db_question)
    return db_question

def get_question_by_id(db: Session, question_id: int):
    return db.query(models.Question).filter(models.Question.id == question_id).first()

def update_question_by_id(db: Session, question_id: int, question: schemas.QuestionUpdate):
    db_question = get_question_by_id(db, question_id)
    if db_question:
        db_question.question_number = question.question_number
        db_question.difficulty = question.difficulty
        db_question.description = question.description

        # Update topics
        db_question.topics = []
        for topic_s in question.topics_str:
            topic = schemas.TopicCreate(title=topic_s)
            db_topic = upsert_topic(db, topic)
            db_question.topics.append(db_topic)

        db.commit()
        db.refresh(db_question)
    return db_question  # return None if not found, should be handled in the route

def delete_question_by_id(db: Session, question_id: int):
    db_question = get_question_by_id(db, question_id)
    if db_question:
        db.delete(db_question)
        db.commit()
        return db_question
    return None


# TODO: more CRUD operations - need to further develop these in future to use for model
def get_learning_outcome(db: Session, learning_outcome_id: int):
    return db.query(models.LearningOutcome).filter(models.LearningOutcome.id == learning_outcome_id).first()

def get_statistic(db: Session, statistic_id: int):
    return db.query(models.Statistic).filter(models.Statistic.id == statistic_id).first()


