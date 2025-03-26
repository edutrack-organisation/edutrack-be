"""
This module provides CRUD (Create, Read, Update, Delete) operations for the EduTrack application.
It includes functions to manage papers, questions, and topics within the database using SQLAlchemy ORM.
This file may not contain the full set of CRUD operation (if it is not needed in this current iteration)
"""

from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError
import models, schemas
from typing import List, Optional
from config import client  # Add this import at the top

from sentence_transformers import SentenceTransformer

# Initialize the model at module level
sentence_model = SentenceTransformer("all-MiniLM-L6-v2")


# Paper operations
def get_papers(db: Session, skip: int = 0, limit: int = 100) -> List[models.Paper]:
    """Retrieves a list of papers."""
    try:
        return db.query(models.Paper).offset(skip).limit(limit).all()
    except SQLAlchemyError as e:
        raise e


def get_paper_by_id(db: Session, paper_id: int) -> Optional[models.Paper]:
    """Retrieves a specific paper by ID."""
    try:
        return db.query(models.Paper).filter(models.Paper.id == paper_id).first()
    except SQLAlchemyError as e:
        raise e


def get_paper_by_title(db: Session, title: str) -> Optional[models.Paper]:
    """Retrieves a paper by its title."""
    try:
        return db.query(models.Paper).filter(models.Paper.title == title).first()
    except SQLAlchemyError as e:
        raise e


# We want to create paper and its associated questions topics in one transaction
def create_paper_with_associated_items(db: Session, title: str, questions: List[dict]) -> models.Paper:
    """Creates a paper with its associated questions and topics in a single transaction."""
    try:
        # Create a new paper entry
        db_paper = models.Paper(title=title)
        db.add(db_paper)

        # Process each question
        for i, question in enumerate(questions):
            # Generate embedding using sentence transformer
            embedding = sentence_model.encode(question.description).tolist()

            db_question = models.Question(
                question_number=i + 1,
                description=question.description,
                mark=question.mark,
                difficulty=question.difficulty,
                embedding=embedding,  # this is for similarity search for RAG
            )

            topics_for_question = question.topics

            # Process topics for the question
            for topic_title in question.topics:
                db_topic = get_topic_by_title(db, topic_title)
                if not db_topic:
                    db_topic = models.Topic(title=topic_title)
                    db.add(db_topic)
                    db.flush()  # Ensure the topic is added to the session before appending, handle duplicates within same paper
                db_question.topics.append(db_topic)

            db.add(db_question)
            db_paper.questions.append(db_question)
        # one shot commit
        db.commit()
        db.refresh(db_paper)
        return db_paper
    except SQLAlchemyError as e:
        db.rollback()
        raise e


# Topic operations
def get_topics(db: Session, skip: int = 0, limit: int = 100) -> List[models.Topic]:
    """Retrieves a list of topics."""
    try:
        return db.query(models.Topic).offset(skip).limit(limit).all()
    except SQLAlchemyError as e:
        raise e


def get_topic_by_id(db: Session, topic_id: int) -> Optional[models.Topic]:
    """Retrieves a specific topic by ID."""
    try:
        return db.query(models.Topic).filter(models.Topic.id == topic_id).first()
    except SQLAlchemyError as e:
        raise e


def get_topic_by_title(db: Session, title: str) -> Optional[models.Topic]:
    """Retrieves a specific topic by title."""
    try:
        return db.query(models.Topic).filter(models.Topic.title == title).first()
    except SQLAlchemyError as e:
        raise e


def upsert_topic(db: Session, topic: schemas.TopicCreate) -> models.Topic:
    """Creates a topic if it doesn't exist, otherwise updates the existing topic."""
    try:
        db_topic = get_topic_by_title(db, topic.title)
        if db_topic:
            # Update existing topic if needed
            db_topic.title = topic.title  # In this case, only title can be updated
            db.commit()
            db.refresh(db_topic)
        else:
            # Create new topic
            db_topic = models.Topic(title=topic.title)
            db.add(db_topic)
            db.commit()
            db.refresh(db_topic)
        return db_topic
    except SQLAlchemyError as e:
        db.rollback()
        raise e


# Question operations


def get_similar_questions(
    db: Session, embedding: list[float], limit: int = 5, similarity_threshold: float = 0.5
) -> List[models.Question]:
    """Retrieves questions similar to the given embedding vector.
    db (Session): SQLAlchemy database session
        embedding (list[float]): Vector representation of the query text (384 dimensions)
        limit (int, optional): Maximum number of similar questions to return. Defaults to 5.
        similarity_threshold (float, optional): Minimum cosine similarity score (0-1). Defaults to 0.5.
            - 1.0 means exactly similar
            - 0.0 means completely different
            - 0.5 means moderately similar

    Returns:
        List[models.Question]: List of Question objects ordered by similarity (most similar first)

    Raises:
        SQLAlchemyError: If there's any database error during the query

    Note:
        - Uses pgvector's cosine_distance (1 - cosine_similarity)
        - Filter keeps questions with similarity >= threshold
        - Orders results by most similar first (lowest distance)
    """
    try:

        return (
            db.query(models.Question)
            .filter(models.Question.embedding.cosine_distance(embedding) <= (1 - similarity_threshold))
            .order_by(models.Question.embedding.cosine_distance(embedding))
            .limit(limit)
            .all()
        )

    except SQLAlchemyError as e:
        raise e


def get_questions(
    db: Session, skip: int = 0, limit: int = 100, topic_id: Optional[int] = None
) -> List[models.Question]:
    """Retrieves a list of questions."""
    try:
        query = db.query(models.Question)
        if topic_id:
            query = query.filter(models.Question.topics.any(id=topic_id))
        return query.offset(skip).limit(limit).all()
    except SQLAlchemyError as e:
        raise e


def get_questions_with_topic(db: Session, topic_id: int, skip: int = 0, limit: int = 100) -> List[models.Question]:
    """Retrieves a list of questions for a specific topic."""
    try:
        return (
            db.query(models.Question).filter(models.Question.topics.any(id=topic_id)).offset(skip).limit(limit).all()
        )
    except SQLAlchemyError as e:
        raise e


def create_question(db: Session, question: schemas.QuestionCreate):
    """Creates a question."""
    db_question = models.Question(
        question_number=question.question_number,
        mark=question.mark,
        difficulty=question.difficulty,
        description=question.description,
    )

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


def get_question_by_id(db: Session, question_id: int) -> Optional[models.Question]:
    """Retrieves a specific question by ID."""
    try:
        return db.query(models.Question).filter(models.Question.id == question_id).first()
    except SQLAlchemyError as e:
        raise e


def update_question_by_id(db: Session, question_id: int, question: schemas.QuestionUpdate):
    """Updates a specific question."""
    try:
        db_question = get_question_by_id(db, question_id)
        if db_question:
            db_question.question_number = question.question_number
            db_question.difficulty = question.difficulty
            db_question.description = question.description

            # Update topics
            db_question.topics = []
            for topic_s in question.topics_str:
                topic = schemas.TopicCreate(title=topic_s)
                # Use get_topic_by_title first
                db_topic = get_topic_by_title(db, topic.title)
                if not db_topic:
                    # Create new topic without committing
                    db_topic = models.Topic(title=topic.title)
                    db.add(db_topic)
                    db.flush()  # Ensure the topic is added to the session
                db_question.topics.append(db_topic)

            # Single commit for the entire operation
            db.commit()
            db.refresh(db_question)
        return db_question  # return None if not found, should be handled in the route
    except SQLAlchemyError as e:
        db.rollback()
        raise e


def delete_question_by_id(db: Session, question_id: int) -> Optional[models.Question]:
    """Deletes a specific question."""
    try:
        db_question = get_question_by_id(db, question_id)
        if db_question:
            db.delete(db_question)
            db.commit()
        return db_question
    except SQLAlchemyError as e:
        db.rollback()
        raise e
