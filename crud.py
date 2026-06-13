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
from sqlalchemy import update

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
def create_paper_with_associated_items(db: Session, paper_data: schemas.PaperCreate) -> models.Paper:
    """Creates a paper with its associated questions and topics in a single transaction."""
    try:
        # Create a new paper entry
        db_paper = models.Paper(
            title=paper_data.title,
            description=paper_data.description,
            module_code=paper_data.module_code,
            module_name=paper_data.module_name,
            academic_year=paper_data.academic_year,
            semester=paper_data.semester,
            overall_difficulty=paper_data.overall_difficulty,
        )
        db.add(db_paper)
        db.flush()

        # Process each question
        for i, question in enumerate(paper_data.questions):
            # Generate embedding using sentence transformer
            embedding = sentence_model.encode(question.description).tolist()
            overview_data = None
            if question.overview:
                overview_data = {
                    key: (
                        value.dict() if hasattr(value, "dict") else value
                    )
                    for key, value in question.overview.items()
                }

            db_question = models.Question(
                question_number=question.question_number,
                type=question.type,
                description=question.description,
                mark=question.mark,
                difficulty=question.difficulty,
                embedding=embedding,  # this is for similarity search for RAG
                overview=overview_data,
                level=question.level
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


def update_paper_metadata(db: Session, paper_id: int, paper_update: schemas.PaperUpdate) -> Optional[models.Paper]:
    """Updates the metadata fields of a paper."""
    try:
        db_paper = get_paper_by_id(db, paper_id)
        if not db_paper:
            return None

        # Update only provided fields
        update_data = paper_update.dict(exclude_unset=True)
        for key, value in update_data.items():
            setattr(db_paper, key, value)

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
        distance = models.Question.embedding.cosine_distance(embedding)
        return db.query(models.Question).filter(distance <= (1 - similarity_threshold)).limit(limit).all()

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
        type=question.type,
        mark=question.mark,
        difficulty=question.difficulty,
        description=question.description,
        level=question.level
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
            db_question.type = question.type
            db_question.difficulty = question.difficulty
            db_question.description = question.description
            
            if question.level is not None:
                db_question.level = question.level

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
    

def get_grade_distribution_by_paper_id(db: Session, paper_id: int) -> Optional[models.GradeDistribution]:
    """Retrieves the grade distribution for a specific paper by its ID."""
    try:
        return db.query(models.GradeDistribution).filter(models.GradeDistribution.paper_id == paper_id).first()
    except SQLAlchemyError as e:
        raise e


def create_or_update_grade_distribution(db: Session, paper_id: int, distribution: schemas.GradeDistributionCreate) -> models.GradeDistribution:
    """Creates a new grade distribution or updates an existing one for a paper."""
    try:
        db_distribution = get_grade_distribution_by_paper_id(
            db, paper_id=paper_id)
        if db_distribution:
            # Update existing record
            db_distribution.bins = distribution.bins
            db_distribution.counts = distribution.counts
            db_distribution.total_students = distribution.total_students
            db_distribution.average_score = distribution.average_score
            db_distribution.standard_deviation = distribution.standard_deviation
        else:
            # Create new record
            db_distribution = models.GradeDistribution(
                **distribution.dict(), paper_id=paper_id)
            db.add(db_distribution)
        db.commit()
        db.refresh(db_distribution)
        return db_distribution
    except SQLAlchemyError as e:
        db.rollback()
        raise e
    
# 16 oct


def create_dashboard_data(db: Session, paper_id: int, dashboard_data: schemas.DashboardCreate) -> models.Dashboard:
    """Creates dashboard data for a paper using the payload structure."""
    db_dashboard = models.Dashboard(
        paper_id=paper_id,
        payload=dashboard_data.payload.dict() if dashboard_data.payload else {
        },  # ✅ ensure JSON
    )
    db.add(db_dashboard)
    db.commit()
    db.refresh(db_dashboard)
    return db_dashboard

def get_dashboard_data_by_paper_id(db: Session, paper_id: int) -> Optional[models.Dashboard]:
    """Retrieves dashboard data for a specific paper."""
    return db.query(models.Dashboard).filter(models.Dashboard.paper_id == paper_id).first()

# In crud.py


def upsert_dashboard_data(db: Session, paper_id: int, data: schemas.DashboardCreate) -> models.Dashboard:
    """Insert or update dashboard data for a paper safely."""
    try:
        existing_dashboard = db.query(
            models.Dashboard).filter_by(paper_id=paper_id).first()

        if existing_dashboard:
            # ✅ Update payload safely
            existing_dashboard.payload = data.payload.dict() if data.payload else {}
            db.commit()
            db.refresh(existing_dashboard)
            return existing_dashboard
        else:
            # ✅ Create new record and add it to the session
            new_dashboard = models.Dashboard(
                paper_id=paper_id,
                payload=data.payload.dict() if data.payload else {},
            )
            db.add(new_dashboard)  # ✅ Must add before commit
            db.commit()
            db.refresh(new_dashboard)
            return new_dashboard
    except Exception as e:
        db.rollback()
        raise Exception(f"Error upserting dashboard data: {str(e)}")

# 30 oct

def update_question_statistics_bulk(db: Session, updates: List[schemas.QuestionChoiceStatistics]):
    """
    Performs a bulk update of student choice statistics for multiple questions.
    """
    try:
        for item in updates:
            # Prepare the values to update
            values_to_update = {}

            if item.statistics is not None:
                values_to_update["student_choice_statistics"] = item.statistics

            if item.grade_statistics is not None:
                values_to_update["grade_statistics"] = item.grade_statistics
                
            if item.quartile_statistics is not None:
                values_to_update["quartile_statistics"] = item.quartile_statistics
            
            if item.individual_choice_statistics is not None:
                values_to_update["individual_choice_statistics"] = item.individual_choice_statistics
            
            if item.answer is not None:
                values_to_update["answer"] = item.answer

            if values_to_update:
                db.execute(
                    update(models.Question)
                    .where(models.Question.id == item.question_id)
                    .values(**values_to_update)
                )
        db.commit()
    except SQLAlchemyError as e:
        db.rollback()
        raise e
    

def upsert_paper_analysis(db: Session, paper_id: int, data: schemas.PaperAnalysisCreate) -> models.PaperAnalysis:
    """Insert or update AI analysis data for a paper safely."""
    try:
        existing_analysis = db.query(models.PaperAnalysis).filter(
            models.PaperAnalysis.paper_id == paper_id).first()

        if existing_analysis:
            # Update existing payload
            existing_analysis.analysis_data = data.analysis_data.dict() if data.analysis_data else {}
            db.commit()
            db.refresh(existing_analysis)
            return existing_analysis
        else:
            # Create new record
            new_analysis = models.PaperAnalysis(
                paper_id=paper_id,
                analysis_data=data.analysis_data.dict() if data.analysis_data else {},
            )
            db.add(new_analysis)
            db.commit()
            db.refresh(new_analysis)
            return new_analysis
    except SQLAlchemyError as e:
        db.rollback()
        raise e


def get_paper_analysis_by_paper_id(db: Session, paper_id: int) -> Optional[models.PaperAnalysis]:
    """Retrieves the AI analysis for a specific paper by its ID."""
    try:
        return db.query(models.PaperAnalysis).filter(models.PaperAnalysis.paper_id == paper_id).first()
    except SQLAlchemyError as e:
        raise e


def get_paper_comparison(db: Session, baseline_id: int, comparison_id: int) -> Optional[models.PaperComparison]:
    """Retrieves an existing comparison between two specific papers."""
    try:
        return db.query(models.PaperComparison).filter(
            models.PaperComparison.baseline_paper_id == baseline_id,
            models.PaperComparison.comparison_paper_id == comparison_id
        ).first()
    except SQLAlchemyError as e:
        raise e


def upsert_paper_comparison(
    db: Session,
    baseline_id: int,
    comparison_id: int,
    data: schemas.PaperComparisonCreate
) -> models.PaperComparison:
    """Inserts a new comparison or updates an existing one."""
    try:
        existing_comparison = get_paper_comparison(
            db, baseline_id, comparison_id)

        if existing_comparison:
            # Update existing record
            existing_comparison.comparison_data = data.comparison_data.dict(
            ) if data.comparison_data else {}
            db.commit()
            db.refresh(existing_comparison)
            return existing_comparison
        else:
            # Create new record
            new_comparison = models.PaperComparison(
                baseline_paper_id=baseline_id,
                comparison_paper_id=comparison_id,
                comparison_data=data.comparison_data.dict() if data.comparison_data else {},
            )
            db.add(new_comparison)
            db.commit()
            db.refresh(new_comparison)
            return new_comparison
    except SQLAlchemyError as e:
        db.rollback()
        raise e


def get_all_paper_comparisons(db: Session) -> List[models.PaperComparison]:
    """Retrieves all saved paper comparisons."""
    try:
        return db.query(models.PaperComparison).all()
    except SQLAlchemyError as e:
        raise e
