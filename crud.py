"""
This module provides CRUD (Create, Read, Update, Delete) operations for the EduTrack application.
It includes functions to manage courses, papers, questions, and topics within the database using SQLAlchemy ORM.
"""

from sqlalchemy.orm import Session
import models, schemas

# CRUD operations for course
def get_courses(db: Session, skip: int = 0, limit: int = 100):
    return db.query(models.Course).offset(skip).limit(limit).all()

def get_course_by_id(db: Session, course_id: int):
    return db.query(models.Course).filter(models.Course.id == course_id).first()

def get_course_by_title(db: Session, course_title: str):
    return db.query(models.Course).filter(models.Course.title == course_title).first()

def create_course(db: Session, course_title: str):
    db_course = models.Course(title=course_title)
    db.add(db_course)
    db.commit()
    db.refresh(db_course)
    return db_course

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
        question_create = schemas.QuestionCreate(description=question.get("description"), question_number=i + 1, difficulty=question.get("difficulty", 1), topics_str=question.get("topics", []))
        
        db_question = models.Question(
        question_number=question_create.question_number,
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

# Updates the question paper with modifications to the question description, topics, marks and difficulty.
# Some questions might be deleted
def update_paper(db: Session, paper_id: int, paper_data: schemas.PaperUpdate):
    # Get the current paper from the DB
    current_paper: models.Paper = db.query(models.Paper).filter(models.Paper.id == paper_id).first()
    if not current_paper:
        raise ValueError("Paper not found")

    # # Update basic paper properties
    # current_paper.title = paper_data.title                            # Reserved for future changes
    # current_paper.description = paper_data.description                # Reserved for future changes
    # current_paper.course_id = paper_data.course_id                    # Reserved for future changes
    # current_paper.module = paper_data.module                          # Reserved for future changes
    # current_paper.year = paper_data.year                              # Reserved for future changes
    # current_paper.overall_difficulty = paper_data.overall_difficulty  # Reserved for future changes

    # Keep a copy of the old questions for deletion logic and student_scores update.
    old_questions = list(current_paper.questions)

    # Process incoming questions. Assume paper_data.questions is the complete list of questions
    updated_question_ids = set()
    for incoming_q in paper_data.questions:
        if incoming_q.id is not None:
            # Existing question: update its fields
            q_obj = next((q for q in current_paper.questions if q.id == incoming_q.id), None)
            if q_obj:
                q_obj.description = incoming_q.description
                q_obj.difficulty = incoming_q.difficulty
                q_obj.topics = db.query(models.Topic).filter(models.Topic.title.in_(incoming_q.topics_str)).all()
                q_obj.marks = incoming_q.marks
                # q_obj.question_number = incoming_q.question_number # Optionally update question_number if that ordering may have changed:
                updated_question_ids.add(q_obj.id)
        # else: # Reserved for future changes
        #     # New question: create and append to the paper
        #     new_question = models.Question(
        #         description=incoming_q.description,
        #         difficulty=incoming_q.difficulty,
        #         # question_number=incoming_q.question_number,
        #         topics=incoming_q.topics,
        #         marks=incoming_q.marks,
        #         paper_id=paper_data.id,
        #         paper=current_paper,
        #     )
        #     current_paper.questions.append(new_question)
        #     # Note: new_question.id will be assigned after flush/commit

    # Determine which questions have been deleted:
    # These are questions present before but not in the updated set.
    deleted_questions = [q for q in old_questions if q.id not in updated_question_ids]

    # Before deleting questions, remove the corresponding student_scores columns (if any).
    if current_paper.student_scores:
        # Assume the ordering is determined by question_number in the original paper.
        # We create a sorted list of the old questions.
        sorted_old_questions = sorted(old_questions, key=lambda q: q.question_number or 0)
        # Find the indices (columns) that correspond to deleted questions.
        indices_to_remove = [
            idx for idx, q in enumerate(sorted_old_questions) if q.id not in updated_question_ids
        ]
        # Remove columns in descending order (to avoid shifting indices)
        for row in current_paper.student_scores:
            for idx in sorted(indices_to_remove, reverse=True):
                if idx < len(row):
                    row.pop(idx)

    # Delete the removed questions from the DB
    for dq in deleted_questions:
        db.delete(dq)

    db.commit()
    db.refresh(current_paper)
    return current_paper

# Update the student scores for a paper
def update_student_scores(db: Session, paper_id: int, student_scores: list[list[int]]):
    paper = db.query(models.Paper).filter(models.Paper.id == paper_id).first()
    if not paper:
        raise ValueError(f"Paper with id {paper_id} not found")
    paper.student_scores = student_scores
    db.commit()
    db.refresh(paper)
    return paper

# Update the question difficulties for a paper
def update_paper_question_difficulties(db: Session, paper_id: int, difficulties: list[int]):
    questions = db.query(models.Question).filter(models.Question.paper_id == paper_id).order_by(models.Question.question_number).all()
    if len(questions) != len(difficulties):
        raise ValueError(f"Number of difficulty values ({len(difficulties)}) does not match the number of questions ({len(questions)}) in paper {paper_id}")
    for question, difficulty in zip(questions, difficulties):
        question.difficulty = difficulty
    db.commit()
    for question in questions:
        db.refresh(question)
    return questions

# CRUD operations for topic
def get_topics(db: Session, skip: int = 0, limit: int = 100):
    return db.query(models.Topic).offset(skip).limit(limit).all()

def get_topic(db: Session, topic_id: int):
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

def create_question(db: Session, question: schemas.QuestionCreate):
    db_question = models.Question(
        question_number=question.question_number,
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


