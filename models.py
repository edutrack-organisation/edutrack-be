'''
This models.py file contains the models for the database tables.
It defines the structure of the tables and the relationships between them.
'''

from __future__ import annotations # this is important to have at the top
from sqlalchemy import Column, ForeignKey, Integer, String, Float, UniqueConstraint, Table, event, JSON
from sqlalchemy.orm import relationship, Mapped, mapped_column, DeclarativeBase, Session
from typing import List, ForwardRef
from database import Base

# https://docs.sqlalchemy.org/en/20/orm/cascades.html#passive-deletes

class Base(DeclarativeBase):
    pass

paper_learning_outcome_association_table = Table(
    "paper_learning_outcomes_association_table",
    Base.metadata,
    Column("paper_id", Integer, ForeignKey("papers.id"), primary_key=True),
    Column("learning_outcome_id", Integer, ForeignKey("learning_outcomes.id"), primary_key=True),
)

topic_question_association_table = Table(
    "topic_question_association_table",
    Base.metadata,
    Column("topic_id", Integer, ForeignKey("topics.id", ondelete="CASCADE"), primary_key=True),
    Column("question_id", Integer, ForeignKey("questions.id", ondelete="CASCADE"), primary_key=True),
)

class Topic(Base):
    __tablename__ = "topics"

    id = Column(Integer, primary_key=True)
    title = Column(String)

    # Many to many relationship
    questions: Mapped[List["Question"]] = relationship(secondary=topic_question_association_table, back_populates="topics", cascade="all, delete")

class Course(Base):
    __tablename__ = "courses"

    id: Mapped[int] = mapped_column(primary_key=True)
    title: Mapped[String] = mapped_column(String)

    papers: Mapped[List["Paper"]] = relationship(back_populates="course", cascade="all, delete-orphan")  # One to Many relationship


#TODO: Important to set default values for the columns
class Paper(Base):
    __tablename__ = "papers"

    id: Mapped[int] = mapped_column(primary_key=True)
    title: Mapped[String] = mapped_column(String)
    description: Mapped[String] = mapped_column(String, nullable=True)
    course_id: Mapped[int] = mapped_column(ForeignKey("courses.id"))
    course: Mapped["Course"] = relationship(back_populates="papers")
    module: Mapped[String] = mapped_column(String, nullable=True)
    year: Mapped[int] = mapped_column(Integer, nullable=True)
    overall_difficulty: Mapped[float] = mapped_column(Float, nullable=True)

    questions: Mapped[List["Question"]] = relationship(back_populates="paper", cascade="all, delete-orphan")  # One to Many relationship
    statistics: Mapped["Statistic"] = relationship(back_populates="paper") # One to One relationship

    # Many to many relationship
    learning_outcomes: Mapped[List["LearningOutcome"]] = relationship(secondary=paper_learning_outcome_association_table, back_populates="papers")

    student_scores: Mapped[List[List[int]]] = mapped_column(JSON, nullable=True)
   
class Question(Base):
    __tablename__ = "questions"

    id: Mapped[int] = mapped_column(primary_key=True)
    question_number: Mapped[int] = mapped_column(Integer, nullable=True)
    description: Mapped[String] = mapped_column(String)
    difficulty: Mapped[int] = mapped_column(Integer, nullable=True)
    
    paper_id: Mapped[int] = mapped_column(ForeignKey("papers.id", ondelete="CASCADE")) # Many to one relationship
    paper: Mapped["Paper"] = relationship(back_populates="questions") # Many to one relationship

    # Many to many relationship
    topics: Mapped[List["Topic"]] = relationship(secondary=topic_question_association_table, back_populates="questions", cascade="all, delete")

# NOTE: This classes are not used yet, just here to prep for future use
class Statistic(Base):
    __tablename__ = "statistics"

    id: Mapped[int] = mapped_column(primary_key=True)
    normalised_average_marks: Mapped[float] = mapped_column(Float)
    normalised_mean_marks: Mapped[float] = mapped_column(Float)
    normalised_median_marks: Mapped[float] = mapped_column(Float)
    normalised_min_marks: Mapped[float] = mapped_column(Float)
    normalised_max_marks: Mapped[float] = mapped_column(Float)

    paper_id: Mapped[int] = mapped_column(ForeignKey("papers.id")) # one to one relationship
    paper: Mapped["Paper"] = relationship(back_populates="statistics") # one to one relationship


class LearningOutcome(Base):
    __tablename__ = "learning_outcomes"

    id: Mapped[int] = mapped_column(primary_key=True)
    title: Mapped[String] = mapped_column(String)
    description: Mapped[String] = mapped_column(String)

    # Many to many relationship
    papers: Mapped[List["Paper"]] = relationship(
        secondary=paper_learning_outcome_association_table, back_populates="learning_outcomes")


# NON ANNOTATED VERSION (Old version, Harder to set FKs and relationships)

# from sqlalchemy import Boolean, Column, ForeignKey, Integer, String, Float, UniqueConstraint
# from sqlalchemy.orm import relationship

# from database import Base

# # https://docs.sqlalchemy.org/en/20/orm/cascades.html#passive-deletes


# class User(Base):
#     __tablename__ = "users"

#     id = Column(Integer, primary_key=True)
#     email = Column(String, unique=True, index=True)
#     hashed_password = Column(String)
#     is_active = Column(Boolean, default=True)

#     items = relationship("Item", back_populates="owner")


# class Item(Base):
#     __tablename__ = "items"

#     id = Column(Integer, primary_key=True)
#     title = Column(String, index=True)
#     description = Column(String, index=True)
#     owner_id = Column(Integer, ForeignKey("users.id"))

#     owner = relationship("User", back_populates="items")

# class Topic(Base):
#     __tablename__ = "topics"

#     id = Column(Integer, primary_key=True)
#     title = Column(String)

#     questions = relationship("Question", back_populates="topic")

# #NOTE: Important to set default values for the columns
# # We want to create schemas and models for validation as well
# class Paper(Base):
#     __tablename__ = "papers"

#     id = Column(Integer, primary_key=True)
#     title = Column(String)
#     description = Column(String)
#     module = Column(String)
#     year = Column(Integer)
#     overall_difficulty = Column(Float)

#     questions = relationship("Question", back_populates="paper")
#     statistics = relationship("Statistic", back_populates="paper") 
#     paperLearningOutcome = relationship("paperLearningOutcome", back_populates="paper")

# class Question(Base):
#     __tablename__ = "questions"

#     id = Column(Integer, primary_key=True)
#     question_number = Column(Integer)
#     description = Column(String)
#     difficulty = Column(Integer)

#     topic_id = Column(Integer, ForeignKey("topics.id")) # one question can have multiple topics
#     paper_id = Column(Integer, ForeignKey("papers.id")) # should enforce one to one relationship

#     topic = relationship("Topic", back_populates="questions")
#     paper = relationship("Paper", back_populates="questions") 
    
# class Statistic(Base):
#     __tablename__ = "statistics"

#     id = Column(Integer, primary_key=True)
#     normalised_average_marks = Column(Float)
#     normalised_mean_marks = Column(Float)
#     normalised_median_marks = Column(Float)
#     normalised_min_marks = Column(Float)
#     normalised_max_marks = Column(Float)

#     paper_id = Column(Integer, ForeignKey("papers.id")) # should enforce one to one relationship
#     paper = relationship("Paper", back_populates="statistics")

# class LearningOutcome(Base):
#     __tablename__ = "learning_outcomes"

#     id = Column(Integer, primary_key=True)
#     title = Column(String)
#     description = Column(String)

#     # FK to paperLearningOutcome table
#     paper_learning_outcome = relationship("paperLearningOutcome", back_populates="learning_outcome")

# # This is a table to join learningOutcome and papers together to form Many-to-Many relationship
# class paperLearningOutcome(Base):
#     __tablename__ = "paper_learning_outcomes"

#     id = Column(Integer, primary_key=True)
    
#     # FKs
#     paper_id = Column(Integer, ForeignKey("papers.id"))
#     learning_outcome_id = Column(Integer, ForeignKey("learning_outcomes.id"))

#     # Relationships
#     learning_outcome = relationship("LearningOutcome", back_populates="paperLearningOutcome")
#     paper = relationship("Paper", back_populates="paperLearningOutcome")

#     # Unique constraint, (paper_id, learning_outcome_id) should be unique
#     __table_args__ = (UniqueConstraint('paper_id', 'learning_outcome_id', name='_paper_learning_outcome_uc'),)
    

