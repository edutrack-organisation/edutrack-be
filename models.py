"""
This models.py file contains the models for the database tables.
It defines the structure of the tables and the relationships between them.

Refer to alembic/README for more information on how to run migrations after defining model.
"""

from __future__ import annotations  # this is important to have at the top
from sqlalchemy import Column, ForeignKey, Integer, String, Float, UniqueConstraint, Table, event, Index, JSON
from sqlalchemy.orm import relationship, Mapped, mapped_column, DeclarativeBase, Session
from typing import List, ForwardRef
from database import Base
from pgvector.sqlalchemy import Vector


# https://docs.sqlalchemy.org/en/20/orm/cascades.html#passive-deletes


class Base(DeclarativeBase):
    pass


# Association table for many-to-many relationship between Papers and Learning Outcomes
# This creates an intermediate table that connects papers with their learning outcomes
# Each row represents a paper-learning outcome pair with their respective IDs
paper_learning_outcome_association_table = Table(
    "paper_learning_outcomes_association_table",
    Base.metadata,
    Column("paper_id", Integer, ForeignKey("papers.id"), primary_key=True),
    Column("learning_outcome_id", Integer, ForeignKey("learning_outcomes.id"), primary_key=True),
)

# Association table for many-to-many relationship between Topics and Questions
# This creates an intermediate table that connects questions with their topics
# Each row represents a topic-question pair with their respective IDs
# ondelete="CASCADE" ensures that when a topic or question is deleted,
# their associations are automatically removed from this table
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
    questions: Mapped[List["Question"]] = relationship(
        # back_populates links to the 'topics' attribute in Question class
        secondary=topic_question_association_table,
        back_populates="topics",
        cascade="all, delete",  # Configures cascade behavior for related records
    )

# 9 oct

class GradeDistribution(Base):
    __tablename__ = "grade_distributions"

    id = Column(Integer, primary_key=True, index=True)
    paper_id = Column(Integer, ForeignKey("papers.id"),
                      unique=True, nullable=False)
    bins = Column(JSON, nullable=False)
    counts = Column(JSON, nullable=False)
    total_students = Column(Integer, nullable=False)
    average_score = Column(Float, nullable=False)
    standard_deviation = Column(Float, nullable=True)

    paper = relationship("Paper", back_populates="grade_distribution")

# 16 oct


class Dashboard(Base):
    __tablename__ = "dashboards"
    id = Column(Integer, primary_key=True, index=True)
    paper_id = Column(Integer, ForeignKey("papers.id"),
                      unique=True, nullable=False)
    payload = Column(JSON, nullable=False)
    paper = relationship("Paper", back_populates="dashboard")
    
    
# TODO: handle nullable values if no longer nullable
class Paper(Base):
    __tablename__ = "papers"

    id: Mapped[int] = mapped_column(primary_key=True)
    title: Mapped[String] = mapped_column(String)
    description: Mapped[String] = mapped_column(String, nullable=True)
    module_code: Mapped[String] = mapped_column(String, nullable=True)
    module_name: Mapped[String] = mapped_column(String, nullable=True)
    academic_year: Mapped[String] = mapped_column(
        String, nullable=True)  # e.g. "AY25/26"
    semester: Mapped[String] = mapped_column(
        String, nullable=True)      # e.g. "1" or "2"
    overall_difficulty: Mapped[float] = mapped_column(Float, nullable=True)

    questions: Mapped[List["Question"]] = relationship(
        # if delete paper -> delete questions (orphan FK)
        back_populates="paper",
        cascade="all, delete-orphan",
    )  # One to Many relationship
    statistics: Mapped["Statistic"] = relationship(
        back_populates="paper", cascade="all, delete-orphan",)  # One to One relationship

    # Many to many relationship
    learning_outcomes: Mapped[List["LearningOutcome"]] = relationship(
        secondary=paper_learning_outcome_association_table, back_populates="papers"
    )
    grade_distribution = relationship(
        "GradeDistribution", back_populates="paper", uselist=False, cascade="all, delete-orphan")
    dashboard = relationship(
        "Dashboard", back_populates="paper", uselist=False, cascade="all, delete-orphan")
    analysis = relationship(
        "PaperAnalysis", back_populates="paper", uselist=False, cascade="all, delete-orphan"
    )


class Question(Base):
    __tablename__ = "questions"

    id: Mapped[int] = mapped_column(primary_key=True)
    question_number: Mapped[int] = mapped_column(Integer)
    type: Mapped[String] = mapped_column(String, server_default="MCQ", nullable=False)
    answer = Column(String, nullable=True)
    description: Mapped[String] = mapped_column(String)
    mark: Mapped[int] = mapped_column(Integer)
    difficulty: Mapped[int] = mapped_column(Integer)
    
    level: Mapped[String] = mapped_column(String, nullable=True)

    paper_id: Mapped[int] = mapped_column(ForeignKey("papers.id", ondelete="CASCADE"))  # Many to one relationship
    paper: Mapped["Paper"] = relationship(back_populates="questions")  # Many to one relationship

    # Many to many relationship
    topics: Mapped[List["Topic"]] = relationship(
        secondary=topic_question_association_table, back_populates="questions", cascade="all, delete"
    )
    overview = Column(JSON, nullable=True)

    embedding = Column(Vector(384), nullable=False)  # embedding column for vector similarity search
    __table_args__ = (Index("questions_embedding_idx", "embedding", postgresql_using="ivfflat"),)
    student_choice_statistics = Column(JSON, nullable=True)
    grade_statistics = Column(JSON, nullable=True)
    quartile_statistics = Column(JSON, nullable=True)
    individual_choice_statistics = Column(JSON, nullable=True)


# NOTE: This classes are not in used in the current iteration of the application.
class Statistic(Base):
    __tablename__ = "statistics"

    id: Mapped[int] = mapped_column(primary_key=True)
    normalised_average_marks: Mapped[float] = mapped_column(Float)
    normalised_mean_marks: Mapped[float] = mapped_column(Float)
    normalised_median_marks: Mapped[float] = mapped_column(Float)
    normalised_min_marks: Mapped[float] = mapped_column(Float)
    normalised_max_marks: Mapped[float] = mapped_column(Float)

    paper_id: Mapped[int] = mapped_column(ForeignKey("papers.id"))  # one to one relationship
    paper: Mapped["Paper"] = relationship(back_populates="statistics")  # one to one relationship


class LearningOutcome(Base):
    __tablename__ = "learning_outcomes"

    id: Mapped[int] = mapped_column(primary_key=True)
    title: Mapped[String] = mapped_column(String)
    description: Mapped[String] = mapped_column(String)

    # Many to many relationship
    papers: Mapped[List["Paper"]] = relationship(
        secondary=paper_learning_outcome_association_table, back_populates="learning_outcomes"
    )
    

class PaperAnalysis(Base):
    __tablename__ = "paper_analyses"

    id = Column(Integer, primary_key=True, index=True)
    paper_id = Column(Integer, ForeignKey("papers.id"),
                      unique=True, nullable=False)
    analysis_data = Column(JSON, nullable=False)

    paper = relationship("Paper", back_populates="analysis")


class PaperComparison(Base):
    __tablename__ = "paper_comparisons"

    id = Column(Integer, primary_key=True, index=True)
    baseline_paper_id = Column(Integer, ForeignKey(
        "papers.id", ondelete="CASCADE"), nullable=False)
    comparison_paper_id = Column(Integer, ForeignKey(
        "papers.id", ondelete="CASCADE"), nullable=False)
    comparison_data = Column(JSON, nullable=False)

    # Ensure we don't create duplicate comparison records for the exact same pair of papers
    __table_args__ = (
        UniqueConstraint('baseline_paper_id', 'comparison_paper_id',
                         name='_baseline_comparison_uc'),
    )
