"""
This models.py file contains the models for the database tables.
It defines the structure of the tables and the relationships between them.

Refer to alembic/README for more information on how to run migrations after defining model.
"""

from __future__ import annotations  # this is important to have at the top
from sqlalchemy import Column, ForeignKey, Integer, String, Float, UniqueConstraint, Table, event, Index
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


# TODO: handle nullable values if no longer nullable
class Paper(Base):
    __tablename__ = "papers"

    id: Mapped[int] = mapped_column(primary_key=True)
    title: Mapped[String] = mapped_column(String)
    description: Mapped[String] = mapped_column(String, nullable=True)
    module: Mapped[String] = mapped_column(String, nullable=True)
    year: Mapped[int] = mapped_column(Integer, nullable=True)
    overall_difficulty: Mapped[float] = mapped_column(Float, nullable=True)

    questions: Mapped[List["Question"]] = relationship(
        # if delete paper -> delete questions (orphan FK)
        back_populates="paper",
        cascade="all, delete-orphan",
    )  # One to Many relationship
    statistics: Mapped["Statistic"] = relationship(back_populates="paper")  # One to One relationship

    # Many to many relationship
    learning_outcomes: Mapped[List["LearningOutcome"]] = relationship(
        secondary=paper_learning_outcome_association_table, back_populates="papers"
    )


class Question(Base):
    __tablename__ = "questions"

    id: Mapped[int] = mapped_column(primary_key=True)
    question_number: Mapped[int] = mapped_column(Integer)
    description: Mapped[String] = mapped_column(String)
    mark: Mapped[int] = mapped_column(Integer)
    difficulty: Mapped[int] = mapped_column(Integer)

    paper_id: Mapped[int] = mapped_column(ForeignKey("papers.id", ondelete="CASCADE"))  # Many to one relationship
    paper: Mapped["Paper"] = relationship(back_populates="questions")  # Many to one relationship

    # Many to many relationship
    topics: Mapped[List["Topic"]] = relationship(
        secondary=topic_question_association_table, back_populates="questions", cascade="all, delete"
    )

    embedding = Column(Vector(384), nullable=False)  # embedding column for vector similarity search
    __table_args__ = (Index("questions_embedding_idx", "embedding", postgresql_using="ivfflat"),)


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
