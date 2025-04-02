from pydantic import BaseModel
import os
from constants import open_ai_generate_question_prompt
from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError
import crud
import random
from config import client  # Import openAI client from config.py
import models
from typing import List
from sentence_transformers import SentenceTransformer


class GeneratedQuestion(BaseModel):
    description: str
    topics: list[str]
    mark: int
    difficulty: int


def format_input_prompt(content_of_prompt):
    """
    This function is to format the input prompt for GPT API. This include such as formatting the prompt into a format that is recommended by OpenAI, as well as providing other information such as context dump.
    Essentially, this is a wrapper prompt around the user input prompt.
    """
    return open_ai_generate_question_prompt.format(content_of_prompt=content_of_prompt)


# Initialize the model (do this at module level)
sentence_model = SentenceTransformer("all-MiniLM-L6-v2")


def get_embedding(text: str) -> list[float]:
    """Get embedding using Sentence Transformer"""
    return sentence_model.encode(text).tolist()  # Convert numpy array to list


def find_similar_questions(db: Session, prompt: str, prompt_embedding, limit: int = 5):
    """Find similar questions using vector similarity"""
    return crud.get_similar_questions(db, prompt_embedding, limit)


# This is the entry function to generate a question from a prompt using GPT.
def generate_question_from_prompt(db: Session, prompt: str):
    """Generate a question using GPT based on user prompt."""
    print("Generating question using GPT")  # logging
    prompt_embedding = get_embedding(prompt)  # get embedding on raw prompt
    formatted_prompt = format_input_prompt(prompt)
    generated_question = generate_question_with_gpt(db, formatted_prompt, prompt_embedding)
    return generated_question


def generate_question_with_gpt(db: Session, prompt: str, prompt_embedding) -> str:
    """Make API call to OpenAI to generate a question."""

    # Find similar questions first
    similar_questions = find_similar_questions(db, prompt, prompt_embedding)

    # Enhance prompt with similar questions as context
    context = "Here are some similar existing questions for reference:\n"
    for q in similar_questions:
        context += f"- {q.description}\n"

    enhanced_prompt = f"Original prompt: {prompt}\n\nSimilar questions for reference (The generated question should follow the style and formatting of reference questions):\n{context}"

    # enhanced_prompt = prompt
    messages = [{"role": "system", "content": enhanced_prompt}]

    completion = client.beta.chat.completions.parse(
        model="gpt-4o-2024-08-06",
        messages=messages,
        max_tokens=16384,
        response_format=GeneratedQuestion,  # enforce structured format of response https://platform.openai.com/docs/guides/structured-outputs?example=structured-data
    )

    return completion.choices[0].message.parsed or ""


def select_random_questions_for_topic_with_limit_marks(db: Session, topic_id: int, max_allocated_marks: int):
    """
    Select random questions for a topic within mark limit.

    Args:
        db: Database session
        topic_id: ID of the topic to select questions from
        max_allocated_marks: Maximum total marks to allocate

    Returns:
        List of selected questions within mark limit
    """

    # Get all questions and remove duplicates
    all_questions = crud.get_questions_with_topic(db, topic_id)
    unique_questions = {q.description: q for q in all_questions}.values()
    questions = list(unique_questions)

    selected_questions = []
    current_marks = 0

    # Select questions randomly until mark limit is reached
    while questions and current_marks < max_allocated_marks:
        question = random.choice(questions)
        questions.remove(question)  # Remove used question from pool

        if current_marks + question.mark <= max_allocated_marks:
            selected_questions.append(question)
            current_marks += question.mark

    return selected_questions
