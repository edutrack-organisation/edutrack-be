from pydantic import BaseModel
from openai import OpenAI
import os
from constants import open_ai_generate_question_prompt
from sqlalchemy.orm import Session
import crud
import random

OPEN_AI_API_KEY = os.getenv("OPEN_AI_API_KEY")
client = OpenAI(api_key=OPEN_AI_API_KEY)

def format_input_prompt(content_of_prompt):
    return open_ai_generate_question_prompt.format(content_of_prompt=content_of_prompt)

def generate_question_from_prompt(prompt):
    print("Generating question using GPT")  # logging 
    formatted_prompt = format_input_prompt(prompt)
    generated_question = generate_question_with_gpt(formatted_prompt)
    return generated_question

class GeneratedQuestion(BaseModel):
    description: str
    topics: list[str]
    mark: int
    difficulty: int

def generate_question_with_gpt(prompt: str) -> str:
    messages=[
        {
            "role": "system",
            "content": prompt
        }]
    
    completion = client.beta.chat.completions.parse(
        model="gpt-4o-2024-08-06",
        messages=messages,
        max_tokens=16384,
        response_format=GeneratedQuestion,   # enforce structured format of response https://platform.openai.com/docs/guides/structured-outputs?example=structured-data
    )

    return completion.choices[0].message.parsed or ""

def select_random_questions_for_topic_with_limit_marks(db: Session, topic_id: int, max_allocated_marks: int):
    """
    Randomly select questions for a topic until just before exceed max_allocated_marks
    """
    all_questions = crud.get_questions_with_topic(db, topic_id)
    unique_questions = {q.description: q for q in all_questions}.values()
    questions = list(unique_questions)

    selected_questions = []
    current_marks = 0

    while questions and current_marks < max_allocated_marks:
        question = random.choice(questions)
        questions.remove(question)  # Remove used question from pool
       
        if current_marks + question.mark <= max_allocated_marks:
            selected_questions.append(question)
            current_marks += question.mark

    return selected_questions
    


