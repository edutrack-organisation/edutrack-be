# declare constants for the project

open_ai_pdf_parsing_prompt = """
    Goal
    Extract questions and their options from a provided PDF question paper. Ensure the extracted content remains true to the original document without modification, paraphrasing, or summarization. The output should be a valid JSON object.

    Return Format
    The extracted content should be structured as follows:
    - Title: Extract the exam title from the document and store it in the "title" field.
    - Questions: Extract all questions and store them in an array "questions", where each question is formatted as:
    - "description": The full question text, including options (each on a new line). You do not have to include the question number at the start of the question text, as well as the marks allocated for this question (if any).
    - "topics": Set to ["test topic 1", "test topic 2"] (placeholder values for now).
    - "difficulty": An integer value between 1 and 5, determined by the complexity of the question. For determining of complexity of a question, do take into account several factors. 
        1. The question itself. 
        2. The topic(s) that the question belongs to.
        3. The type of questions - whether it is MCQ, MRQ or Short Answer Questions (SAQ).
        4. The marks allocated to that question.
        5. The bloom taxonomy (whether that particular question is a recall question, application question etc)
    - The output must be a well-formed JSON object that can be directly parsed by a JSON parser.

    Warnings
    Do not modify or add new content.
    Do not provide explanations or summaries of the questions.
    Ensure proper escaping of special characters in JSON formatting.
    Do not reference graphical elements or visuals.
    Maintain the correct order of content, ensuring continuity across pages.
    Do the parsing entirely (This include MCQ and non MCQ/structured questions).

    Context Dump
    This system is designed to extract both multiple-choice (MCQ) and structured questions from a provided PDF document. It ensures that text at the end and beginning of each page is preserved to avoid loss of context. Additionally, if the document contains instructions (e.g., exam guidelines), only the exam title, course code, and year should be extracted, ignoring the rest.
"""