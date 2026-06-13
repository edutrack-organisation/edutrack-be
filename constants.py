"""
System-wide constants and prompt templates for OpenAI API interactions.
Defines structured prompts for PDF parsing and question generation.
"""

open_ai_pdf_parsing_prompt = """
    Goal
    Extract questions and their options from a provided PDF question paper. Ensure the extracted content remains true to the original document without modification, paraphrasing, or summarization. The output should be a valid JSON object.

    Taxonomy Reference (for "level" field)
    Use the Revised Bloom's Taxonomy (2001) to categorize the cognitive process required:
    1. Remember: Recall facts and basic concepts.
    2. Understand: Explain ideas or concepts, interpret and summarize information.
    3. Apply: Use information in new situations.
    4. Analyze: Break down information into parts, organize and compare to distinguish fact from opinion.
    5. Evaluate: Justify a decision or viewpoint, assess importance, and make a point by comparing concepts.
    6. Create: Produce original work or ideas by compiling information and proposing alternative solutions.
    
    Return Format
    The extracted content should be structured as follows:
    - Title: Extract the exam title from the document and store it in the "title" field.
    - Questions: Extract all questions and store them in an array "questions", where each question is formatted as:
    - "description": The full question text, including options (each on a new line). You do not have to include the question number at the start of the question text, as well as the marks allocated for this question (if any).
    - "type": Accurately identify and output the type of the question as either "MCQ" (Multiple Choice Question - where only one correct answer is expected) or "MRQ" (Multiple Response Question - where multiple correct answers are expected or allowed).
    - "topics": Set to ["test topic 1", "test topic 2"] (placeholder values for now).
    - "level": A string indicating the Bloom's Taxonomy level. Must be one of: ["Remember", "Understand", "Apply", "Analyze", "Evaluate", "Create"]. Determine this by analyzing the question verbs and outcomes against the "Taxonomy Reference" provided above.
    - "difficulty": An integer value between 1 and 5, determined by the complexity of the question. For determining of complexity of a question, do take into account several factors. 
        1. The question itself. 
        2. The topic(s) that the question belongs to.
        3. The type of questions - whether it is MCQ, MRQ or Short Answer Questions (SAQ).
        4. The marks allocated to that question.
        5. The bloom taxonomy (whether that particular question is a recall question, application question etc)
    - "mark": The total marks for this question. If the description consist of multiple parts, it should be the sum of all the parts for this question.
    - "overview": For each option (e.g., A, B, C, D) in the question, provide: 
        1. "interpretation": What it means to choose this option.
        2. "likely_misunderstanding": Where the students' misunderstanding lies at.
        The structure of "overview" must be a dictionary keyed by option label, each containing these two fields.
    - The output must be a well-formed JSON object that can be directly parsed by a JSON parser.

    Warnings
    Do not modify or add new content.
    Do not provide explanations or summaries of the questions.
    Ensure proper escaping of special characters in JSON formatting.
    Do not reference graphical elements or visuals.
    Maintain the correct order of content, ensuring continuity across pages.
    Do the parsing entirely (This include MCQ and non MCQ/structured questions).

    Context Dump
    This system is designed to extract both multiple-choice (MCQ) and structured questions from a provided PDF document. It ensures that text at the end and beginning of each page is preserved to avoid loss of context. Additionally, if the document contains instructions (e.g., exam guidelines), only the exam title, course code, and year should be extracted, ignoring the rest. Crucially, the system identifies Section boundaries to accurately categorize question types and grouping.
"""


open_ai_generate_question_prompt = """
    Goal
    Generate one single exam question for me. The output should be a valid JSON object.
    {content_of_prompt}

    Return Format
    The generated question should be structured as follows:
    - description: This is the content of the generated itself. (including options)
    - type: A string indicating the exact type of the generated question, strictly either "MCQ" (one correct answer) or "MRQ" (multiple correct answers).
    - topics: This should be the topic of the generated question. For example, Basic Networking Concepts: Host, packet, protocol, throughput, store-and-forward, autonomous system.
    - difficulty: An integer value between 1 and 5, determined by the complexity of the question. For determining of complexity of a question, do take into account several factors. 
        1. The question itself. 
        2. The topic(s) that the question belongs to.
        3. The type of questions - whether it is MCQ, MRQ or Short Answer Questions (SAQ).
        4. The marks allocated to that question.
        5. The bloom taxonomy (whether that particular question is a recall question, application question etc)

    Warning:
    - For the description section, Generate the response with explicit \n newlines so that I can process and render it properly in my application. (Important!)
    - Do not include the answer.

    Context Dump:
    If the type of question generated is MCQ, usually 1 to 2 marks are allocated to it depending on the complexity of the MCQ question.
    If the type of question generated is MRQ, then usually it is ard 2 marks (but still depending on the complexity of the MCQ question).
    For the type of question generated is SAQ, they are usually worth 1-3 marks, depending on complexity of the question.
                
"""

gemini_pdf_parsing_prompt = """
    Goal
    Extract questions and their options from a provided PDF question paper. Ensure the extracted content remains true to the original document without modification, paraphrasing, or summarization. The output should be a valid JSON object.

    Taxonomy Reference (for "level" field)
    Use the Revised Bloom's Taxonomy (2001) to categorize the cognitive process required:
    1. Remember: Recall facts and basic concepts.
    2. Understand: Explain ideas or concepts, interpret and summarize information.
    3. Apply: Use information in new situations.
    4. Analyze: Break down information into parts, organize and compare to distinguish fact from opinion.
    5. Evaluate: Justify a decision or viewpoint, assess importance, and make a point by comparing concepts.
    6. Create: Produce original work or ideas by compiling information and proposing alternative solutions.
    
    Return Format
    The extracted content should be structured as follows:
    - Title: Extract the exam title from the document and store it in the "title" field.
    - Questions: Extract all questions and store them in an array "questions", where each question is formatted as:
    - "description": The full question text, including options. You MUST separate the question text and EACH option with an explicit '\n' newline character so they render on separate lines (e.g., "Question text.\nA. First option\nB. Second option"). You do not have to include the question number at the start of the question text, as well as the marks allocated for this question (if any).
    - "type": Accurately identify and output the type of the question as either "MCQ" (Multiple Choice Question - where only one correct answer is expected) or "MRQ" (Multiple Response Question - where multiple correct answers are expected or allowed).
    - "topics": Set to ["test topic 1", "test topic 2"] (placeholder values for now).
    - "level": A string indicating the Bloom's Taxonomy level. Must be one of: ["Remember", "Understand", "Apply", "Analyze", "Evaluate", "Create"]. Determine this by analyzing the question verbs and outcomes against the "Taxonomy Reference" provided above.
    - "difficulty": An integer value between 1 and 5, determined by the complexity of the question. For determining of complexity of a question, do take into account several factors. 
        1. The question itself. 
        2. The topic(s) that the question belongs to.
        3. The type of questions - whether it is MCQ, MRQ or Short Answer Questions (SAQ).
        4. The marks allocated to that question.
        5. The bloom taxonomy (whether that particular question is a recall question, application question etc)
    - "mark": The total marks for this question. If the description consist of multiple parts, it should be the sum of all the parts for this question.
    - "overview": For each option (e.g., A, B, C, D) in the question, provide: 
        1. "interpretation": What it means to choose this option.
        2. "likely_misunderstanding": Where the students' misunderstanding lies at.
        The structure of "overview" must be a list of objects. Each object must contain a "label" (the option letter, e.g., "A", "B"), "interpretation", and "likely_misunderstanding".
    - The output must be a well-formed JSON object that can be directly parsed by a JSON parser.

    Warnings
    Do not modify or add new content.
    Do not provide explanations or summaries of the questions.
    Ensure proper escaping of special characters in JSON formatting.
    Do not reference graphical elements or visuals.
    Maintain the correct order of content, ensuring continuity across pages.
    Do the parsing entirely (This include MCQ and non MCQ/structured questions).
    EXTRACT QUESTIONS STRICTLY IN THE NUMERICAL ORDER THEY APPEAR. Do not skip, rearrange, or group questions out of sequence.

    Context Dump
    This system is designed to extract both multiple-choice (MCQ) and structured questions from a provided PDF document. It ensures that text at the end and beginning of each page is preserved to avoid loss of context. Additionally, if the document contains instructions (e.g., exam guidelines), only the exam title, course code, and year should be extracted, ignoring the rest.
"""
