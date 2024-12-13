from io import BytesIO
import base64
import pypdfium2 as pdfium
from openai import OpenAI
import json
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
OPEN_AI_API = os.getenv("OPEN_AI_KEY")
client = OpenAI(api_key=OPEN_AI_API)

def parse_page_with_gpt(base64_images: str) -> str:
    messages=[
        {
            "role": "system",
            "content": """You are a system to extract questions and it's options.
            Please only extract the information given in the document.
            Do not answer with any additional explanations or text.
            Do not provide a summary of the questions asked.
            You should just parse the questions and its options and display them. 
            Please do not reference graphical elements or visualization in your answer. Just answer with the extracted text.
            Make sure that text at the end of the page as well as text at the beginning of the page are also at the end and beginning of your extraction - as this might be continuations of the previous and next page.
            If the page is the instruction of the paper, please extract the paper title such as exam name, course code, and year and ignore the rest.

            The questions and options should be together in a field called description and should be able to render properly as text (string) in the frontend. 

            Do this entirely. (This include MCQ and non MCQ/structured questions)

            Extract the following information from the provided text (extracted from a PDF question paper) and format it as JSON:

            You are tasked with extracting information from a text and formatting it as a JSON object.

            Requirements:
            1. Extract the title of the paper and insert it as the "title" field in the JSON.
            2. Extract all the questions from the text. Each question should:
            - Be stored as an object in the "questions" array.
            - Have a "description" field containing the full question text (including options if applicable). Each options should be presented on a new line (see below)
            
            <Question>
            <Option A>
            <Option B>
            <Option C>
            <Option D>
            and so on. 
            
            - Have a "topics" field set to ["test topic 1", "test topic 2"] (placeholder values for now).
            - Have a "difficulty" field set to 1 (placeholder value for now).

            Important:
            - Return the output as a valid JSON object, not as a code snippet.
            - Ensure the JSON object is well-formed and can be directly parsed by a JSON parser.
            - The invalid escape sequence in the JSON should be escaped properly.
            - All strings should be enclosed in double quotes.

            Output Format:
            {
                "title": "Extracted title from the paper",
                "questions": [
                    {
                        "description": "Full question text with options if applicable",
                        "topics": ["test topic 1", "test topic 2"],
                        "difficulty": 1
                    }
                ]
            }
        """
        }]
    
    for base64_image in base64_images:
        messages.append(
            {
                "role": "user",
                "content": [
                    {"type": "image_url", 
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}",
                            "detail": "low"
                        }
                    }
                ],
            }
        )

    response = client.chat.completions.create(
        model="gpt-4o-2024-08-06",
        messages=messages,
        max_tokens=16384,
    )

    return response.choices[0].message.content or ""



def open_AI_parse(pdf_file_path):
    pdf = pdfium.PdfDocument(pdf_file_path)
    images = []
    for i in range(len(pdf)):
        page = pdf[i]
        image = page.render(scale=4).to_pil()
        buffered = BytesIO()
        image.save(buffered, format="JPEG")
        img_byte = buffered.getvalue()
        img_base64 = base64.b64encode(img_byte).decode("utf-8")
        images.append(img_base64)

        if i == 2:
            break

    text_of_pages = parse_page_with_gpt(images)
    print(text_of_pages)

    # close to pdf file
    pdf.close()

    return json.loads(text_of_pages)