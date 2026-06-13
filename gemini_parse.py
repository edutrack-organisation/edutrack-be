from io import BytesIO
from pydantic import BaseModel
import pypdfium2 as pdfium
import os
# You can safely reuse the same prompt
from constants import gemini_pdf_parsing_prompt
from topics import predict_topics
from config import gemini_client
from typing import Optional, Dict
from google.genai import types
import re


class GeminiOptionOverview(BaseModel):
    label: str
    interpretation: str
    likely_misunderstanding: str


class GeminiParsedQuestion(BaseModel):
    description: str
    topics: list[str]
    level: str
    difficulty: int
    mark: int
    overview: Optional[list[GeminiOptionOverview]] = None


class ParsedPaper(BaseModel):
    title: str
    questions: list[GeminiParsedQuestion]


async def parse_page_with_gemini(image_bytes_list: list[bytes]) -> dict:
    """
    Parse the exam paper images using Gemini 2.5 Pro.
    """
    # 1. Start the contents array with the system instruction prompt
    contents = [gemini_pdf_parsing_prompt]
    
    for index, img_bytes in enumerate(image_bytes_list):
        # <-- Anchors the reading order
        contents.append(f"--- Document Page {index + 1} ---")
        contents.append(
            types.Part.from_bytes(data=img_bytes, mime_type="image/jpeg")
        )

    # 2. Append each page as an image part
    for img_bytes in image_bytes_list:
        contents.append(
            types.Part.from_bytes(data=img_bytes, mime_type="image/jpeg")
        )

    # 3. Call the async client
    response = await gemini_client.aio.models.generate_content(
        model="gemini-2.5-flash",  # faster and cheaper
        contents=contents,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=ParsedPaper,  # Natively enforces your Pydantic structure
            temperature=0.2,
        )
    )

    if response.parsed:
        parsed_dict = response.parsed.model_dump()
    else:
        import json
        parsed_dict = json.loads(response.text)

    # Transform Gemini's Array back to GPT's dictionary format
    for question in parsed_dict.get("questions", []):

        # 1. Force explicit newlines before options A. through E. if they are missing
        # This looks for A., B., C., D., or E. preceded by anything other than a newline, and injects one.
        question["description"] = re.sub(
            r'(?<!\n)([A-E]\.)\s', r'\n\1 ', question["description"]).strip()

        # 2. Fix the overview dictionary structure (from our previous step)
        if question.get("overview"):
            transformed_overview = {}
            for item in question["overview"]:
                label = item.pop("label")
                transformed_overview[label] = item

            question["overview"] = transformed_overview

    return parsed_dict


async def parse_PDF_Gemini(pdf_file_path):
    try:
        print("Parsing PDF with Gemini")
        pdf = pdfium.PdfDocument(pdf_file_path)
        image_bytes_list = []

        print("Converting PDF to images")
        for i in range(len(pdf)):
            page = pdf[i]
            image = page.render(scale=4).to_pil()
            buffered = BytesIO()
            image.save(buffered, format="JPEG")
            # Store raw bytes instead of base64 string
            image_bytes_list.append(buffered.getvalue())

        print("Parsing images with Gemini")
        parsed_paper_dict = await parse_page_with_gemini(image_bytes_list)

        # Extract and combine the questions for topics prediction
        extracted_questions = [q["description"]
                               for q in parsed_paper_dict["questions"]]
        predicted_topics = predict_topics(extracted_questions)

        # Iterate through the questions and add the topics
        for index, q in enumerate(parsed_paper_dict["questions"]):
            parsed_paper_dict["questions"][index]["topics"] = predicted_topics[index]

        return parsed_paper_dict

    except Exception as e:
        print(f"Error parsing PDF: {e}")
        raise Exception("Error parsing PDF")
    finally:
        if 'pdf' in locals() and pdf:
            pdf.close()
