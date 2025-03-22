from io import BytesIO
from pydantic import BaseModel
import unicodedata
import base64
import pypdfium2 as pdfium
import os
import re
from constants import open_ai_pdf_parsing_prompt
from topics import predict_topics
from config import async_client  # Import openAI async client from config.py


# Pydantic model representing a parsed exam question.
class ParsedQuestion(BaseModel):
    description: str
    topics: list[str]
    difficulty: int
    mark: int


# Pydantic model representing a parsed exam paper.
class ParsedPaper(BaseModel):
    title: str
    questions: list[ParsedQuestion]


async def parse_page_with_gpt(base64_images: str) -> str:
    """
    Parse a page of an exam paper using GPT-4o.
    """
    messages = [{"role": "system", "content": open_ai_pdf_parsing_prompt}]

    for base64_image in base64_images:
        messages.append(
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{base64_image}", "detail": "high"},
                    }
                ],
            }
        )

    completion = await async_client.beta.chat.completions.parse(
        model="gpt-4o-2024-08-06",
        messages=messages,
        max_tokens=16384,
        temperature=0.2,  # lower temperature means more deterministic output
        response_format=ParsedPaper,  # enforce structured format of response https://platform.openai.com/docs/guides/structured-outputs?example=structured-data
    )

    return completion.choices[0].message.parsed or ""


"""
General flow: For each page of PDF, convert to images and use GPT-4o to parse the images
Reason: OpenAI API does not handle .pdf files as the UI does. You need to convert the PDF to TXT (if numerical) or PDF to PNG (if image) first. Source: https://community.n8n.io/t/extract-parse-analyse-pdf-using-openai-chatgpt-vision-api/57360/6
"""


async def parse_PDF_OpenAI(pdf_file_path):
    try:
        print("Parsing PDF with OpenAI GPT-4o")  # logging
        pdf = pdfium.PdfDocument(pdf_file_path)
        images = []

        print("Converting PDF to images")  # logging
        for i in range(4):
            page = pdf[i]  # Retrieves the i-th page from the PDF document.
            image = page.render(
                scale=4
            ).to_pil()  # Renders the page into an image, scaling it by a factor of 4 and Converts the rendered image into a PIL image object.
            buffered = (
                BytesIO()
            )  # This line creates an in-memory byte buffer using BytesIO. This buffer will be used to temporarily store the image data in memory.
            image.save(
                buffered, format="JPEG"
            )  # This line saves the PIL image object to the in-memory byte buffer in JPEG format. The buffered object now contains the binary data of the image in JPEG format.
            img_byte = buffered.getvalue()  # Retrieves the binary data of the JPEG image from the byte buffer.
            img_base64 = base64.b64encode(img_byte).decode(
                "utf-8"
            )  # Encodes the binary data into a base64-encoded string.
            images.append(img_base64)  # Appends the base64-encoded string to the images list

        print("Parsing images with OpenAI GPT-4o")  # logging

        parsed_paper = await parse_page_with_gpt(images)
        parsed_paper_dict = parsed_paper.dict()

        extracted_questions = []  # extract and combine the questions itself for topics prediction
        for q in parsed_paper_dict["questions"]:
            extracted_questions.append(q["description"])

        # print(f"Extracted questions: {extracted_questions}")
        # print(predict_topics(extracted_questions))

        predicted_topics = predict_topics(extracted_questions)

        # iterate through the questions and add the topics
        for index, q in enumerate(parsed_paper_dict["questions"]):
            parsed_paper_dict["questions"][index]["topics"] = predicted_topics[index]

        # print("printing python dict after adding")
        return parsed_paper_dict

    except Exception as e:
        print(f"Error parsing PDF: {e}")
        raise Exception("Error parsing PDF")
    finally:
        if pdf:
            pdf.close()
