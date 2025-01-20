from io import BytesIO
import unicodedata
import base64
import pypdfium2 as pdfium
from openai import OpenAI
import json
import os
import re
from constants import open_ai_pdf_parsing_prompt
from topics import predict_topics

OPEN_AI_API_KEY = os.getenv("OPEN_AI_API_KEY")
client = OpenAI(api_key=OPEN_AI_API_KEY)

def parse_page_with_gpt(base64_images: str) -> str:
    messages=[
        {
            "role": "system",
            "content": open_ai_pdf_parsing_prompt
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


def sanitize_json(json_string):
    print("Sanitising JSON")  # logging 

    # Normalize the text to NFKD form (decomposed form) to handle special characters
    sanitized_json_string = unicodedata.normalize('NFKD', json_string)
    sanitized_json_string = re.sub(r'":\s*"([^"]*)"', r'": "\1"', sanitized_json_string)

    # Remove invalid control characters
    sanitized_json_string = re.sub(r'[\x00-\x1F\x7F]', '', sanitized_json_string)

    return sanitized_json_string


'''
General flow: For each page of PDF, convert to images and use GPT-4o to parse the images
Reason: OpenAI API does not handle .pdf files as the UI does. You need to convert the PDF to TXT (if numerical) or PDF to PNG (if image) first. Source: https://community.n8n.io/t/extract-parse-analyse-pdf-using-openai-chatgpt-vision-api/57360/6
'''
def parse_PDF_OpenAI(pdf_file_path):
    try:
        print("Parsing PDF with OpenAI GPT-4o")  # logging
        pdf = pdfium.PdfDocument(pdf_file_path)
        images = []

        print("Converting PDF to images")  # logging
        for i in range(len(pdf)):
            page = pdf[i]   # Retrieves the i-th page from the PDF document.
            image = page.render(scale=4).to_pil()  # Renders the page into an image, scaling it by a factor of 4 and Converts the rendered image into a PIL image object.
            buffered = BytesIO() # This line creates an in-memory byte buffer using BytesIO. This buffer will be used to temporarily store the image data in memory.
            image.save(buffered, format="JPEG")  # This line saves the PIL image object to the in-memory byte buffer in JPEG format. The buffered object now contains the binary data of the image in JPEG format.
            img_byte = buffered.getvalue()   # Retrieves the binary data of the JPEG image from the byte buffer.
            img_base64 = base64.b64encode(img_byte).decode("utf-8") # Encodes the binary data into a base64-encoded string.
            images.append(img_base64) # Appends the base64-encoded string to the images list

        print("Parsing images with OpenAI GPT-4o")  # logging

        parsed = parse_page_with_gpt(images)
        sanitised_parsed = sanitize_json(parsed)
        python_dict = json.loads(sanitised_parsed)

        extracted_questions = []  # extract and combine the questions itself for topics prediction
        for q in python_dict["questions"]:
            extracted_questions.append(q['description'])
        
        # print(f"Extracted questions: {extracted_questions}")
        # print(predict_topics(extracted_questions))
        
        predicted_topics = predict_topics(extracted_questions)

        # iterate through the questions and add the topics
        for index, q in enumerate(python_dict["questions"]):
            python_dict["questions"][index]["topics"] = predicted_topics[index]
        
        # print("printing python dict after adding")
        # print(python_dict)
        return python_dict

    except Exception as e:
        print(f"Error parsing PDF: {e}")
        print(sanitised_parsed)
        raise Exception("Error parsing PDF")
    finally:
        if pdf:
            pdf.close()
    
