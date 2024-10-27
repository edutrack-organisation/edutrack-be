"""
This parse.py file contains the code for parsing PDF files and converting them to structured data formats like Markdown and JSON.
When the FastAPI application's parsePDF endpoint receives a PDF file from the frontend, it uses the `parse_pdf` function to extract the content and convert it to a JSON format.
The parsed JSON data can then be send back to the frontend. 

Flow of the parsing process:
1. Receive PDF from endpoint.
2. Parse the PDF file to Markdown format using the `llama_parse_pdf_to_markdown` function.
3. Convert the Markdown content to JSON format using the `parse_markdown_to_json` function.
4. Return the JSON data to the frontend.

We make use of LlamaParse, a document parsing platform that leverages Large Language Models (LLMs) to extract structured data from documents like PDFs. This is giving us the best parsing performance and accuracy. (so far).
"""

#NOTE: V3 function edition implementation
# https://github.com/run-llama/llama_parse/blob/main/examples/demo_advanced.ipynb
import markdown
import re
import json
import nest_asyncio

from bs4 import BeautifulSoup
from dotenv import load_dotenv

nest_asyncio.apply()

# Load environment variables from .env file
load_dotenv()

# bring in deps
from llama_parse import LlamaParse
from llama_index.core import SimpleDirectoryReader

# Setting up of LlamaParse
parser = LlamaParse(
    result_type="markdown",  # "markdown" and "text" are available
    # Prompt engineering: provide a prompt to help the Llama Parse model understand the task and help it generate better results
    parsing_instruction = "Seperate the questions with ____QUESTION SEPERATOR______, make sure subquestions such as (a), (b) to be part of the same question.",
    premium_mode=True,
    # target_pages="0"   # Optional: specify the page number to parse
)
file_extractor = {".pdf": parser}

# LlamaParse function
def llama_parse_pdf_to_markdown(pdf_file_path):
    # Convert PDF to Markdown using llama_parse
    documents = SimpleDirectoryReader(input_files=[pdf_file_path], file_extractor=file_extractor).load_data()

    # Combine the documents(pages) into a single document
    combined_document = ""
    for document in documents:
        combined_document += document.get_text() + "\n"  # Add a newline for separation

    return combined_document

# markdown to json function
def parse_markdown_to_json(md_text):
    # Convert Markdown to HTML
    html_content = markdown.markdown(md_text)

    # Parse HTML content
    soup = BeautifulSoup(html_content, 'html.parser')

    # Extract plain text from HTML
    plain_text = soup.get_text()

    # Extracting the title and semester (#TODO: not made use yet)
    # Regular expressions to match course title and semester
    title_pattern = re.compile(r'CS\d{4}\s*[-–—]\s*[A-Za-z ]+', re.IGNORECASE)
    semester_pattern = re.compile(r'Semester \d, \d{4}/\d{4}', re.IGNORECASE)

    course_title = title_pattern.search(plain_text)
    semester = semester_pattern.search(plain_text)

    # Extract the matched text or set to empty string if not found
    course_title = course_title.group(0) if course_title else "<Course Title>"
    semester = semester.group(0) if semester else "<Semester>"
    title = f"{course_title} - {semester}"

    # Extracting the questions
    #NOTE: need to double check this, seems to be different from markdown
    sections = plain_text.split('_QUESTION SEPERATOR___')

    # Initialize an empty list to store questions
    questions = []

    # Regular expression to match question number and content
    # question_pattern = re.compile(r'\[(\d+ marks?)\](.*?)(?=\[|\Z)', re.DOTALL)

    # Process each section
    for section in sections:
        questions.append({
            'description': section.strip(),
            'topics': ["test topic 1", "test topic 2"],
            'difficulty': 1
        })
        
    parsed_paper = {
        'title': title,
        'questions': questions
    }
    
    # Convert the list of questions to JSON
    json_parsed_paper = json.dumps(parsed_paper, indent=2, ensure_ascii=False)

    # Return the JSON content as a dictionary
    return json.loads(json_parsed_paper)

# parse pdf function
def parse_pdf(pdf_file_path):
    # Convert PDF to Markdown
    md_text = llama_parse_pdf_to_markdown(pdf_file_path)

    # open the mark down file, for testing beyond limits
    # with open(pdf_file_path, 'r', encoding='utf-8') as file:
        # md_text = file.read()
        
    # Convert Markdown to JSON
    return parse_markdown_to_json(md_text)



# #NOTE: V2 parsing implementation using llama parse

# # I want to parse a given PDF and display the content in a structured format.
# # https://pypi.org/project/pymupdf4llm/

# import pymupdf4llm
# import markdown_to_json
# import markdown
# from bs4 import BeautifulSoup
# import re

# import pathlib
# import json
# import uuid

# # Replace with your Markdown file path
# markdown_file_path = r"C:\Users\User\workspace\edutrack-repo\edutrack-be\papers\test_markdown_two.md"

# # Read the Markdown file
# with open(markdown_file_path, 'r', encoding='utf-8') as file:
#     md_text = file.read()

# # Convert Markdown to HTML
# html_content = markdown.markdown(md_text)

# # Parse HTML content
# soup = BeautifulSoup(html_content, 'html.parser')

# # Extract plain text from HTML
# plain_text = soup.get_text()

# # Save the plain text to a file
# text_file_path = "output.txt"
# pathlib.Path(text_file_path).write_text(plain_text, encoding='utf-8')

# print("Markdown has been converted to plain text and saved to", text_file_path)

# with open(text_file_path, 'r', encoding='utf-8') as file:
#     text_content = file.read()

# # Split the content into sections based on the question separator
# sections = text_content.split('_QUESTION SEPERATOR___')

# # Initialize an empty list to store questions
# questions = []

# # Regular expression to match question number and content
# question_pattern = re.compile(r'\[(\d+ marks?)\](.*?)(?=\[|\Z)', re.DOTALL)

# # Process each section
# for i, section in enumerate(sections):
#     questions.append({
#         'question_uuid': str(uuid.uuid4()),
#         'description': section.strip(),
#         'topics': ["test topic 1, test topic 2"],
#         'difficulty': 1
#     })
    

# # Convert the list of questions to JSON
# json_content = json.dumps(questions, indent=2, ensure_ascii=False)

# # Save JSON to a file
# json_file_path = 'questions.json'
# with open(json_file_path, 'w', encoding='utf-8') as file:
#     file.write(json_content)

# print("Questions have been extracted and saved to", json_file_path)

# # i want to unparse the json file and display the content in a structured format

# json_file_path = 'questions.json'
# with open(json_file_path, 'r', encoding='utf-8') as file:
#     json_content = file.read()

# # Parse JSON content
# questions = json.loads(json_content)

# i = 0
# for question in questions:
#     # print(f"Question:\n{question['content']}\n")
#     print(question["description"])
#     i += 1
#     if i == 3:
#         break







# NOTE: V1 parsing implementation
# # I want to parse a given PDF and display the content in a structured format.
# # https://pypi.org/project/pymupdf4llm/

# import pymupdf4llm
# import markdown_to_json
# import markdown
# from bs4 import BeautifulSoup

# # NOTE: convert to markdown
# file_path = r"C:\Users\User\workspace\edutrack-repo\edutrack-be\papers\CS2105_Endterm_2022_23_sem2.pdf"  # Replace with your PDF file path

# # NOTE: convert to markdown

# md_text = pymupdf4llm.to_markdown(file_path)

# # now work with the markdown text, e.g. store as a UTF8-encoded file
# import pathlib
# pathlib.Path("output.md").write_bytes(md_text.encode())

# # Convert Markdown to HTML
# html_content = markdown.markdown(md_text)

# # Parse HTML content
# soup = BeautifulSoup(html_content, 'html.parser')

# # Extract plain text from HTML
# plain_text = soup.get_text()

# # Save the plain text to a file
# text_file_path = "output.txt"
# pathlib.Path(text_file_path).write_text(plain_text, encoding='utf-8')

# print("Markdown has been converted to plain text and saved to", text_file_path)

# # #NOTE: convert from pdf to text 
# # import fitz  # PyMuPDF
# # import pathlib
# # # Read the text file
# # text_file_path = 'output.txt'  # Replace with your text file path

# # def pdf_to_text(file_path):
# #     # Open the PDF file
# #     pdf_document = fitz.open(file_path)
    
# #     # Extract text from each page
# #     text_content = ""
# #     for page_num in range(len(pdf_document)):
# #         page = pdf_document.load_page(page_num)
# #         text_content += page.get_text("text") + "\n"
    
# #     return text_content

# # # Example usage
# # file_path = r"C:\Users\User\workspace\edutrack-repo\edutrack-be\2021SEM1-CS2105.pdf"  # Replace with your PDF file path
# # text_content = pdf_to_text(file_path)

# # # Save the extracted text to a file
# # text_file_path = "output.txt"
# # pathlib.Path(text_file_path).write_text(text_content, encoding='utf-8')

# # print("PDF has been converted to text and saved to", text_file_path)

# # NOTE: cleaning and converting to json
# with open(text_file_path, 'r', encoding='utf-8') as file:
#     text_content = file.read()

# import re

# # Remove markdown table formatting and other markdown-specific syntax
# # Remove markdown table formatting and other markdown-specific syntax
# text_content = re.sub(r'\|---.*?\|', '', text_content)  # Remove table headers
# text_content = re.sub(r'\|.*?\|', '', text_content)  # Remove table rows
# text_content = re.sub(r'#+\s', '', text_content)  # Remove headers
# text_content = re.sub(r'\*\*.*?\*\*', '', text_content)  # Remove bold text
# text_content = re.sub(r'\*.*?\*', '', text_content)  # Remove italic text
# text_content = re.sub(r'`.*?`', '', text_content)  # Remove inline code
# text_content = re.sub(r'\[.*?\]\(.*?\)', '', text_content)  # Remove links
# text_content = re.sub(r'^\s*\|+\s*\|+\s*$', '', text_content, flags=re.MULTILINE)  # Remove lines with only pipes


# # Use regular expressions to split the questions by number and filter out options
# pattern = re.compile(r'(\d+)\.\s*(.*?)(?=\n\d+\.|\Z)', re.DOTALL)
# matches = pattern.findall(text_content)

# # Store questions in an array
# questions = []
# for match in matches:
#     question_number, question_content = match
#     # Remove options from the question content
#     question_content = re.sub(r'\n[^\n]*\n', '\n', question_content)
#     # Remove newline characters
#     question_content = question_content.replace('\n', ' ')
#     questions.append({
#         'question_number': question_number,
#         'content': question_content.strip()
#     })

# import json
# # Convert the array to JSON format
# json_content = json.dumps(questions, indent=2, ensure_ascii=False)

# # Save JSON to a file
# json_file_path = 'questions.json'  # Replace with your desired JSON file path
# with open(json_file_path, 'w', encoding='utf-8') as file:
#     file.write(json_content)


# print("Questions have been extracted and saved to", json_file_path)







