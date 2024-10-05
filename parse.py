#NOTE: V3 function edition implementation

import markdown
from bs4 import BeautifulSoup
import re

import pathlib
import json
import uuid


# TODO: to add larma parse implementation to markdown before this 

def parse_markdown_to_json(markdown_file_path):
    # Read the Markdown file
    with open(markdown_file_path, 'r', encoding='utf-8') as file:
        md_text = file.read()
    
    # Convert Markdown to HTML
    html_content = markdown.markdown(md_text)

    # Parse HTML content
    soup = BeautifulSoup(html_content, 'html.parser')

    # Extract plain text from HTML
    plain_text = soup.get_text()

    # Save the plain text to a file
    text_file_path = "output.txt"
    pathlib.Path(text_file_path).write_text(plain_text, encoding='utf-8')

    print("Markdown has been converted to plain text and saved to", text_file_path)

    with open(text_file_path, 'r', encoding='utf-8') as file:
        text_content = file.read()

    # Split the content into sections based on the question separator
    sections = text_content.split('_QUESTION SEPERATOR___')

    # Initialize an empty list to store questions
    questions = []

    # Regular expression to match question number and content
    question_pattern = re.compile(r'\[(\d+ marks?)\](.*?)(?=\[|\Z)', re.DOTALL)

    # Process each section
    for section in sections:
        questions.append({
            'question_uuid': str(uuid.uuid4()),
            'description': section.strip(),
            'topics': ["test topic 1, test topic 2"],
            'difficulty': 1
        })
        

    # Convert the list of questions to JSON
    json_content = json.dumps(questions, indent=2, ensure_ascii=False)

    # Return the JSON content as a dictionary
    return json.loads(json_content)

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







