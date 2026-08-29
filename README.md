# Edutrack Backend

FAST API Backend for Edutrack.

## Important

Please be careful on python dependency management. See [Python Dependency Management](#python-dependency-management) section for more details.

## Setting Up

1. Clone the repository:

    ```sh
    git clone https://github.com/edutrack-organisation/edutrack-be
    ```

2. Ensure you have python `Python 3.11.0` installed.
3. Create and activate a virtual environment using bash: Please download Git if you do not have Git Bash.
    ```sh
    python -m venv venv
    source venv/Scripts/activate  # On Windows Bash
    ```
4. Install pipreqs using `pip install pipreqs`.

5. Install the dependencies:
    ```sh
    pip install -r requirements-dev.txt -v
    ```

_Info: **pipreqs**: Python tool that generates requirements.txt files by scanning your project for imports. It's more intelligent than pip freeze because it only includes direct dependencies that are actually used in your code. This helps avoid including unnecessary nested dependencies_

## Training the Topic Identification Model

7. To train the Topic Identification Model, first get the training data from google drive.
   [Link to Google Drive](https://drive.google.com/drive/folders/1bIkp0cBUN5GfgKSmr3pQmJjjNRrm9P7U?usp=sharing)

    - The file should be called training_data.xlsx and should be placed in the top level of the folder hierachy.

8. Run the command `python train_model.py` in bash. This can take quite a while because we also run hyperparameter tuning on it.

NOTE:

-   The PDF Parsing Feature and Topic Identification is linked together.
-   Parsing of PDF -> Run Topic Identification Classification Model -> Display on frontend.
-   **IT IS NECESSARY** to setup the model locally before running the application.

## Docker Container

8. Make sure you have docker desktop installed.
9. Run the command `docker compose up -d`
10. To set up the database with the necessary tables, run the migrations by running the command`alembic upgrade head`. You can read more about migrations at alembic/README.

## Starting the server

6. Make sure your .env files are populated with the necessary credentials and remember to source them.
7. Start the server by running `uvicorn main:app --reload`

## Local PDF question extraction

The `POST /papers/parse` endpoint uses two local components:

1. **Docling** converts a PDF into layout-aware Markdown and applies OCR where needed.
2. **Ollama with Qwen 2.5 7B Instruct** converts that Markdown into schema-validated
   MCQ/MRQ question data.

No exam content is sent to OpenAI by this endpoint.

### macOS setup

1. Install and start the [Ollama macOS application](https://docs.ollama.com/macos).
2. Download the configured model:

    ```sh
    ollama pull qwen2.5:7b-instruct
    ```

3. Copy the local parsing settings from `.env.example` into your `.env`. On an
   Apple Silicon Mac, `DOCLING_DEVICE=mps` enables Metal acceleration.
4. Optionally download Docling models before the first request:

    ```sh
    docling-tools models download
    ```

   Otherwise, Docling downloads its required models on the first parse.

5. Confirm that Ollama is available:

    ```sh
    ollama list
    ```

   The output should include `qwen2.5:7b-instruct`.

### Parse a paper from the frontend

The normal application workflow does not require `curl`. Keep Ollama and this
backend running, then start the EduTrack frontend. The browser uploads the PDF
to this backend's `POST /papers/parse` endpoint, which runs Docling and Qwen
locally.

1. Start Ollama and confirm that the configured Qwen model is installed.
2. Start this backend:

    ```sh
    uvicorn main:app --reload
    ```

3. In the frontend repository, configure the backend address if it is not the
   default `http://127.0.0.1:8000`:

    ```env
    VITE_API_BASE_URL=http://127.0.0.1:8000
    ```

4. Start the frontend:

    ```sh
    pnpm run dev
    ```

5. Open the frontend's `/uploadpdf` page, select a PDF, and keep the page open
   while the local models process it.
6. Review the extracted questions, types, options, topics, marks, difficulty,
   and answer-key indicators in the frontend before saving the paper.

The request path is:

```text
Frontend upload page -> POST /papers/parse -> Docling -> Ollama/Qwen -> Frontend review page
```

The endpoint response retains the fields expected by the frontend
(`description`, `topics`, `difficulty`, and `mark`) and adds structured fields:

```json
{
  "title": "CS2105 Final Examination",
  "questions": [
    {
      "question_number": "1",
      "question_type": "MCQ",
      "description": "Question stem\nA. First option\nB. Second option",
      "options": [
        {"label": "A", "text": "First option"},
        {"label": "B", "text": "Second option"}
      ],
      "correct_answers": [],
      "mark": 1,
      "difficulty": 0,
      "topics": []
    }
  ],
  "all_topics": []
}
```

`correct_answers` remains empty unless the uploaded document includes an answer
key. Missing marks and unassessed difficulty are returned as `0` for lecturer
review; the model is explicitly instructed not to solve questions or infer
topics. Diagram-dependent questions still require manual review because the
configured Qwen model is text-only.

### Tests

Tests use fake Docling and Ollama clients, so they run without model downloads:

```sh
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -m pytest -q
```

## Python Dependency Management

Note:
When you install new packages, please do `pipreqs <location where u want to save>`, if you using the new packages as import. This command will automatically add it to `requirements.txt` as a dependency.

**IF THIS DOESN'T WORK**, please manually add it to `requirements.txt`.

-   I am not using pip freeze because it will include all the **nested dependencies** into `requirements.txt` and this can cause many issues in the parsing library.
-   You can make use of a better dependency management tool if you know of any.
