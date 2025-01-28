# Edutrack Backend

FAST API Backend for Edutrack.

## Important

Please be careful on python dependency management.

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
    pip install -r requirements.txt
    ```

## Starting the server

6. Make sure your .env files are populated with the necessary credentials and remember to source them.
7. Start the server by running `uvicorn main:app --reload`

## Docker Container

8. Make sure you have docker desktop installed.
9. Run the command `docker compose up -d`
10. To set up the database with the necessary tables, run the migrations by following alembic/README.

## PDF Parsing Feature and Topic Identification

11. The PDF Parsing Feature and Topic Identification is linked together.
12. Parsing of PDF -> Run Topic Identification Classification Model -> Display on frontend.

**Steps needed to set up the model locally**

-   You need to first run `python train_model.py` script in bash to train the model.

<br>

Note:
When you install new packages, please do `pipreqs <location where u want to save>`, if you using the new packages as import. This command will automatically add it to `requirements.txt` as a dependency.

Else, please manually add it to `requirements.txt`.

-   I am not using pip freeze because it will include all the **nested dependencies** into `requirements.txt` and this can cause many issues in the parsing library.
