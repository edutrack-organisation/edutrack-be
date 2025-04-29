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
    pip install -r requirements.txt -v
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

## Python Dependency Management

Note:
When you install new packages, please do `pipreqs <location where u want to save>`, if you using the new packages as import. This command will automatically add it to `requirements.txt` as a dependency.

**IF THIS DOESN'T WORK**, please manually add it to `requirements.txt`.

-   I am not using pip freeze because it will include all the **nested dependencies** into `requirements.txt` and this can cause many issues in the parsing library.
-   You can make use of a better dependency management tool if you know of any.
