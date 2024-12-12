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
3. Create and activate a virtual environment using bash:
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

`uvicorn main:app --reload`

Note:
When you install new packages, please do `pipreqs <location where u want to save>`, if you using the new packages as import. This command will automatically add it to `requirements.txt` as a dependency.

Else, please manually add it to `requirements.txt`.

-   I am not using pip freeze because it will include all the **nested dependencies** into `requirements.txt` and this can cause many issues in the parsing library.
