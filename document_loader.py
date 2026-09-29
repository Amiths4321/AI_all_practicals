from pathlib import Path


def load_documents():

    data_folder = Path("data")
    documents = []

    for file_path in sorted(data_folder.glob("*.txt")):

        # -----------------------------
        # PRODUCT
        # -----------------------------

        if "banking_policy" in file_path.name:
            product = "home_loan"

        elif "personal_loan" in file_path.name:
            product = "personal_loan"

        else:
            product = "unknown"

        # -----------------------------
        # POLICY VERSION
        # -----------------------------

        if file_path.name == "banking_policy.txt":

            policy_version = "v1.0"
            effective_date = "2026-01-01"

        elif file_path.name == "banking_policy_v2.txt":

            policy_version = "v2.0"
            effective_date = "2026-06-01"

        elif file_path.name == "personal_loan_policy.txt":

            policy_version = "v1.0"
            effective_date = "2026-01-01"

        else:

            policy_version = "v1.0"
            effective_date = "2026-01-01"

        # -----------------------------
        # READ DOCUMENT
        # -----------------------------

        with open(
            file_path,
            "r",
            encoding="utf-8"
        ) as file:

            text = file.read()

        # -----------------------------
        # CHUNKING
        # -----------------------------

        chunks = [
            chunk.strip()
            for chunk in text.split("\n\n")
            if chunk.strip()
        ]

        # -----------------------------
        # CREATE DOCUMENT RECORDS
        # -----------------------------

        for i, chunk in enumerate(
            chunks,
            start=1
        ):

            first_line = chunk.split("\n")[0].strip()

            if first_line.endswith(":"):
                section = first_line[:-1]
            else:
                section = "General"

            documents.append({

                "document": file_path.name,

                "product": product,

                "document_type": "loan_policy",

                "policy_version": policy_version,

                "effective_date": effective_date,

                "section": section,

                "chunk_id": i,

                "text": chunk
            })

    return documents