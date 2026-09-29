import re


def extract_policy_values(text):

    values = {}

    income_match = re.search(
        r"minimum monthly income.*?₹\s*([\d,]+)",
        text,
        re.IGNORECASE
    )

    if income_match:
        values["minimum_income"] = int(
            income_match.group(1).replace(",", "")
        )

    age_match = re.search(
        r"at least (\d+) years old",
        text,
        re.IGNORECASE
    )

    if age_match:
        values["minimum_age"] = int(
            age_match.group(1)
        )

    maturity_match = re.search(
        r"maximum applicant age.*?(\d+)",
        text,
        re.IGNORECASE
    )

    if maturity_match:
        values["maximum_maturity_age"] = int(
            maturity_match.group(1)
        )

    tenure_match = re.search(
        r"maximum.*?tenure.*?(\d+)\s*years",
        text,
        re.IGNORECASE
    )

    if tenure_match:
        values["maximum_tenure"] = int(
            tenure_match.group(1)
        )

    return values


def detect_policy_conflict(results):

    policy_values = {}

    for result in results:

        document = result.get("document", {})

        product = document.get("product")
        section = document.get("section")
        text = document.get("text", "")

        values = extract_policy_values(text)

        key = (product, section)

        if key not in policy_values:
            policy_values[key] = []

        policy_values[key].append({
            "document": document.get("document"),
            "policy_version": document.get("policy_version"),
            "effective_date": document.get("effective_date"),
            "values": values
        })

    conflicts = []

    for key, policies in policy_values.items():

        for field in [
            "minimum_income",
            "minimum_age",
            "maximum_maturity_age",
            "maximum_tenure"
        ]:

            field_values = []

            for policy in policies:

                if field in policy["values"]:
                    field_values.append(
                        (
                            policy["policy_version"],
                            policy["effective_date"],
                            policy["values"][field]
                        )
                    )

            unique_values = set(
                value[2]
                for value in field_values
            )

            if len(unique_values) > 1:

                conflicts.append({
                    "product": key[0],
                    "section": key[1],
                    "field": field,
                    "values": field_values
                })

    return conflicts