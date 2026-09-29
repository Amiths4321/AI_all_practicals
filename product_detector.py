def detect_product(query):

    query = query.lower()

    home_loan_terms = [
        "home loan",
        "home-loan",
        "homeloan",
        "housing loan",
        "house loan",
        "mortgage"
    ]

    personal_loan_terms = [
        "personal loan",
        "personal-loan",
        "personalloan",
        "consumer loan"
    ]

    for term in home_loan_terms:

        if term in query:
            return "home_loan"

    for term in personal_loan_terms:

        if term in query:
            return "personal_loan"

    return None