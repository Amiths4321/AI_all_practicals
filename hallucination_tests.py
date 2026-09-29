from grounding_evaluator import evaluate_grounding


TEST_CASES = [

    {
        "name": "SUPPORTED",
        "context": """
Applicants must have a minimum monthly income
of ₹60,000.
""",
        "answer": """
The minimum monthly income required is ₹60,000.
""",
        "expected_grounded": True
    },

    {
        "name": "HALLUCINATED_NUMBER",
        "context": """
Applicants must have a minimum monthly income
of ₹60,000 .
""",
        "answer": """
The minimum monthly income is ₹60,000
and the processing fee is ₹2,000.
""",
        "expected_grounded": False
    },

    {
        "name": "HALLUCINATED_POLICY",
        "context": """
Applicants must provide valid identity proof.
""",
        "answer": """
Applicants must provide valid identity proof
and a credit score above 750.
""",
        "expected_grounded": False
    }
]


print("\n==============================")
print("HALLUCINATION TEST SUITE")
print("==============================")


passed = 0


for number, test in enumerate(TEST_CASES, start=1):

    result = evaluate_grounding(
        test["answer"],
        test["context"]
    )

    test_passed = (
        result["grounded"]
        == test["expected_grounded"]
    )

    if test_passed:
        passed += 1

    print(f"\nTEST {number}")
    print("------------------------------")
    print("Type:", test["name"])
    print("Grounded:", result["grounded"])
    print("Expected:", test["expected_grounded"])
    print("Result:", "PASS" if test_passed else "FAIL")


print("\n==============================")
print("TEST SUMMARY")
print("==============================")

print("Passed:", passed)
print("Total:", len(TEST_CASES))
print(
    "Pass Rate:",
    round(passed / len(TEST_CASES) * 100, 2),
    "%"
)