from rag_judge import judge_answer


question = "Who approves annual leave?"

context = """
Annual leave must be approved by
the employee's manager.
"""

answer = """
Annual leave requires approval from
the employee's manager and must be
submitted 30 days in advance.
"""


result = judge_answer(
    question,
    context,
    answer
)

print(result)