from pairwise_judge import pairwise_judge


question = "Who approves annual leave?"


context = """
Employees must submit annual leave requests to their manager.
Annual leave must be approved by the employee's manager.
HR maintains the leave records.
"""


answer_a = """
Annual leave must be approved by the employee's manager.
"""


answer_b = """
Annual leave is automatically approved for 30 days every year.
Employees do not need manager approval.
"""


result = pairwise_judge(
    question=question,
    context=context,
    answer_a=answer_a,
    answer_b=answer_b
)

reversed_result = pairwise_judge(
    question=question,
    context=context,
    answer_a=answer_b,
    answer_b=answer_a
)

print("\nREVERSED ORDER")
print("=" * 60)

for key, value in reversed_result.items():
    print(f"{key}: {value}")

print("\nPAIRWISE JUDGEMENT")
print("=" * 60)

for key, value in result.items():
    print(f"{key}: {value}")