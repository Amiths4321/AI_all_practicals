import json


class QuestionAdaptiveController:

    def __init__(
        self,
        policy_file="question_budget_policy.json",
        fallback_budget=3
    ):

        self.fallback_budget = fallback_budget

        with open(
            policy_file,
            "r",
            encoding="utf-8"
        ) as file:

            data = json.load(file)

        self.policy = data.get(
            "policy",
            {}
        )

    def choose_budget(
        self,
        question,
        evidence_recall,
        reranked_documents
    ):

        if not reranked_documents:

            return {
                "budget": 0,
                "reason": "no_documents"
            }

        # Evidence gate always has priority.

        if evidence_recall < 1.0:

            return {
                "budget": 0,
                "reason": "insufficient_evidence"
            }

        question_policy = self.policy.get(
            question
        )

        if question_policy is None:

            return {
                "budget": min(
                    self.fallback_budget,
                    len(reranked_documents)
                ),
                "reason": "fallback_question"
            }

        budget = question_policy.get(
            "budget"
        )

        if budget is None:

            return {
                "budget": min(
                    self.fallback_budget,
                    len(reranked_documents)
                ),
                "reason": (
                    "no_successful_historical_budget"
                )
            }

        budget = min(
            budget,
            len(reranked_documents)
        )

        return {
            "budget": budget,
            "reason": (
                "question_specific_policy"
            ),
            "historical_quality":
                question_policy.get(
                    "quality"
                ),
            "historical_evidence_recall":
                question_policy.get(
                    "evidence_recall"
                )
        }