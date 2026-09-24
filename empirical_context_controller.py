import json


class EmpiricalContextController:

    def __init__(
        self,
        policy_file="empirical_context_policy.json",
        fallback_budget=3
    ):
        self.policy_file = policy_file
        self.fallback_budget = fallback_budget

        self.policy = self._load_policy()

    def _load_policy(self):

        with open(
            self.policy_file,
            "r",
            encoding="utf-8"
        ) as file:

            data = json.load(file)

        return data.get("policy", [])

    def choose_budget(
        self,
        evidence_recall,
        reranked_documents
    ):

        if not reranked_documents:

            return {
                "budget": 0,
                "reason": "no_documents"
            }

        # Never generate an answer when required
        # evidence was not completely retrieved.

        if evidence_recall < 1.0:

            return {
                "budget": 0,
                "reason": "insufficient_evidence",
                "evidence_recall": evidence_recall
            }

        if not self.policy:

            return {
                "budget": min(
                    self.fallback_budget,
                    len(reranked_documents)
                ),
                "reason": "fallback_policy"
            }

        # The policy is ordered by increasing budget.
        #
        # Select the smallest empirically validated
        # budget that has full evidence recall.

        valid = [
            row
            for row in self.policy
            if row.get(
                "average_evidence_recall",
                0.0
            ) >= 1.0
        ]

        if not valid:

            return {
                "budget": min(
                    self.fallback_budget,
                    len(reranked_documents)
                ),
                "reason": "no_empirical_budget"
            }

        selected = min(
            valid,
            key=lambda row: (
                row["average_context_characters"],
                row["budget"]
            )
        )

        budget = min(
            selected["budget"],
            len(reranked_documents)
        )

        return {
            "budget": budget,
            "reason": "empirical_policy",
            "policy_quality":
                selected.get(
                    "average_quality",
                    0.0
                ),
            "policy_evidence_recall":
                selected.get(
                    "average_evidence_recall",
                    0.0
                ),
            "policy_context_characters":
                selected.get(
                    "average_context_characters",
                    0
                )
        }