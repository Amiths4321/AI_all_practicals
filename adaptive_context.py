from context_builder import ContextBuilder


class AdaptiveContextPolicy:

    def __init__(
        self,
        context_builder,
        evidence_evaluator,
        budgets=(1, 2, 3, 5),
        evidence_threshold=0.60
    ):
        self.context_builder = context_builder
        self.evidence_evaluator = evidence_evaluator
        self.budgets = sorted(budgets)
        self.evidence_threshold = evidence_threshold

    def choose_budget(
        self,
        required_evidence,
        reranked_documents
    ):
        if not reranked_documents:
            return {
                "budget": 0,
                "reason": "no_documents"
            }

        if not required_evidence:
            return {
                "budget": min(2, len(reranked_documents)),
                "reason": "no_required_evidence"
            }

        evidence_result = self.evidence_evaluator.evaluate(
            required_evidence=required_evidence,
            retrieved_documents=reranked_documents
        )

        evidence_recall = evidence_result["evidence_recall"]

        # Cannot safely answer if required evidence is missing.
        if evidence_recall < 1.0:
            return {
                "budget": 0,
                "reason": "insufficient_evidence",
                "evidence_recall": evidence_recall,
                "evidence_matches": evidence_result["matches"]
            }

        # Find the strongest evidence match.
        best_score = 0.0

        for match in evidence_result["matches"]:
            best_score = max(
                best_score,
                match["score"]
            )

        # Strong evidence: use a small context.
        if best_score >= 0.75:
            budget = 1

        # Good evidence: use a little more context.
        elif best_score >= 0.65:
            budget = 2

        # Weaker but sufficient evidence: use more context.
        else:
            budget = 3

        budget = min(
            budget,
            len(reranked_documents)
        )

        return {
            "budget": budget,
            "reason": "evidence_sufficient",
            "evidence_recall": evidence_recall,
            "best_evidence_score": best_score,
            "evidence_matches": evidence_result["matches"]
        }

    def build_context(
        self,
        question,
        required_evidence,
        reranked_documents
    ):
        decision = self.choose_budget(
            required_evidence=required_evidence,
            reranked_documents=reranked_documents
        )

        if decision["budget"] == 0:
            return {
                "decision": decision,
                "documents": []
            }

        documents = self.context_builder.build(
            strategy="evidence_first",
            question=question,
            reranked_documents=reranked_documents,
            required_evidence=required_evidence,
            max_documents=decision["budget"]
        )

        return {
            "decision": decision,
            "documents": documents
        }