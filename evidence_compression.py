from semantic_evidence import SemanticEvidenceEvaluator


class EvidenceAwareCompressor:

    def __init__(
        self,
        evaluator=None,
        threshold=0.60
    ):
        self.evaluator = (
            evaluator
            or SemanticEvidenceEvaluator()
        )
        self.threshold = threshold

    def compress(
        self,
        required_evidence,
        reranked_documents,
        max_documents=3
    ):
        if not reranked_documents:
            return []

        if not required_evidence:
            return reranked_documents[:max_documents]

        selected = []
        selected_ids = set()

        # First, select documents that cover
        # required evidence.
        for evidence in required_evidence:

            best_document = None
            best_score = -1.0

            for document in reranked_documents:

                if document["id"] in selected_ids:
                    continue

                score = self.evaluator.score(
                    evidence,
                    document["document"]
                )

                if score > best_score:
                    best_score = score
                    best_document = document

            if (
                best_document is not None
                and best_score >= self.threshold
            ):
                selected.append(best_document)
                selected_ids.add(
                    best_document["id"]
                )

            if len(selected) >= max_documents:
                break

        # Fill remaining slots with the
        # highest-ranked reranked documents.
        for document in reranked_documents:

            if len(selected) >= max_documents:
                break

            if document["id"] in selected_ids:
                continue

            selected.append(document)
            selected_ids.add(document["id"])

        return selected