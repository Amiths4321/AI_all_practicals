from sentence_transformers import CrossEncoder


MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class DocumentReranker:

    def __init__(self, model_name=MODEL_NAME):

        self.model = CrossEncoder(
            model_name
        )

    def rerank(
        self,
        question,
        documents,
        top_n=3
    ):

        if not documents:
            return []

        pairs = []

        for item in documents:

            pairs.append(
                (
                    question,
                    item["document"]
                )
            )

        scores = self.model.predict(
            pairs
        )

        results = []

        for item, score in zip(
            documents,
            scores
        ):

            result = item.copy()

            result["rerank_score"] = float(
                score
            )

            results.append(
                result
            )

        results.sort(
            key=lambda item:
                item["rerank_score"],
            reverse=True
        )

        return results[:top_n]

def filter_by_relevance(
    reranked_documents,
    threshold=0.0
):

    return [
        item
        for item in reranked_documents
        if item["rerank_score"] >= threshold
    ]

    