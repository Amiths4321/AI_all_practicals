from rank_bm25 import BM25Okapi


class HybridRetriever:

    def __init__(self, documents):

        self.documents = documents

        self.bm25 = BM25Okapi(
            [
                self.tokenize(document["document"])
                for document in documents
            ]
        )

    @staticmethod
    def tokenize(text):

        return text.lower().split()

    def keyword_search(
        self,
        question,
        top_k=10
    ):

        query_tokens = self.tokenize(
            question
        )

        scores = self.bm25.get_scores(
            query_tokens
        )

        ranked_indexes = sorted(
            range(len(scores)),
            key=lambda index: scores[index],
            reverse=True
        )

        results = []

        for index in ranked_indexes[:top_k]:

            candidate = self.documents[index]

            results.append(
                {
                    "id": candidate["id"],
                    "document": candidate["document"],
                    "metadata": candidate.get(
                        "metadata",
                        {}
                    ),
                    "score": float(
                        scores[index]
                    )
                }
            )

        return results

    def reciprocal_rank_fusion(
        self,
        vector_results,
        keyword_results,
        top_k=10,
        rrf_k=60
    ):

        combined = {}

        # --------------------------------------
        # VECTOR RESULTS
        # --------------------------------------

        for rank, item in enumerate(
            vector_results,
            start=1
        ):

            document_id = item["id"]

            if document_id not in combined:

                combined[document_id] = {
                    "id": document_id,
                    "document": item["document"],
                    "metadata": item.get(
                        "metadata",
                        {}
                    ),
                    "vector_rank": None,
                    "keyword_rank": None,
                    "rrf_score": 0.0
                }

            combined[
                document_id
            ]["vector_rank"] = rank

            combined[
                document_id
            ]["rrf_score"] += (
                1 / (rrf_k + rank)
            )

        # --------------------------------------
        # KEYWORD RESULTS
        # --------------------------------------

        for rank, item in enumerate(
            keyword_results,
            start=1
        ):

            document_id = item["id"]

            if document_id not in combined:

                combined[document_id] = {
                    "id": document_id,
                    "document": item["document"],
                    "metadata": item.get(
                        "metadata",
                        {}
                    ),
                    "vector_rank": None,
                    "keyword_rank": None,
                    "rrf_score": 0.0
                }

            combined[
                document_id
            ]["keyword_rank"] = rank

            combined[
                document_id
            ]["rrf_score"] += (
                1 / (rrf_k + rank)
            )

        # --------------------------------------
        # SORT
        # --------------------------------------

        results = sorted(
            combined.values(),
            key=lambda item: item[
                "rrf_score"
            ],
            reverse=True
        )

        return results[:top_k]

    def search(
        self,
        question,
        vector_results,
        top_k=10
    ):

        keyword_results = self.keyword_search(
            question=question,
            top_k=top_k
        )

        return self.reciprocal_rank_fusion(
            vector_results=vector_results,
            keyword_results=keyword_results,
            top_k=top_k
        )