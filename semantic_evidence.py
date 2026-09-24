from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim


MODEL_NAME = "all-MiniLM-L6-v2"


class SemanticEvidenceEvaluator:

    def __init__(
        self,
        model_name=MODEL_NAME
    ):
        self.model = SentenceTransformer(
            model_name
        )

    def score(
        self,
        evidence,
        document
    ):
        embeddings = self.model.encode(
            [
                evidence,
                document
            ],
            convert_to_tensor=True,
            normalize_embeddings=True
        )

        similarity = cos_sim(
            embeddings[0],
            embeddings[1]
        )

        return float(
            similarity.item()
        )

    def evaluate(
        self,
        required_evidence,
        retrieved_documents
    ):
        if not required_evidence:
            return {
                "evidence_recall": 1.0,
                "matches": []
            }

        if not retrieved_documents:
            return {
                "evidence_recall": 0.0,
                "matches": []
            }

        documents = [
            item["document"]
            for item in retrieved_documents
        ]

        document_embeddings = self.model.encode(
            documents,
            convert_to_tensor=True,
            normalize_embeddings=True
        )

        matches = []
        found = 0

        for evidence in required_evidence:

            evidence_embedding = self.model.encode(
                evidence,
                convert_to_tensor=True,
                normalize_embeddings=True
            )

            scores = cos_sim(
                evidence_embedding,
                document_embeddings
            )[0]

            best_index = int(
                scores.argmax().item()
            )

            best_score = float(
                scores[best_index].item()
            )

            best_document = retrieved_documents[
                best_index
            ]

            matches.append({
                "evidence": evidence,
                "score": best_score,
                "document_id": best_document["id"],
                "source": best_document[
                    "metadata"
                ].get("source")
            })

            if best_score >= 0.60:
                found += 1

        return {
            "evidence_recall": (
                found / len(required_evidence)
            ),
            "matches": matches
        }