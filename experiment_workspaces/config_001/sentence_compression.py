from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim


MODEL_NAME = "all-MiniLM-L6-v2"


class SentenceCompressor:

    def __init__(
        self,
        model_name=MODEL_NAME,
        threshold=0.45
    ):
        self.model = SentenceTransformer(model_name)
        self.threshold = threshold

    def split_sentences(self, text):
        sentences = []

        for sentence in text.replace("\n", " ").split("."):
            sentence = sentence.strip()

            if sentence:
                sentences.append(sentence + ".")

        return sentences

    def compress(
        self,
        question,
        documents,
        max_sentences=None
    ):

        if not documents:
            return []

        all_sentences = []

        for document in documents:

            sentences = self.split_sentences(
                document["document"]
            )

            for sentence in sentences:
                all_sentences.append({
                    "sentence": sentence,
                    "source_document": document
                })

        if not all_sentences:
            return []

        sentence_texts = [
            item["sentence"]
            for item in all_sentences
        ]

        embeddings = self.model.encode(
            [question] + sentence_texts,
            convert_to_tensor=True,
            normalize_embeddings=True
        )

        question_embedding = embeddings[0]
        sentence_embeddings = embeddings[1:]

        scores = cos_sim(
            question_embedding,
            sentence_embeddings
        )[0]

        scored = []

        for item, score in zip(
            all_sentences,
            scores
        ):
            result = item.copy()

            result["similarity"] = float(
                score.item()
            )

            scored.append(result)

        scored.sort(
            key=lambda item: item["similarity"],
            reverse=True
        )

        filtered = [
            item
            for item in scored
            if item["similarity"] >= self.threshold
        ]

        if max_sentences is not None:
            filtered = filtered[:max_sentences]

        return filtered

    def reconstruct_documents(
        self,
        compressed_sentences
    ):

        documents = {}

        for item in compressed_sentences:

            source_document = item[
                "source_document"
            ]

            document_id = source_document["id"]

            if document_id not in documents:

                documents[document_id] = {
                    "id": document_id,
                    "document": "",
                    "metadata": source_document.get(
                        "metadata",
                        {}
                    )
                }

            if documents[document_id]["document"]:
                documents[document_id]["document"] += " "

            documents[document_id]["document"] += (
                item["sentence"]
            )

        return list(documents.values())