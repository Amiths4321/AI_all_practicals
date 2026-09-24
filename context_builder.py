from context_compression import compress_context
from evidence_compression import EvidenceAwareCompressor
from sentence_compression import SentenceCompressor


class ContextBuilder:

    def __init__(
        self,
        evidence_threshold=0.60,
        sentence_threshold=0.45
    ):

        self.evidence_compressor = (
            EvidenceAwareCompressor(
                threshold=evidence_threshold
            )
        )

        self.sentence_compressor = (
            SentenceCompressor(
                threshold=sentence_threshold
            )
        )

    def build(
        self,
        strategy,
        question,
        reranked_documents,
        required_evidence,
        max_documents=3
    ):

        required_evidence = (
            required_evidence or []
        )

        if not reranked_documents:
            return []

        # -----------------------------------------
        # 1. Original
        # -----------------------------------------

        if strategy == "original":

            return reranked_documents

        # -----------------------------------------
        # 2. Top-N
        # -----------------------------------------

        if strategy == "top_n":

            return compress_context(
                reranked_documents=reranked_documents,
                max_documents=max_documents
            )

        # -----------------------------------------
        # 3. Evidence-aware
        # -----------------------------------------

        if strategy == "evidence_aware":

            return self.evidence_compressor.compress(
                required_evidence=required_evidence,
                reranked_documents=reranked_documents,
                max_documents=max_documents
            )

        # -----------------------------------------
        # 4. Evidence-first
        # -----------------------------------------

        if strategy == "evidence_first":

            return self._evidence_first(
                question=question,
                documents=reranked_documents,
                required_evidence=required_evidence
            )

        # -----------------------------------------
        # 5. Evidence-last
        # -----------------------------------------

        if strategy == "evidence_last":

            return self._evidence_last(
                question=question,
                documents=reranked_documents,
                required_evidence=required_evidence
            )

        # -----------------------------------------
        # 6. Sentence compression
        # -----------------------------------------

        if strategy == "sentence":

            compressed_sentences = (
                self.sentence_compressor.compress(
                    question=question,
                    documents=reranked_documents
                )
            )

            return (
                self.sentence_compressor
                .reconstruct_documents(
                    compressed_sentences
                )
            )

        raise ValueError(
            f"Unknown context strategy: {strategy}"
        )

    def _find_evidence(
          self,
          documents,
          required_evidence
    ):

          if not required_evidence:
            return None

          best_document = None
          best_score = -1.0

          for document in documents:

                    document_score = 0.0

                    for evidence in required_evidence:

                              score = self.sentence_compressor.model.encode(
                              [evidence, document["document"]],
                              convert_to_tensor=True,
                              normalize_embeddings=True
                    )

                              similarity = float(
                              score[0] @ score[1]
                    )

                              document_score = max(
                              document_score,
                              similarity
                    )

                    if document_score > best_score:

                              best_score = document_score
                              best_document = document

          return best_document

    def _evidence_first(
        self,
        question,
        documents,
        required_evidence
    ):

        evidence = self._find_evidence(
            documents,
            required_evidence
        )

        if evidence is None:
            return documents

        evidence_id = evidence["id"]

        return [
            evidence
        ] + [
            document
            for document in documents
            if document["id"] != evidence_id
        ]

    def _evidence_last(
        self,
        question,
        documents,
        required_evidence
    ):

        evidence = self._find_evidence(
            documents,
            required_evidence
        )

        if evidence is None:
            return documents

        evidence_id = evidence["id"]

        return [
            document
            for document in documents
            if document["id"] != evidence_id
        ] + [
            evidence
        ]