from datetime import datetime

from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue


COLLECTION_NAME = "banking_policies"

TOP_K = 3
SCORE_THRESHOLD = 0.40


client = QdrantClient(
    path="qdrant_storage"
)

model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)


def get_latest_effective_date(product):

    results, _ = client.scroll(
        collection_name=COLLECTION_NAME,

        scroll_filter=Filter(
            must=[
                FieldCondition(
                    key="product",
                    match=MatchValue(
                        value=product
                    )
                )
            ]
        ),

        limit=1000,
        with_payload=True,
        with_vectors=False
    )

    if not results:
        return None

    dates = []

    for result in results:

        date_text = result.payload.get(
            "effective_date"
        )

        if date_text:

            dates.append(
                datetime.strptime(
                    date_text,
                    "%Y-%m-%d"
                )
            )

    if not dates:
        return None

    return max(dates).strftime(
        "%Y-%m-%d"
    )


def retrieve(query, product):

    query_vector = model.encode(query).tolist()

    latest_date = get_latest_effective_date(product)

    print("\n==============================")
    print("POLICY VERSION FILTER")
    print("==============================")

    print("Product:", product)
    print("Latest Effective Date:", latest_date)
    
    if latest_date is None:
        return []

    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,

        query_filter=Filter(
            must=[
                FieldCondition(
                    key="product",
                    match=MatchValue(
                        value=product
                    )
                ),

                FieldCondition(
                    key="effective_date",
                    match=MatchValue(
                        value=latest_date
                    )
                )
            ]
        ),

        limit=TOP_K
    ).points

    return [
        result
        for result in results
        if result.score >= SCORE_THRESHOLD
    ]