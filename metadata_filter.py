from qdrant_client.models import (
    Filter,
    FieldCondition,
    MatchValue
)


def build_metadata_filter(
    product=None,
    policy_version=None,
    effective_date=None,
    document_type=None
):

    conditions = []

    if product:
        conditions.append(
            FieldCondition(
                key="product",
                match=MatchValue(value=product)
            )
        )

    if policy_version:
        conditions.append(
            FieldCondition(
                key="policy_version",
                match=MatchValue(value=policy_version)
            )
        )

    if effective_date:
        conditions.append(
            FieldCondition(
                key="effective_date",
                match=MatchValue(value=effective_date)
            )
        )

    if document_type:
        conditions.append(
            FieldCondition(
                key="document_type",
                match=MatchValue(value=document_type)
            )
        )

    if not conditions:
        return None

    return Filter(must=conditions)