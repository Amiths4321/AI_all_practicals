from rag_pipeline import run_rag
from product_detector import detect_product
from retriever import client

# ==========================================
# SESSION CONTEXT
# ==========================================

current_product = None
conversation_history = []


print("\nBanking RAG Assistant")
print("Type 'exit' to quit.")


while True:

    query = input("\nYou: ").strip()


    # --------------------------------------
    # Exit
    # --------------------------------------

    if query.lower() == "exit":
        break


    # --------------------------------------
    # Detect product
    # --------------------------------------

    detected_product = detect_product(query)

    if detected_product:
        current_product = detected_product

        print(
            f"Product selected: {current_product}"
        )


    # --------------------------------------
    # Check product
    # --------------------------------------

    if current_product is None:

        print(
            "\nPlease mention the loan product first."
        )

        continue


    # --------------------------------------
    # Add previous conversation to query
    # --------------------------------------

    history_text = ""

    for item in conversation_history:

        history_text += (
            f"\nUser: {item['question']}"
            f"\nAssistant: {item['answer']}\n"
        )


    


    # --------------------------------------
    # Run RAG
    # --------------------------------------
    result = run_rag(
        query,
        current_product,
        conversation_history
    )
    
    if result is None:
        print("Error: run_rag returned None.")
        # Handle the fallback safely
        answer = "An error occurred while processing your request."
        sources = []
    else:
        answer = result["answer"]
        sources = result.get("sources", [])
    # --------------------------------------
    # Store conversation
    # --------------------------------------

    conversation_history.append({
        "question": query,
        "answer": result["answer"]
    })


    # --------------------------------------
    # Output
    # --------------------------------------

    print("\nAssistant:")
    print(result["answer"])

    print("\n==============================")
    print("CONFIDENCE")
    print("==============================")

    print(
        f"Score: "
        f"{result['confidence']['score']:.4f}"
    )

    print(
        f"Level: "
        f"{result['confidence']['level']}"
    )

    print("\n==============================")
    print("REVIEW STATUS")
    print("==============================")

    if result["review_required"]:
        print("MANUAL REVIEW REQUIRED")
    else:
        print("AUTOMATED ANSWER")
        print("\nEvidence:")

    for source in result["evidence"]:

        print(
            f"- {source.get('document', 'N/A')} | "
            f"{source.get('section', 'N/A')} | "
            f"Chunk {source.get('chunk_id', 'N/A')} | "
            f"Score {source.get('score', 0.0):.4f}"
        )


client.close()