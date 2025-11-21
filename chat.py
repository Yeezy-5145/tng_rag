"""
Interactive chat interface for TNG Digital RAG System
"""

from rag_system import ask_tngd_bot


def main():
    """Interactive chat interface"""
    print("=" * 70)
    print("TNG Digital RAG Chat Interface")
    print("=" * 70)
    print("\nInitializing RAG system...")
    print("(This may take a moment on first run...)")

    # Initialize RAG system by making a dummy call
    # This will trigger the initialization and cache it
    try:
        _ = ask_tngd_bot("test")
        print("✓ RAG system ready!\n")
    except Exception as e:
        print(f"✗ Error initializing RAG system: {e}")
        print("Make sure you've run the scraper to collect FAQ data.")
        print("The system will still work, but may not have FAQ data loaded.")
        print()

    print("Type your questions about TNG Digital (or 'quit' to exit)")
    print("=" * 70)

    while True:
        try:
            # Get user input
            user_input = input("\n💬 You: ").strip()

            # Check for exit commands
            if user_input.lower() in ["quit", "exit", "q", "bye"]:
                print("\n👋 Goodbye!")
                break

            # Skip empty input
            if not user_input:
                continue

            # Query the RAG system using the function interface
            print("\n🤖 Assistant: ", end="", flush=True)
            result = ask_tngd_bot(user_input)

            # Check if blocked
            if result["blocked"]:
                print(f"⚠️  {result['final_answer']}")
                continue

            # Display answer
            print(result["final_answer"])

            # Show sources from retrieved chunks
            if result["retrieved_chunks"]:
                print("\n📚 Sources:")
                for i, chunk in enumerate(result["retrieved_chunks"], 1):
                    print(f"  {i}. {chunk['question']}")
                    print(f"     {chunk['url']}")

        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")


if __name__ == "__main__":
    main()
