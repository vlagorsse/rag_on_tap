import argparse
import logging
import sys

from services.chat_service import ChatService
from services.config_service import ConfigService, LLMProvider


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        stream=sys.stdout,
    )
    # Quiet down external libs
    logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("langchain_google_genai").setLevel(logging.WARNING)
    logging.getLogger("google.ai.generativelanguage").setLevel(logging.WARNING)


def main():
    parser = argparse.ArgumentParser(description="Beer Expert Chat CLI")
    parser.add_argument(
        "--model", default=None, help="LLM model name (overrides config)"
    )
    parser.add_argument(
        "--collection", default="beer_recipes", help="Vector collection name"
    )
    args = parser.parse_args()

    setup_logging()
    try:
        config = ConfigService()
    except Exception as e:
        print(f"Configuration Error: {e}")
        sys.exit(1)

    print(
        f"\n--- 🍺 Beer Expert RAG-on-Tap Chat ({config.llm_provider.value.upper()}) ---"
    )
    print("Initializing services (loading models)...")

    try:
        chat_service = ChatService(
            config=config, model_name=args.model, collection_name=args.collection
        )
    except Exception as e:
        print(f"Failed to initialize ChatService: {e}")
        sys.exit(1)

    print("Ready! Ask me anything about beer recipes or brewing.")
    print("Type 'exit' or 'quit' to leave.\n")

    while True:
        try:
            user_input = input("You> ")
            if user_input.lower() in ("exit", "quit"):
                break
            if not user_input.strip():
                continue

            print("\nRAG-on-Tap is thinking...")
            response = chat_service.chat(user_input)

            print(f"RAG-on-Tap> {response}\n")

        except (KeyboardInterrupt, EOFError):
            print("\nExiting...")
            break
        except Exception as e:
            print(f"\nError: {e}")


if __name__ == "__main__":
    main()
