import os
from openai import OpenAI
from dotenv import load_dotenv
from retrieval import get_wikipedia_chunks, embed_chunks, retrieve_relevant_chunks
from prompts import build_prompt, get_available_styles, format_context_preview


def get_kong_client():
    """
    Create an OpenAI client configured to use Kong API Gateway.
    
    Returns:
        OpenAI client configured for Kong
    """
    load_dotenv()
    
    kong_token = os.getenv("KONG_API_TOKEN")
    kong_base_url = os.getenv("KONG_BASE_URL")
    
    if not kong_token:
        print("Error: KONG_API_TOKEN not found in .env file")
        print("Please create a .env file with your Kong API token:")
        print("KONG_API_TOKEN=your_kong_token_here")
        exit(1)
    
    if not kong_base_url:
        print("Error: KONG_BASE_URL not found in .env file")
        print("Please create a .env file with your Kong base URL:")
        print("KONG_BASE_URL=your_kong_base_url_here")
        exit(1)
    
    client = OpenAI(
        api_key=kong_token,
        base_url=kong_base_url,
        default_headers={"apikey": kong_token}
    )
    
    return client


def setup_kong():
    """
    Validate Kong API Gateway configuration.
    """
    load_dotenv()
    
    kong_token = os.getenv("KONG_API_TOKEN")
    kong_base_url = os.getenv("KONG_BASE_URL")
    
    if not kong_token or not kong_base_url:
        print("Error: Kong configuration incomplete")
        print("Please create a .env file with:")
        print("KONG_API_TOKEN=your_kong_token_here")
        print("KONG_BASE_URL=your_kong_base_url_here")
        exit(1)
    
    print("Kong API Gateway configuration loaded successfully")


def select_response_style():
    """
    Let the user choose a response style for the chatbot.
    
    Returns:
        Selected style string
    """
    available_styles = get_available_styles()
    
    print("\nChoose a response style:")
    for i, style in enumerate(available_styles, 1):
        style_descriptions = {
            "default": "Normal factual responses",
            "pirate": "Pirate-themed responses with 'arr' and 'matey'",
            "kid": "Simple explanations suitable for children", 
            "bullets": "Organized bullet-point format"
        }
        description = style_descriptions.get(style, "")
        print(f"{i}. {style} - {description}")
    
    while True:
        try:
            choice = input(f"\nEnter your choice (1-{len(available_styles)}) or press Enter for default: ").strip()
            
            if not choice:
                return "default"
            
            choice_num = int(choice)
            if 1 <= choice_num <= len(available_styles):
                selected_style = available_styles[choice_num - 1]
                print(f"Selected style: {selected_style}")
                return selected_style
            else:
                print(f"Please enter a number between 1 and {len(available_styles)}")
        except ValueError:
            print("Please enter a valid number or press Enter for default")


def process_question(question, style):
    """
    Process a user question through the complete RAG pipeline.
    
    Args:
        question: User's question
        style: Response style to use
    """
    try:
        print(f"\nSearching Wikipedia for: '{question}'...")
        
        # Step 1: Get Wikipedia chunks
        chunks = get_wikipedia_chunks(question, max_chunks=10)
        if not chunks:
            print("No relevant Wikipedia content found for your question.")
            return
        
        print(f"Found {len(chunks)} Wikipedia chunks")
        
        # Step 2: Embed the chunks
        print("Generating embeddings for Wikipedia content...")
        embeddings = embed_chunks(chunks)
        
        # Step 3: Retrieve relevant chunks for the question
        print("Finding most relevant content...")
        relevant_chunks = retrieve_relevant_chunks(question, chunks, embeddings, top_k=3)
        
        if not relevant_chunks:
            print("Could not find relevant chunks for your question.")
            return
        
        # Optional: Show preview of matched context
        context_preview = format_context_preview(relevant_chunks, max_preview_length=150)
        print(f"\nUsing context: {context_preview}")
        
        # Step 4: Build prompt with selected style
        messages = build_prompt(relevant_chunks, question, style)
        
        # Step 5: Get response from OpenAI via Kong
        print(f"\nGenerating {style} response...")
        client = get_kong_client()
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
            temperature=0.3
        )
        
        # Extract and display the response
        bot_response = response.choices[0].message.content
        print(f"\nBot ({style} style):")
        print("-" * 50)
        print(bot_response)
        print("-" * 50)
        
    except Exception as e:
        print(f"Error processing question: {e}")
        print("Please try again with a different question.")


def main():
    """
    Main CLI application for the Wikipedia RAG chatbot.
    """
    print("=" * 60)
    print("         Wikipedia-Powered RAG Chatbot")
    print("=" * 60)
    
    # Setup Kong API Gateway
    setup_kong()
    
    # Let user select response style
    style = select_response_style()
    
    print(f"\nChatbot ready! Using '{style}' response style.")
    print("You can ask questions about any topic and I'll search Wikipedia for answers.")
    print("Type 'exit' to quit or 'style' to change response style.\n")
    
    # Main conversation loop
    while True:
        try:
            # Get user input
            question = input("Ask a question (or 'exit'): ").strip()
            
            # Handle empty input
            if not question:
                print("Please enter a question or 'exit' to quit.")
                continue
            
            # Handle exit command
            if question.lower() in ['exit', 'quit', 'bye']:
                print("Thanks for using the Wikipedia RAG Chatbot!")
                break
            
            # Handle style change command
            if question.lower() == 'style':
                style = select_response_style()
                continue
            
            # Process the question
            process_question(question, style)
            
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"Unexpected error: {e}")
            print("Please try again.")


if __name__ == "__main__":
    main() 