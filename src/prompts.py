from typing import List, Dict


def build_prompt(context: List[str], question: str, style: str = "default") -> List[Dict[str, str]]:
    """
    Build a prompt for the RAG chatbot with different response styles.
    
    Args:
        context: List of relevant Wikipedia text chunks
        question: User's question
        style: Response style ("default", "pirate", "kid", "bullets")
        
    Returns:
        List of message dictionaries in OpenAI ChatCompletion format
    """
    # Format the context by joining chunks with double newlines
    formatted_context = "\n\n".join(context) if context else "No relevant context found."
    
    # Define system messages for different styles
    system_messages = {
        "default": {
            "role": "system",
            "content": "You are a helpful assistant that answers questions based on Wikipedia content. "
                      "Provide accurate, informative responses using the given context. "
                      "If the context doesn't contain enough information to answer the question, "
                      "say so clearly and provide what information you can."
        },
        "pirate": {
            "role": "system",
            "content": "You are a pirate who answers questions based on Wikipedia content. "
                      "Respond in pirate speak with 'arr', 'matey', 'ye', and other pirate expressions. "
                      "Still provide accurate information from the context, but make it fun and pirate-like. "
                      "If ye don't have enough information in the context, say so like a true pirate!"
        },
        "kid": {
            "role": "system",
            "content": "You are a friendly teacher who explains things to children. "
                      "Use simple words, short sentences, and fun examples. "
                      "Make complex topics easy to understand for kids. "
                      "Use the Wikipedia context to give accurate but kid-friendly explanations. "
                      "If the context doesn't have enough info, explain that in a nice way kids can understand."
        },
        "bullets": {
            "role": "system",
            "content": "You are an assistant that provides clear, organized answers in bullet point format. "
                      "Structure your responses using bullet points and sub-bullets when helpful. "
                      "Base your answers on the Wikipedia context provided. "
                      "Use bullet points to break down complex information into digestible pieces. "
                      "If context is insufficient, clearly state this in bullet format."
        }
    }
    
    # Build the messages list
    messages = []
    
    # Add system message
    system_msg = system_messages.get(style, system_messages["default"])
    messages.append(system_msg)
    
    # Add few-shot examples for certain styles
    if style == "pirate":
        messages.extend([
            {
                "role": "user",
                "content": "Context: The ocean covers 71% of Earth's surface.\n\nWhat percentage of Earth is covered by ocean?"
            },
            {
                "role": "assistant",
                "content": "Arr matey! According to me trusty knowledge, the mighty ocean covers 71% of our beautiful Earth's surface! "
                          "That be more than two-thirds of our planet, ye savvy sailor! The seas be vast and full of treasures!"
            }
        ])
    elif style == "kid":
        messages.extend([
            {
                "role": "user",
                "content": "Context: Elephants are the largest land animals. They can weigh up to 6,000 kilograms.\n\nHow big are elephants?"
            },
            {
                "role": "assistant",
                "content": "Wow! Elephants are REALLY big! They're the biggest animals that walk on land. "
                          "An elephant can weigh as much as 6,000 kilograms - that's like 4 cars put together! "
                          "Isn't that amazing? They're like gentle giants!"
            }
        ])
    elif style == "bullets":
        messages.extend([
            {
                "role": "user",
                "content": "Context: Python is a programming language created by Guido van Rossum in 1991. It emphasizes code readability.\n\nTell me about Python programming language."
            },
            {
                "role": "assistant",
                "content": "• **Creator**: Guido van Rossum\n"
                          "• **Year Created**: 1991\n"
                          "• **Key Feature**: Emphasizes code readability\n"
                          "• **Type**: Programming language\n"
                          "• **Philosophy**: Makes code easy to read and understand"
            }
        ])
    
    # Add the main user message with context and question
    user_content = f"Context from Wikipedia:\n{formatted_context}\n\nQuestion: {question}"
    
    messages.append({
        "role": "user",
        "content": user_content
    })
    
    return messages


def get_available_styles() -> List[str]:
    """
    Get a list of available response styles.
    
    Returns:
        List of available style names
    """
    return ["default", "pirate", "kid", "bullets"]


def format_context_preview(context: List[str], max_preview_length: int = 200) -> str:
    """
    Create a preview of the context for debugging or display purposes.
    
    Args:
        context: List of context chunks
        max_preview_length: Maximum length of preview text
        
    Returns:
        Formatted preview string
    """
    if not context:
        return "No context available"
    
    full_context = " ".join(context)
    
    if len(full_context) <= max_preview_length:
        return full_context
    
    preview = full_context[:max_preview_length].rsplit(" ", 1)[0]
    return f"{preview}..." 