"""Gemma (Ollama) keyword extraction."""

_PROMPT = """
    Act as an expert art historian extracting features for an Iconclass database search.
    Analyze the image and provide a dense list of keywords covering:
    1. Primary subjects (figures, animals, specific objects).
    2. Actions and postures.
    3. Setting and environment.
    4. Obvious symbolic elements.

    CRITICAL RULES:
    - Output ONLY a comma-separated string of short noun phrases.
    - NO introductory text. NO bullet points. NO full sentences.
    
    Example output format:
    sleeping dog, wooden floor, domestic interior, collar, resting animal
    """


def _run(trained_model, image_name):
    import ollama

    response = ollama.chat(
        model=trained_model,
        messages=[{'role': 'user', 'content': _PROMPT, 'images': [image_name]}],
        options={'temperature': 0.0}
    )

    content = response['message']['content'].strip()
    return [item.strip() for item in content.split(',')] if content else []


def detect(trained_model, image_name):
    return _run(trained_model, image_name)


def detect_with_speed(trained_model, image_name):
    return _run(trained_model, image_name), {}
