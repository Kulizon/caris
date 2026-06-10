import json
import os


def load_data_from_json(filename: str):
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            data = json.load(f)
        return data
    return None


def save_data_to_json(filename: str, data):
    with open(filename, "w") as f:
        json.dump(data, f, indent=4)


def save_image_codes_to_json(filename, image_name, iconclass_codes):
    if os.path.exists(filename):
        data = load_data_from_json(filename)
    else:
        data = {}
    if not isinstance(iconclass_codes, list):
        iconclass_codes = list(iconclass_codes)
    data[image_name] = iconclass_codes
    save_data_to_json(filename, data)


def read_image_codes_from_json(filename, image_name):
    if not os.path.exists(filename):
        return None
    data = load_data_from_json(filename)
    iconclass_codes = data.get(image_name)
    return iconclass_codes


def get_codes_from_associations_list(filename, iconclass_codes):
    associated_iconclass_codes = []
    data = load_data_from_json(filename)
    for entry in data['associations_list']:
        if set(entry['input_codes']).issubset(iconclass_codes):
            associated_iconclass_codes.append(entry['output_code'])
    return list(set(associated_iconclass_codes))

def detect_objects_yolo(trained_model, image_name):
    from ultralytics import YOLO
    detected_objects = []
    model = YOLO(trained_model)
    image_recognition = model(image_name)
    
    speed_metrics = {}
    if image_recognition and len(image_recognition) > 0:
        # ultralytics speed is in ms, convert to seconds to be consistent
        speed_metrics = {k: v / 1000.0 for k, v in image_recognition[0].speed.items()}

    for result in image_recognition:
        for box in result.boxes:
            detected_object = model.names[int(box.cls)]
            # TODO delete this
            if detected_object == "person":
                detected_object = "human being"
            # TODO delete this
            if detected_object not in detected_objects:
                detected_objects.append(detected_object)
    return detected_objects, speed_metrics

# Gemma
def detect_objects_gemma(trained_model, image_name):
    import ollama

    prompt = """
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
    
    response = ollama.chat(
        model=trained_model,
        messages=[{'role': 'user', 'content': prompt, 'images': [image_name]}],
        options={'temperature': 0.0} 
    )
    
    content = response['message']['content'].strip()
    return ([item.strip() for item in content.split(',')] if content else []), {}

# RAM (Recognize Anything Plus)
def detect_objects_ram(trained_model, image_name):
    import warnings
    import contextlib
    import os

    warnings.filterwarnings("ignore")

    if not (trained_model.startswith("http://") or trained_model.startswith("https://") or os.path.isfile(trained_model)):
        trained_model = "https://huggingface.co/xinyu1205/recognize-anything-plus-model/resolve/main/ram_plus_swin_large_14m.pth"

    with open(os.devnull, 'w') as f:
        with contextlib.redirect_stdout(f), contextlib.redirect_stderr(f):
            import torch
            from PIL import Image
            from ram.models import ram_plus
            from ram import inference_ram
            from ram import get_transform

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            model = ram_plus(
                pretrained=trained_model,
                image_size=384,
                vit="swin_l"
            )
            model.eval()
            model = model.to(device)

            transform = get_transform(image_size=384)
            image = Image.open(image_name).convert("RGB")
            image = transform(image).unsqueeze(0).to(device)

            tags_en, _ = inference_ram(image, model)

    if tags_en:
        return [tag.strip() for tag in tags_en.split('|')], {}
    return [], {}

def detect_objects_in_image(trained_model, image_name):
    if "yolo" in trained_model.lower():
        return detect_objects_yolo(trained_model, image_name)
    elif "gemma" in trained_model.lower():
        return detect_objects_gemma(trained_model, image_name)
    elif "ram" in trained_model.lower():
        return detect_objects_ram(trained_model, image_name)
    else:
        raise ValueError("Unknown model")

def get_meanings_from_codes(codes, filename=None):
    if filename is None:
        filename = os.path.join(os.path.dirname(__file__), "data", "iconclass_clean.txt")
    
    if not os.path.exists(filename):
        return []

    code_to_meaning = {}
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('|', 1)
            if len(parts) == 2:
                code_to_meaning[parts[0]] = parts[1]
    
    meanings = []
    for code in codes:
        if code in code_to_meaning:
            meanings.append(code_to_meaning[code])
    return meanings

