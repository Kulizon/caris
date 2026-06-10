from utils import load_data_from_json, detect_objects_in_image, get_meanings_from_codes
from embeddings_utils import get_iconclass_codes_embeddings


def search_for_equal_tags_in_subtree(iconclass_subtree, detected_objects_list):
    result_codes = []
    for child in iconclass_subtree:
        if detected_objects_list == child.keywords():
            result_codes.append(child)
        else:
            result_codes.extend(search_for_equal_tags_in_subtree(child, detected_objects_list))
    return result_codes


def search_for_subset_of_tags_in_subtree(iconclass_subtree, detected_objects_list):
    result_codes = []
    for child in iconclass_subtree:
        if set(detected_objects_list).issubset(child.keywords()):
            result_codes.append(child)
        else:
            result_codes.extend(search_for_subset_of_tags_in_subtree(child, detected_objects_list))
    return result_codes


def search_for_codes_with_detected_object_in_name(iconclass_subtree, detected_objects_list):
    result_codes = []
    for child in iconclass_subtree:
        for detected_object in detected_objects_list:
            if detected_object.lower() in child().lower():
                result_codes.append(child)
            else:
                result_codes.extend(search_for_codes_with_detected_object_in_name(child, detected_objects_list))
    return result_codes


def reduce_iconclass_codes(iconclass_codes, detected_object):
    best_code = None
    min_extra_words = float('inf')
    for code in iconclass_codes:
        words = set(str(code).lower().split())
        if detected_object.lower() in words:
            extra_words = len(words) - 1
            if extra_words < min_extra_words:
                min_extra_words = extra_words
                best_code = code
    return best_code


def get_iconclass_codes_gemma(trained_model: str="gemma4:12b", image_path: str=""):
    """
    Orchestrates the Gemma/Vector search pipeline:
    1. Detect objects using Gemma (keyword extraction).
    2. Map keywords to Iconclass codes using embeddings.
    Returns: (list of codes, list of keywords, speed_metrics)
    """
    # Use the dispatcher in utils.py which handles both YOLO and Gemma
    classified_objects, speed_metrics = detect_objects_in_image(trained_model, image_path)
    
    # Use the robust implementation in embeddings_utils.py
    codes = get_iconclass_codes_embeddings(classified_objects)

    return codes, classified_objects, speed_metrics

