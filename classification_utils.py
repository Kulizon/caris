from utils import load_data_from_json


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
