from classification_utils import search_for_equal_tags_in_subtree, search_for_subset_of_tags_in_subtree
from recommendation_utils import recommend_image_simple, recommend_image_idf_based, recommend_image_jaccard
from utils import detect_objects_in_image
import iconclass

DEFAULT_TRAINED_MODEL = 'models/yolo26n.pt'

# Global cache for iconclass hierarchy
_ICONCLASS_CACHE = None

def get_iconclass_hierarchy():
    global _ICONCLASS_CACHE
    if _ICONCLASS_CACHE is None:
        _ICONCLASS_CACHE = iconclass.init()
    return _ICONCLASS_CACHE


def find_iconclass_tags(image_name: str, iconclass_branch_to_start_from: str = '', search_individually: str = "ALWAYS",
                        trained_model: str = DEFAULT_TRAINED_MODEL, detected_objects: list = None):
    if search_individually not in ["ALWAYS", "IF_NONE_FOUND", "NEVER"]:
        search_individually = "NEVER"

    result_codes = []
    # Use cached hierarchy
    iconclass_subbranch = get_iconclass_hierarchy()[iconclass_branch_to_start_from]

    if detected_objects is None:
        detected_objects = detect_objects_in_image(trained_model, image_name)

    search_result = search_for_equal_tags_in_subtree(iconclass_subbranch, detected_objects)
    if search_result:
        result_codes.append(search_result)
    else:
        search_result = search_for_subset_of_tags_in_subtree(iconclass_subbranch, detected_objects)
        if search_result:
            result_codes.append(search_result)

    if search_individually == "ALWAYS" or (search_individually == "IF_NONE_FOUND" and not result_codes):
        for detected_entity in detected_objects:
            search_result = search_for_equal_tags_in_subtree(iconclass_subbranch, [detected_entity])
            if search_result:
                result_codes.append(search_result)
            else:
                search_result = search_for_subset_of_tags_in_subtree(iconclass_subbranch, [detected_entity])
                if search_result:
                    result_codes.append(search_result)

    result_codes = [set(codes) for codes in result_codes]

    return result_codes


def recommend_images(filename: str, iconclass_codes_list: list, idf_impact=1):
    recommended_image_simple = recommend_image_simple(filename, iconclass_codes_list)
    recommended_image_idf = recommend_image_idf_based(filename, iconclass_codes_list, idf_impact)
    recommended_image_jaccard = recommend_image_jaccard(filename, iconclass_codes_list)
    return recommended_image_simple, recommended_image_idf, recommended_image_jaccard


def classify_and_recommend(image_name: str, iconclass_branch_to_start_from: str, filename: str, search_individually: str = "ALWAYS", trained_model: str = DEFAULT_TRAINED_MODEL, idf_impact=1):
    iconclass_codes_nested = find_iconclass_tags(
        image_name,
        iconclass_branch_to_start_from,
        search_individually,
        trained_model
    )
    code_list = []
    for iconclass_sets in iconclass_codes_nested:
        for iconclass_code in iconclass_sets:
            code_list.append(str(iconclass_code))
    return recommend_images(filename, code_list, idf_impact)
