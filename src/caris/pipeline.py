"""High-level orchestration: tag finding, recommendation, and the Gemma pipeline."""

from caris.config import DEFAULT_GEMMA_MODEL, DEFAULT_TRAINED_MODEL
from caris.classification.hierarchy import (
    get_iconclass_hierarchy,
    search_equal,
    search_subset,
)
from caris.classification.subtree_search import (
    search_for_equal_tags_in_subtree,
    search_for_subset_of_tags_in_subtree,
)
from caris.detection import detect_objects_in_image, detect_objects_with_speed
from caris.embeddings import get_iconclass_codes_embeddings
from caris.recommendation import (
    recommend_image_idf_based,
    recommend_image_jaccard,
    recommend_image_simple,
)


def find_iconclass_tags(image_name: str, iconclass_branch_to_start_from: str = '', search_individually: str = "ALWAYS",
                        trained_model: str = DEFAULT_TRAINED_MODEL, detected_objects: list = None):
    if search_individually not in ["ALWAYS", "IF_NONE_FOUND", "NEVER"]:
        search_individually = "NEVER"

    result_codes = []

    if detected_objects is None:
        detected_objects = detect_objects_in_image(trained_model, image_name)

    if iconclass_branch_to_start_from == '':
        search_result = search_equal(detected_objects)
        if search_result:
            result_codes.append(search_result)
        else:
            search_result = search_subset(detected_objects)
            if search_result:
                result_codes.append(search_result)

        if search_individually == "ALWAYS" or (search_individually == "IF_NONE_FOUND" and not result_codes):
            for entity in detected_objects:
                search_result = search_equal([entity])
                if search_result:
                    result_codes.append(search_result)
                else:
                    search_result = search_subset([entity])
                    if search_result:
                        result_codes.append(search_result)
    else:
        iconclass_subbranch = get_iconclass_hierarchy()[iconclass_branch_to_start_from]
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


def classify_and_recommend(image_name: str, iconclass_branch_to_start_from: str, filename: str, search_individually: str = "ALWAYS", trained_model: str = DEFAULT_TRAINED_MODEL, idf_impact=1, mapping_mode: str = "auto"):
    code_list, _, _ = get_iconclass_codes(
        trained_model=trained_model,
        image_path=image_name,
        mapping_mode=mapping_mode,
        iconclass_branch_to_start_from=iconclass_branch_to_start_from,
        search_individually=search_individually
    )
    return recommend_images(filename, code_list, idf_impact)


def get_iconclass_codes(
    trained_model: str = DEFAULT_GEMMA_MODEL,
    image_path: str = "",
    mapping_mode: str = "auto",
    iconclass_branch_to_start_from: str = "",
    search_individually: str = "ALWAYS",
    detected_objects: list = None,
):
    """
    Orchestrates the object detection and Iconclass mapping pipeline.

    mapping_mode can be:
      - 'auto': determines mapping mode based on model name (yolo -> structural, gemma/ram -> semantic)
      - 'structural': hierarchy-based search (exact/subset)
      - 'semantic': vector database (embeddings) search
    """
    if detected_objects is None:
        detected_objects, speed_metrics = detect_objects_with_speed(trained_model, image_path)
    else:
        speed_metrics = {}

    mode = mapping_mode.lower()
    if mode == "auto":
        model_id = trained_model.lower()
        if "gemma" in model_id or "ram" in model_id:
            mode = "semantic"
        else:
            mode = "structural"

    if mode == "semantic":
        codes = get_iconclass_codes_embeddings(detected_objects)
    elif mode == "structural":
        iconclass_tags = find_iconclass_tags(
            image_name=image_path,
            iconclass_branch_to_start_from=iconclass_branch_to_start_from,
            search_individually=search_individually,
            trained_model=trained_model,
            detected_objects=detected_objects
        )
        # Flatten nested list of sets
        flat_codes = []
        for code_set in iconclass_tags:
            for code in code_set:
                flat_codes.append(str(code))
        codes = sorted(list(set(flat_codes)))
    else:
        raise ValueError(f"Unknown mapping mode: {mapping_mode}")

    return codes, detected_objects, speed_metrics
