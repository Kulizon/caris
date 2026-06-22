"""Interface of the detection module routing requests to yolo, ram, or gemma submodules."""

def _get_detector_module(trained_model: str):
    """
    Lazily resolves and returns the appropriate detector module based on model name.
    """
    model_id = trained_model.lower()
    if "gemma" in model_id:
        from . import gemma
        return gemma
    elif "ram" in model_id:
        from . import ram
        return ram
    else:
        from . import yolo
        return yolo

def detect_objects_in_image(trained_model: str, image_name: str) -> list[str]:
    """
    Detects objects/tags in the given image using the selected model.
    """
    detector = _get_detector_module(trained_model)
    return detector.detect(trained_model, image_name)

def detect_objects_with_speed(trained_model: str, image_name: str) -> tuple[list[str], dict]:
    """
    Detects objects/tags and returns their list along with the inference speed metrics.
    """
    detector = _get_detector_module(trained_model)
    return detector.detect_with_speed(trained_model, image_name)
