"""YOLO object detection.

The detector returns only the list of detected objects. Its intrinsic
preprocess/inference/postprocess timing (produced by ultralytics as a side
effect of inference) is exposed separately via :func:`detect_with_speed`.
"""

_MODEL_CACHE = {}


def _get_model(model_path, use_cache):
    from ultralytics import YOLO
    if use_cache:
        if model_path not in _MODEL_CACHE:
            _MODEL_CACHE[model_path] = YOLO(model_path)
        return _MODEL_CACHE[model_path]
    return YOLO(model_path)


def _run(trained_model, image_name, remap_person=True, use_cache=False):
    detected_objects = []
    model = _get_model(trained_model, use_cache)
    image_recognition = model(image_name)

    speed_metrics = {}
    if image_recognition and len(image_recognition) > 0:
        # ultralytics speed is in ms, convert to seconds to be consistent
        speed_metrics = {k: v / 1000.0 for k, v in image_recognition[0].speed.items()}

    for result in image_recognition:
        for box in result.boxes:
            detected_object = model.names[int(box.cls)]
            if remap_person and detected_object == "person":
                detected_object = "human being"
            if detected_object not in detected_objects:
                detected_objects.append(detected_object)
    return detected_objects, speed_metrics


def detect(trained_model, image_name, remap_person=True, use_cache=False):
    return _run(trained_model, image_name, remap_person, use_cache)[0]


def detect_with_speed(trained_model, image_name, remap_person=True, use_cache=False):
    return _run(trained_model, image_name, remap_person, use_cache)
