"""RAM (Recognize Anything Plus) tagging."""

_MODEL_CACHE = {}


def _get_model(trained_model):
    """Load and cache the RAM++ model, transform, and device."""
    import warnings
    import contextlib
    import os

    warnings.filterwarnings("ignore")

    if not (trained_model.startswith("http://") or trained_model.startswith("https://") or os.path.isfile(trained_model)):
        trained_model = "https://huggingface.co/xinyu1205/recognize-anything-plus-model/resolve/main/ram_plus_swin_large_14m.pth"

    if trained_model in _MODEL_CACHE:
        return _MODEL_CACHE[trained_model]

    with open(os.devnull, 'w') as f:
        with contextlib.redirect_stdout(f), contextlib.redirect_stderr(f):
            import torch
            from ram.models import ram_plus
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

    _MODEL_CACHE[trained_model] = (model, transform, device)
    return model, transform, device


def _run(trained_model, image_name):
    import warnings
    import contextlib
    import os

    warnings.filterwarnings("ignore")

    model, transform, device = _get_model(trained_model)

    with open(os.devnull, 'w') as f:
        with contextlib.redirect_stdout(f), contextlib.redirect_stderr(f):
            from PIL import Image
            from ram import inference_ram

            image = Image.open(image_name).convert("RGB")
            image = transform(image).unsqueeze(0).to(device)

            tags_en, _ = inference_ram(image, model)

    if tags_en:
        return [tag.strip() for tag in tags_en.split('|')]
    return []


def detect(trained_model, image_name):
    return _run(trained_model, image_name)


def detect_with_speed(trained_model, image_name):
    return _run(trained_model, image_name), {}
