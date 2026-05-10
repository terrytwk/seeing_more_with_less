from pathlib import Path


def resolve_pretrained_reference(model_ref, model_name, recommended_id):
    """
    Return a usable Transformers from_pretrained reference.

    Missing local-looking paths otherwise get passed to Hugging Face Hub
    validation, which produces a misleading repo-id error for paths such as
    data/models/detr-resnet-101.
    """
    model_ref = str(model_ref)
    expanded_path = Path(model_ref).expanduser()
    if expanded_path.exists():
        return str(expanded_path)

    is_path_like = (
        model_ref.startswith((".", "/", "~"))
        or "\\" in model_ref
        or model_ref.count("/") > 1
    )
    if is_path_like:
        raise FileNotFoundError(
            f"{model_name} model path does not exist: {model_ref}\n"
            f"Use a valid local model directory, or pass the Hugging Face model id "
            f"'{recommended_id}'."
        )

    return model_ref
