from pathlib import Path


def dataset_name_from_path(image_root):
    image_root = Path(image_root)
    if image_root.parent.name == "coco" and image_root.name in {"train2017", "val2017", "test2017"}:
        return f"coco_{image_root.name}"
    return image_root.name


def default_fixation_root(image_root, strategy):
    return Path("outputs") / "fixations" / dataset_name_from_path(image_root) / strategy


def default_saliency_root(image_root, model_name, output_kind):
    return Path("outputs") / "saliency" / dataset_name_from_path(image_root) / model_name / output_kind
