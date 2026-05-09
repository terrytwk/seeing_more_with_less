#!/usr/bin/env python3
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import numpy as np

original = "data/coco/val2017/000000000139.jpg"

variants = {
    "Original":       original,
    "var_center\n(paper baseline)": "data/pipeline_output/var_center_30d_3perc/Variable/000000000139_oid_999_fpx_320_fpy_213.jpg",
    "var_random\n(our method)":     "data/pipeline_output/var_random_30d_3perc/Variable/000000000139_oid_0_fpx_495_fpy_38.jpg",
    "var_gradient\n(our method)":   "data/pipeline_output/var_gradient_30d_3perc/Variable/000000000139_oid_0_fpx_561_fpy_336.jpg",
    "var_main_object\n(our method)":"data/pipeline_output/var_main_object_30d_3perc/Variable/000000000139_oid_0_fpx_80_fpy_212.jpg",
    "const/uniform\n(paper baseline)": "data/pipeline_output/const_30d_3perc/Constant/000000000139_oid_999_fpx_320_fpy_213.jpg",
}

fixation_points = {
    "var_center\n(paper baseline)":     (320, 213),
    "var_random\n(our method)":         (495, 38),
    "var_gradient\n(our method)":       (561, 336),
    "var_main_object\n(our method)":    (80, 212),
}

fig, axes = plt.subplots(1, 6, figsize=(20, 4))
fig.suptitle("Foveation strategies — 3% pixel budget\n(high resolution at fixation point, blurry periphery)",
             fontsize=13, y=1.02)

for ax, (title, path) in zip(axes, variants.items()):
    img = np.asarray(Image.open(path).convert("RGB"))
    ax.imshow(img)
    ax.set_title(title, fontsize=9)
    ax.axis("off")

    if title in fixation_points:
        fx, fy = fixation_points[title]
        circle = patches.Circle((fx, fy), radius=18, linewidth=2,
                                 edgecolor="red", facecolor="none")
        ax.add_patch(circle)
        ax.plot(fx, fy, "r+", markersize=10, markeredgewidth=2)

plt.tight_layout()
plt.savefig("foveation_example.png", dpi=150, bbox_inches="tight")
print("Saved foveation_example.png")
