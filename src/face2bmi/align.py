"""Face detection and alignment for inference-time uploads.

The provided dataset images are already pre-cropped faces (per the paper's
manual cleaning step), so training uses them as-is. New webcam/upload images
from the web demo are not pre-cropped, so MTCNN is used to detect and
align a centered face before feeding it to the backbone.
"""

from __future__ import annotations

from functools import lru_cache

from PIL import Image


@lru_cache(maxsize=1)
def _get_mtcnn(image_size: int = 160):
    """Lazy-load MTCNN once. Returns a PIL-cropped face tensor when called."""
    from facenet_pytorch import MTCNN

    return MTCNN(
        image_size=image_size,
        margin=14,
        post_process=False,
        keep_all=False,
        select_largest=True,
        device="cpu",
    )


def align_face(img: Image.Image, image_size: int = 160) -> Image.Image:
    """Return a square aligned face crop, or the resized original if no face found.

    Falls back to a center-crop+resize so the pipeline never hard-fails on a
    missing detection — important for the live demo.
    """
    mtcnn = _get_mtcnn(image_size=image_size)
    face = mtcnn(img.convert("RGB"))
    if face is None:
        w, h = img.size
        side = min(w, h)
        left = (w - side) // 2
        top = (h - side) // 2
        return img.crop((left, top, left + side, top + side)).resize(
            (image_size, image_size), Image.BILINEAR
        )
    # MTCNN returns a tensor in [0, 255] (because post_process=False).
    import numpy as np

    arr = face.permute(1, 2, 0).clamp(0, 255).to(dtype=face.dtype).cpu().numpy()
    return Image.fromarray(arr.astype(np.uint8))
