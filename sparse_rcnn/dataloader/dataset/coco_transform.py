import cv2
import numpy as np


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, bboxes):
        for t in self.transforms:
            image, bboxes = t(image, bboxes)
        return image, bboxes

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string


class RandomFlip(object):
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, image, bboxes):
        h, w, _ = image.shape
        if random.random() < self.prob:
            for _box in bboxes:
                _box[0], _box[2] = w - _box[2], w - _box[0]
            image = image[:, ::-1]  # - np.zeros_like(image)
        return image, bboxes


class ResizeTransform(object):
    """
    Resize the image to a target size.
    """

    def __init__(self, h, w, new_h, new_w, interp=None):
        """
        Args:
            h, w (int): original image size
            new_h, new_w (int): new image size
            interp: PIL interpolation methods, defaults to bilinear.
        """
        self.new_h = new_h
        self.new_w = new_w

    def __call__(self, img, bboxes):
        h, w, _ = img.shape
        img = cv2.resize(img, (self.new_w, self.new_h))
        for _box in bboxes:
            _box[0] *= self.new_w / w
            _box[1] *= self.new_h / h
            _box[2] *= self.new_w / w
            _box[3] *= self.new_h / h
        return img, bboxes


class ResizeShortestEdge(object):
    """
    Scale the shorter edge to the given size, with a limit of `max_size` on the longer edge.
    If `max_size` is reached, then downscale so that the longer edge does not exceed max_size.
    """

    def __init__(
            self, short_edge_length, max_size=4000, sample_style="range", interp=None
    ):
        """
        Args:
            short_edge_length (list[int]): If ``sample_style=="range"``,
                a [min, max] interval from which to sample the shortest edge length.
                If ``sample_style=="choice"``, a list of shortest edge lengths to sample from.
            max_size (int): maximum allowed longest edge length.
            sample_style (str): either "range" or "choice".
        """
        assert sample_style in ["range", "choice"], sample_style

        self.is_range = sample_style == "range"
        self.short_edge_length = short_edge_length
        self.max_size = max_size
        self.sample_style = sample_style
        self.interp = interp
        if self.is_range:
            assert len(short_edge_length) == 2, (
                "short_edge_length must be two values using 'range' sample style."
                f" Got {short_edge_length}!"
            )

    def __call__(self, image, bboxes):
        h, w = image.shape[:2]
        if self.is_range:
            size = np.random.randint(self.short_edge_length[0], self.short_edge_length[1] + 1)
        else:
            size = np.random.choice(self.short_edge_length)

        scale = size * 1.0 / min(h, w)
        if h < w:
            newh, neww = size, scale * w
        else:
            newh, neww = scale * h, size
        if max(newh, neww) > self.max_size:
            scale = self.max_size * 1.0 / max(newh, neww)
            newh = newh * scale
            neww = neww * scale
        neww = int(neww + 0.5)
        newh = int(newh + 0.5)

        return ResizeTransform(h, w, newh, neww, self.interp)(image, bboxes)


from albumentations.augmentations.geometric import functional as F
from albumentations import DualTransform
import random
from typing import Sequence, Dict, Tuple, Union


class SmallestMaxSize_v2(DualTransform):
    def __init__(
            self,
            max_size: Union[int, Sequence[int]] = 1024,
            max_limit: int = 4096,
            interpolation: int = cv2.INTER_LINEAR,
            always_apply: bool = False,
            p: float = 1,
    ):
        super(SmallestMaxSize_v2, self).__init__(always_apply, p)
        self.interpolation = interpolation
        self.max_size = max_size
        self.max_limit = max_limit

    def apply(
            self, img: np.ndarray, max_size: int = 1024, max_limit: int = 4096, interpolation: int = cv2.INTER_LINEAR,
            **params
    ) -> np.ndarray:
        img_h, img_w = img.shape[:2]
        img_min = min(img_h, img_w)
        img_max = max(img_h, img_w)
        scale = max_size / img_min
        scale_h, scale_w = scale * img_h, scale * img_w
        if max(scale_h, scale_w) > max_limit:
            scale = max_limit / img_max
        max_size = scale * min(img_h, img_w)
        return F.smallest_max_size(img, max_size=max_size, interpolation=interpolation)

    def apply_to_bbox(self, bbox: Sequence[float], **params) -> Sequence[float]:
        return bbox

    def get_params(self) -> Dict[str, int]:
        return {"max_size": self.max_size if isinstance(self.max_size, int) else random.choice(self.max_size),
                "max_limit": self.max_limit}

    def get_transform_init_args_names(self) -> Tuple[str, ...]:
        return ("max_size", "interpolation", "max_limit")
