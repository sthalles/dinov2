# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import random
import math
import numpy as np


class MaskingGenerator:
    def __init__(
        self,
        input_size,
        num_masking_patches=None,
        min_num_patches=4,
        max_num_patches=None,
        min_aspect=0.3,
        max_aspect=None,
    ):
        if not isinstance(input_size, tuple):
            input_size = (input_size,) * 2
        self.height, self.width = input_size

        self.num_patches = self.height * self.width
        self.num_masking_patches = num_masking_patches

        self.min_num_patches = min_num_patches
        self.max_num_patches = num_masking_patches if max_num_patches is None else max_num_patches

        max_aspect = max_aspect or 1 / min_aspect
        self.log_aspect_ratio = (math.log(min_aspect), math.log(max_aspect))

    def __repr__(self):
        repr_str = "Generator(%d, %d -> [%d ~ %d], max = %d, %.3f ~ %.3f)" % (
            self.height,
            self.width,
            self.min_num_patches,
            self.max_num_patches,
            self.num_masking_patches,
            self.log_aspect_ratio[0],
            self.log_aspect_ratio[1],
        )
        return repr_str

    def get_shape(self):
        return self.height, self.width

    def _mask(self, mask, max_mask_patches):
        delta = 0
        for _ in range(10):
            target_area = random.uniform(
                self.min_num_patches, max_mask_patches)
            aspect_ratio = math.exp(random.uniform(*self.log_aspect_ratio))
            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))
            if w < self.width and h < self.height:
                top = random.randint(0, self.height - h)
                left = random.randint(0, self.width - w)

                num_masked = mask[top: top + h, left: left + w].sum()
                # Overlap
                if 0 < h * w - num_masked <= max_mask_patches:
                    for i in range(top, top + h):
                        for j in range(left, left + w):
                            if mask[i, j] == 0:
                                mask[i, j] = 1
                                delta += 1

                if delta > 0:
                    break
        return delta

    def __call__(self, num_masking_patches=0):
        mask = np.zeros(shape=self.get_shape(), dtype=bool)
        mask_count = 0
        while mask_count < num_masking_patches:
            max_mask_patches = num_masking_patches - mask_count
            max_mask_patches = min(max_mask_patches, self.max_num_patches)

            delta = self._mask(mask, max_mask_patches)
            if delta == 0:
                break
            else:
                mask_count += delta

        return mask


def random_boolean_array(n, k):
    arr = np.array([True] * k + [False] * (n - k))
    np.random.shuffle(arr)
    return arr


class UniformMasking:
    def __init__(
        self,
        input_size,
    ):
        self.height, self.width = input_size

    def __call__(self, num_masking_patches=0):
        mask = random_boolean_array(
            self.height * self.width, num_masking_patches)
        mask = mask.reshape((self.height, self.width))
        return mask


class BlockMasking:
    def __init__(
        self,
        input_size: tuple[int, int],
        roll: bool = True,
        min_aspect: float = 0.5,
        max_aspect: float | None = None,
    ):
        self.height, self.width = input_size
        self.roll = roll
        max_aspect = max_aspect or 1 / min_aspect
        self.log_aspect_ratio = (math.log(min_aspect), math.log(max_aspect))

    # -> jt.Bool[np.ndarray, "h w"]:
    def __call__(self, num_masking_patches: int = 0):
        if num_masking_patches == 0:
            return np.zeros((self.height, self.width), dtype=bool)
        # Sample aspect ratio, not too large or too small for image
        min_lar = max(self.log_aspect_ratio[0], np.log(
            num_masking_patches / (self.width**2)))
        max_lar = min(
            self.log_aspect_ratio[1],
            np.log(self.height**2 / (num_masking_patches + 1e-5)),
        )
        aspect_ratio = math.exp(random.uniform(min_lar, max_lar))
        # Use ceil so mask is >= num_masking_patches
        h = int(np.ceil(math.sqrt(num_masking_patches * aspect_ratio)))
        w = int(np.ceil(math.sqrt(num_masking_patches / aspect_ratio)))

        top = random.randint(0, self.height - h)
        left = random.randint(0, self.width - w)
        mask = np.zeros((self.height, self.width), dtype=bool)
        mask[top: top + h, left: left + w] = True
        # truncate ids to get exactly num_masking_patches
        ids = np.where(mask.flatten())[0][:num_masking_patches]
        mask = np.zeros((self.height, self.width), dtype=bool).flatten()
        mask[ids] = True
        mask = mask.reshape((self.height, self.width))
        if self.roll:
            shift_x = random.randint(0, mask.shape[0] - 1)
            shift_y = random.randint(0, mask.shape[1] - 1)
            mask = np.roll(mask, (shift_x, shift_y), (0, 1))
        return mask


class InverseBlockMasking(BlockMasking):
    def __call__(self, num_masking_patches=0):
        mask = super().__call__(self.height * self.width - num_masking_patches)
        return ~mask


mask_generators_dict = {
    "Uniform": UniformMasking,
    "Block": BlockMasking,
    "InverseBlock": InverseBlockMasking,
}
