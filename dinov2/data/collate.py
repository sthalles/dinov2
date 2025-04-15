# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import torch
import random


def collate_data_and_cast(samples_list, mask_ratio_tuple, mask_probability, dtype, n_tokens=None, mask_generator=None):
    # dtype = torch.half  # TODO: Remove

    n_global_crops = len(samples_list[0][0]["global_crops"])
    n_local_crops = len(samples_list[0][0]["local_crops"])

    collated_global_crops = torch.stack(
        [s[0]["global_crops"][i] for i in range(n_global_crops) for s in samples_list])

    collated_local_crops = torch.stack(
        [s[0]["local_crops"][i] for i in range(n_local_crops) for s in samples_list])

    B = len(collated_global_crops)
    N = n_tokens
    n_samples_masked = int(B * mask_probability)
    probs = torch.linspace(*mask_ratio_tuple, n_samples_masked + 1)
    upperbound = 0
    
    masks_list = []
    for i in range(0, n_samples_masked):
        prob_min = probs[i]
        prob_max = probs[i + 1]
        n_masked = int(N * random.uniform(prob_min, prob_max))
        m = torch.BoolTensor(mask_generator(n_masked))
        masks_list.append(m)
        upperbound += int(N * prob_max)
    for i in range(n_samples_masked, B):
        m = torch.zeros_like(m, dtype=torch.bool)
        masks_list.append(m)

    random.shuffle(masks_list)

    collated_masks = torch.stack(masks_list).flatten(1)
    mask_indices_list = collated_masks.flatten().nonzero().flatten()

    masks_weight = (1 / collated_masks.sum(-1).clamp(min=1.0)
                    ).unsqueeze(-1).expand_as(collated_masks)[collated_masks]

    return {
        "collated_global_crops": collated_global_crops.to(dtype),
        "collated_local_crops": collated_local_crops.to(dtype),
        "collated_masks": collated_masks,
        "mask_indices_list": mask_indices_list,
        "masks_weight": masks_weight,
        "upperbound": upperbound,
        "n_masked_patches": torch.full((1,), fill_value=mask_indices_list.shape[0], dtype=torch.long),
    }
    
# def collate_data_and_cast(samples_list, mask_ratio_tuple, mask_probability, dtype, n_tokens=None, mask_generator=None):
#     # dtype = torch.half  # TODO: Remove

#     n_global_crops = len(samples_list[0][0]["global_crops"])
#     n_local_crops = len(samples_list[0][0]["local_crops"])

#     collated_global_crops = torch.stack(
#         [s[0]["global_crops"][i] for i in range(n_global_crops) for s in samples_list])

#     collated_local_crops = torch.stack(
#         [s[0]["local_crops"][i] for i in range(n_local_crops) for s in samples_list])

#     B = len(collated_global_crops)
#     N = n_tokens
#     # n_samples_masked = int(B * mask_probability)
#     # probs = torch.linspace(*mask_ratio_tuple, n_samples_masked + 1)
#     upperbound = 0
#     masks_list = []
#     n_masked = int(N * mask_ratio_tuple[0]) # n_masked: 127
#     for i in range(B):
#         # prob_min = probs[i]
#         # prob_max = probs[i + 1]
#         # print(prob_min, prob_max)
#         m = torch.BoolTensor(mask_generator(n_masked))
#         masks_list.append(m)
#         upperbound += int(N * mask_ratio_tuple[0])
#     # for i in range(n_samples_masked, B):
#     #     m = torch.zeros_like(m, dtype=torch.bool)
#     #     masks_list.append(m)

#     # random.shuffle(masks_list)

#     collated_masks = torch.stack(masks_list).flatten(1)
#     # print("collated_masks:", collated_masks.shape)

#     mask_indices_abs = collated_masks.flatten().nonzero().reshape(B, -1)
#     # print("mask_indices_abs:", mask_indices_abs.shape)

#     prediction_subsampling = 0.5
#     n_predict = int(n_masked * prediction_subsampling)
#     # print("n_predict:", n_predict)

#     randperm = torch.argsort(torch.rand(B, n_masked))[:, :n_predict]
#     # print("randperm:", randperm.shape)
#     predict_indices_abs = torch.gather(
#         mask_indices_abs, index=randperm, dim=1)
#     # print("predict_indices_abs:", predict_indices_abs.shape)
#     # print("n_masked:", n_masked)

#     # mask_indices_list = collated_masks.flatten().nonzero().flatten()
#     masks_weight = (1 / collated_masks.sum(-1).clamp(min=1.0)
#                     ).unsqueeze(-1).expand_as(collated_masks)
#     assert masks_weight.shape == collated_masks.shape
#     masks_weight = torch.gather(
#         masks_weight, index=randperm, dim=1).flatten()
    
#     predict_indices_abs = predict_indices_abs.flatten()
#     n_masked_patches = torch.full(
#         (1,), fill_value=predict_indices_abs.shape[0], dtype=torch.long)
#     # print("n_masked_patches:", n_masked_patches)
#     # collated_masks: torch.Size([512, 196])
#     # mask_indices_list: torch.Size([32512])
#     # masks_weight: torch.Size([32512])
#     # upperbound: 32512
#     # n_masked_patches: tensor([32512])

#     return {
#         "collated_global_crops": collated_global_crops.to(dtype),
#         "collated_local_crops": collated_local_crops.to(dtype),
#         "collated_masks": collated_masks,
#         "mask_indices_list": predict_indices_abs,
#         "masks_weight": masks_weight,
#         "upperbound": upperbound,
#         "n_masked_patches": n_masked_patches,
#     }
