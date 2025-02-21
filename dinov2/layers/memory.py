import torch.nn as nn
import torch


class MemoryBank(nn.Module):

    def __init__(
        self,
        K,
        partition_size,
        num_experts_per_concept=2,
        out_dim=256,
        smoothing=0.1,
        num_tasks=1,
        selection_strategy='low_energy'
    ):
        # create the queue
        super().__init__()
        self.K = K
        self.partition_size = partition_size
        self.num_experts_per_concept = num_experts_per_concept
        self.out_dim = out_dim
        self.smoothing = smoothing
        self.num_tasks = num_tasks

        self.register_buffer("queue", torch.randn(out_dim, self.K))
        self.queue = nn.functional.normalize(self.queue, dim=0)
        self.smoothing = smoothing
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
        self.register_buffer("partition_labels",
                             torch.arange(0, partition_size))
        self.partition_labels = self.partition_labels.repeat_interleave(
            num_experts_per_concept, dim=0
        )
        self.selection_strategy = selection_strategy

    def __str__(self):
        return f"MemoryBank(K={self.K}, partition_size={self.partition_size}, num_experts_per_concept={self.num_experts_per_concept}, out_dim={self.out_dim}, smoothing={self.smoothing})"

    @staticmethod
    @torch.no_grad()
    def concat_all_gather(tensor):
        """
        Performs all_gather operation on the provided tensors.
        *** Warning ***: torch.distributed.all_gather has no gradient.
        """
        tensors_gather = [
            torch.ones_like(tensor) for _ in range(torch.distributed.get_world_size())
        ]
        torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

        output = torch.cat(tensors_gather, dim=0)
        return output

    @torch.no_grad()
    def update_queue(self, keys):
        # gather keys before updating queue
        keys = self.concat_all_gather(keys)

        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr)

        # for simplicity
        assert self.K % batch_size == 0, f"{self.K} % {batch_size}"

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr: ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.K  # move pointer
        self.queue_ptr[0] = ptr

    def get_features(self):
        return self.queue.clone().detach()

    def get_partition_and_memory_indices(self):
        rand_cluster_indices = torch.randperm(self.K).cuda()
        partition, memory_block_indices = torch.split(
            rand_cluster_indices,
            split_size_or_sections=(
                self.partition_size, self.K - self.partition_size),
        )
        return partition, memory_block_indices

    def forward(self, student_embeds, teacher_embeds, student_temp, teacher_temp):

        memory_embeds = self.get_features()
        student_partition_probs_list, teacher_partition_probs_list = [], []

        # class-dependent random partitions
        for i in range(self.num_tasks):
            partition_indices, memory_block_indices = self.get_partition_and_memory_indices()
            partition_embeds = torch.take_along_dim(
                memory_embeds, indices=partition_indices.unsqueeze(0), dim=1
            )

            memory_block_embeds = torch.take_along_dim(
                memory_embeds, indices=memory_block_indices.unsqueeze(0), dim=1
            )

            # top-k nearest neighbour
            similarities = partition_embeds.t() @ memory_block_embeds

            neigh_scores, neigh_indices = torch.topk(
                similarities, k=self.num_experts_per_concept - 1, dim=-1
            )

            start = memory_block_embeds.shape[1]
            end = start + neigh_indices.shape[0]
            indices_ = torch.cat(
                (
                    neigh_indices,
                    torch.arange(
                        start=start, end=end, device=student_embeds.device
                    ).unsqueeze(1),
                ),
                dim=1,
            )

            scores_ = torch.cat(
                (
                    neigh_scores,
                    torch.full(
                        (neigh_scores.shape[0], 1),
                        1.0 - self.smoothing,
                        device=student_embeds.device,
                    ),
                ),
                dim=1,
            )
            partition_size = self.partition_size
            if self.selection_strategy == 'low_energy' and self.num_experts_per_concept > 1:
                partition_size = partition_size // 2
                # select a subset of low energy anchor points
                sample_means = torch.mean(neigh_scores, dim=-1).abs()
                sample_stds = torch.mean(neigh_scores, dim=-1)
                coef_variation = sample_stds / (sample_means + 1e-8)
                coef_variation = (
                    1 + 1/(4*similarities.shape[-1])) * coef_variation
                # (coef_variation.max() - coef_variation) + coef_variation.min()
                sample_weights = 1 / coef_variation

                # sample_probs = torch.softmax(coef_variation, dim=-1)
                low_energy_anchor_indices = torch.multinomial(
                    sample_weights.flatten(), partition_size, replacement=False)
                indices_ = indices_[low_energy_anchor_indices]
                scores_ = scores_[low_energy_anchor_indices]

            embeds_ = torch.cat((memory_block_embeds, partition_embeds), dim=1)

            partition_neigh_embeddings = torch.take_along_dim(
                embeds_, indices=indices_.flatten().unsqueeze(0), dim=1
            )

            # smooth_value = self.smoothing/(self.partition_size-1)
            # partition_label_matrix = torch.full(
            #     (len(self.partition_labels), self.partition_size), smooth_value, device=student_embeds.device)
            # partition_label_matrix.scatter_(1, self.partition_labels.unsqueeze(
            #     1), torch.full_like(partition_label_matrix, 1.0-self.smoothing))

            smoothing = 1 - scores_.flatten().unsqueeze(1)
            smooth_value = smoothing / (partition_size - 1)
            partition_label_matrix = torch.ones(
                (len(self.partition_labels[: smooth_value.size(0)]), partition_size), device=student_embeds.device) * smooth_value
            partition_label_matrix.scatter_(1, self.partition_labels[: smooth_value.size(0)].unsqueeze(
                1), torch.ones_like(partition_label_matrix) * scores_.flatten().unsqueeze(1))
            # renormalize partition labels because of numerical imprecisions
            partition_label_matrix /= partition_label_matrix.sum(-1,
                                                                 keepdim=True)

            # smooth_value = self.smoothing/(self.partition_size-1)
            # anchor_label_matrix = torch.full(
            #     (self.partition_size, self.partition_size), smooth_value, device=student_embeds.device)
            # anchor_label_matrix.scatter_(1, torch.arange(self.partition_size, device=student_embeds.device).unsqueeze(1), torch.full_like(anchor_label_matrix, 1-self.smoothing))
            # target_probs = torch.cat((torch.softmax(similarities.t() / 0.1, dim=-1), anchor_label_matrix), dim=0)
            # partition_label_matrix = torch.take_along_dim(target_probs, indices=indices_.flatten().unsqueeze(1), dim=0)

            student_logits = student_embeds @ partition_neigh_embeddings / student_temp
            teacher_logits = teacher_embeds @ partition_neigh_embeddings / teacher_temp

            student_partition_probs = torch.softmax(
                student_logits, dim=-1) @ partition_label_matrix
            teacher_partition_probs = torch.softmax(
                teacher_logits, dim=-1) @ partition_label_matrix

            # print(student_partition_probs.sum(-1).max(), teacher_partition_probs.sum(-1).max())
            # assert torch.allclose(student_partition_probs.sum(-1).max(), torch.ones_like(
            #     student_partition_probs.sum(-1).max())), f"{student_partition_probs.sum(-1).max()}"
            # assert torch.allclose(teacher_partition_probs.sum(-1).max(), torch.ones_like(
            #     teacher_partition_probs.sum(-1).max())), f"{teacher_partition_probs.sum(-1).max()}"
            # assert student_partition_probs.shape[1] == self.partition_size
            # assert teacher_partition_probs.shape[1] == self.partition_size

            student_partition_probs_list.append(student_partition_probs)
            teacher_partition_probs_list.append(teacher_partition_probs)

        return student_partition_probs_list, teacher_partition_probs_list