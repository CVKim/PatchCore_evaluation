import torch
import random
import abc
import gc
import numpy as np
from typing import Union
from tqdm import tqdm
from copy import deepcopy

class BaseSampler(abc.ABC):
    def __init__(self, percentage: float):
        if not 0 <= percentage <= 1:
            raise ValueError("Percentage value not in [0, 1].")
        self.percentage = percentage

    @abc.abstractmethod
    def run(
        self, features: Union[torch.Tensor, np.ndarray]
    ) -> Union[torch.Tensor, np.ndarray]:
        pass

    def _store_type(self, features: Union[torch.Tensor, np.ndarray]) -> None:
        self.features_is_numpy = isinstance(features, np.ndarray)
        if not self.features_is_numpy:
            self.features_device = features.device

    def _restore_type(self, features: torch.Tensor) -> Union[torch.Tensor, np.ndarray]:
        if self.features_is_numpy:
            return features.cpu().numpy()
        return features.clone().to(self.features_device)


class GreedyCoresetSampler(BaseSampler):
    def __init__(
        self,
        percentage: float,
        dimension_to_project_features_to: int,
        device: torch.device,
        brute: bool = True,
        max_coreset_shape=None,
    ):
        """Greedy Coreset sampling base class."""
        super().__init__(percentage)

        self.device = device
        self.dimension_to_project_features_to = dimension_to_project_features_to
        self.brute = brute

        if max_coreset_shape is not None:
            self.max_coreset_size = max_coreset_shape[0]
            self._max_coreset = torch.zeros(
                (int(max_coreset_shape[0] * (1 + percentage)), max_coreset_shape[1])
            )

    # FIXME: refactor
    def run(
        self,
        features: torch.Tensor,
        base_coreset: torch.Tensor = None,
    ) -> torch.Tensor:
        """Subsamples features using Greedy Coreset.

        Args:
            features: [N x D]
        """
        if self.percentage == 1:
            return features

        assert isinstance(features, torch.Tensor)
        assert isinstance(base_coreset, (torch.Tensor, type(None)))

        if base_coreset is not None:
            tqdm.write(f"coreset size: {len(base_coreset)}")

        if hasattr(self, "max_coreset_size"):
            base_coreset = self._suppress_coreset(base_coreset, num_repeats=100)

        reduced_features, reduced_base_coreset = self._reduce_features(
            features, base_features=base_coreset
        )
        sample_indices = self._compute_greedy_coreset_indices(
            reduced_features, base_coreset=reduced_base_coreset
        )
        sample_features = features[sample_indices]

        if hasattr(self, '_max_coreset'):
            n_features = len(sample_features)
            n_base_coreset = len(base_coreset) if base_coreset is not None else 0
            
            self._max_coreset[n_base_coreset : n_base_coreset + n_features] = sample_features
            sample_features = self._max_coreset[: n_features + n_base_coreset]
        else:
            if base_coreset is not None:
                sample_features = torch.cat([sample_features, base_coreset], dim=0)

        return sample_features, sample_indices

    def _reduce_features(
        self, features: torch.Tensor, base_features: torch.Tensor = None
    ):
        if features.shape[1] == self.dimension_to_project_features_to:
            return features
        mapper = torch.nn.Linear(
            features.shape[1], self.dimension_to_project_features_to, bias=False
        )
        _ = mapper.to(self.device)
        features = features.to(self.device)
        with torch.no_grad():
            reduced_features = mapper(features).cpu()

        if base_features is not None:
            base_features = base_features.to(self.device)
            with torch.no_grad():
                reduced_base_features = mapper(base_features).cpu()
        else:
            reduced_base_features = None

        return reduced_features, reduced_base_features

    def _compute_greedy_coreset_indices(
        self, features: torch.Tensor, base_coreset=None
    ) -> torch.Tensor:
        if self.brute:
            return self._brute_compute_greedy_coreset_indices(features, base_coreset)
        return self._parallel_compute_greedy_coreset_indices(features, base_coreset)

    def _brute_compute_greedy_coreset_indices(
        self, features: torch.Tensor, base_coreset=None
    ) -> torch.Tensor:
        """Runs iterative greedy coreset selection.

        Args:
            features: [NxD] input feature bank to sample.
        """
        # torch.cuda.empty_cache()
        # gc.collect()
        print("WARNING: Brute-force greedy-sampling.")
        features = features.to(self.device)
        num_coreset_samples = int(len(features) * self.percentage)
        coreset_indices = [np.random.randint(len(features))]

        if base_coreset is not None:
            base_coreset = base_coreset.to(self.device)
            min_distances = torch.min(
                torch.cdist(features[None, :, :], base_coreset[None, :, :])[0], dim=1
            )[0]
        else:
            min_distances = torch.ones(len(features)).to(features.device) * float("inf")

        for _ in tqdm(
            range(1, num_coreset_samples), desc="brute force greedy sample", leave=False
        ):
            current_distances = torch.norm(
                features - features[coreset_indices[-1]], dim=1
            )
            min_distances = torch.minimum(min_distances, current_distances)
            coreset_indices.append(torch.argmax(min_distances, dim=0).item())
            current_distances = current_distances.cpu()
            del current_distances
            gc.collect()
            torch.cuda.empty_cache()

        return coreset_indices

    def _parallel_compute_greedy_coreset_indices(
        self, features: torch.Tensor, base_coreset: torch.Tensor = None
    ) -> torch.Tensor:
        """Runs iterative greedy coreset selection.

        Args:
            features: [NxD] input feature bank to sample.
        """
        distance_matrix = self._compute_batchwise_differences(features, features)
        coreset_anchor_distances = torch.norm(
            distance_matrix, dim=1
        )  # TODO: reason why? dummy to initialize?

        if base_coreset is not None:
            coreset_anchor_distances = self._compute_set_distances(
                features, base_coreset, num_repeats=1
            )

        coreset_indices = []
        num_coreset_samples = int(len(features) * self.percentage)

        for _ in range(num_coreset_samples):
            select_idx = torch.argmax(coreset_anchor_distances).item()
            coreset_indices.append(select_idx)

            coreset_select_distance = distance_matrix[
                :, select_idx : select_idx + 1  # noqa E203
            ]
            coreset_anchor_distances = torch.cat(
                [coreset_anchor_distances.unsqueeze(-1), coreset_select_distance], dim=1
            )
            coreset_anchor_distances = torch.min(coreset_anchor_distances, dim=1).values

        return coreset_indices

    def _compute_batchwise_differences(
        self, matrix_a: torch.Tensor, matrix_b: torch.Tensor
    ) -> torch.Tensor:
        """Computes batchwise Euclidean distances using PyTorch."""

        matrix_a = matrix_a.to(self.device)
        matrix_b = matrix_b.to(self.device)

        a_times_a = matrix_a.unsqueeze(1).bmm(matrix_a.unsqueeze(2)).reshape(-1, 1)
        b_times_b = matrix_b.unsqueeze(1).bmm(matrix_b.unsqueeze(2)).reshape(1, -1)
        a_times_b = matrix_a.mm(matrix_b.T)

        return (-2 * a_times_b + a_times_a + b_times_b).clamp(0, None).sqrt().cpu()

    def _suppress_coreset(
        self, coreset: torch.Tensor, reduction_rate=0.25, num_repeats: int = None
    ) -> torch.Tensor:
        assert isinstance(self.max_coreset_size, int)

        if coreset is None:
            return coreset

        coreset_size = len(coreset)
        assert isinstance(coreset, torch.Tensor)

        if len(coreset) <= self.max_coreset_size:
            return coreset

        required_size = max(int(self.max_coreset_size * (1 - reduction_rate)), 1)

        print(f"max_coreset_size: {self.max_coreset_size}")
        print(f"coreset size BEFORE suppression: {coreset_size}")

        intra_set_dists = self._compute_intra_set_distances(
            coreset, num_repeats=num_repeats
        )
        n_feas_to_remove = coreset_size - required_size
        idxes_to_remove = torch.topk(-intra_set_dists, n_feas_to_remove).indices
        mask = torch.ones(coreset_size, dtype=torch.bool)
        mask[idxes_to_remove] = False

        coreset = coreset[mask]

        print(f"coreset size AFTER suppression: {len(coreset)}")

        return coreset

    def _compute_intra_set_distances(
        self, features: torch.Tensor, num_repeats: int = None
    ) -> torch.Tensor:
        n = len(features)
        batch_size = features.shape[-1]

        batch_indices = [(i, min(i + batch_size, n)) for i in range(0, n, batch_size)]
        indices = list(range(n))
        random.shuffle(indices)
        batch_indices = [
            indices[i : min(i + batch_size, n)] for i in range(0, n, batch_size)
        ]

        intra_set_distances = []

        # Loop over each batch with tqdm
        for batch_idx in tqdm(
            batch_indices, desc="Processing batches", leave=False
        ):
            batch_features = features[batch_idx]
            distances = self._compute_set_distances(
                batch_features, features, k=2, num_repeats=num_repeats
            )
            intra_set_distances.append(distances)

        intra_set_distances = torch.cat(intra_set_distances)
        
        return intra_set_distances

    def _compute_set_distances(
        self,
        features: torch.Tensor,
        coreset: torch.Tensor,
        num_repeats: int = None,
        k: int = 1,
    ) -> torch.Tensor:
        num_coreset_features = len(coreset)

        def divide_to_parts(input_list, target_len):
            return [
                input_list[i : i + target_len]
                for i in range(0, len(input_list), target_len)
            ]

        list_idxes = divide_to_parts(
            torch.randperm(num_coreset_features), len(features)
        )

        if num_repeats is None:
            num_repeats = len(list_idxes)

        for i in range(num_repeats):
            distance_matrix = self._compute_batchwise_differences(
                features, coreset[list_idxes[i]]
            )
            if k > 1:
                distances = -torch.topk(-distance_matrix, k, dim=1).values[:, -1]
            else:
                distances = torch.min(distance_matrix, dim=1).values
            if i == 0:
                min_distances = distances
            else:
                min_distances = torch.minimum(min_distances, distances)

        return min_distances
