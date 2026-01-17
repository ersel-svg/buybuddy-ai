"""
Advanced Batch Samplers for SOTA Training.

Includes:
- PKDomainSampler: P products x K samples with domain balancing
- CurriculumSampler: Progressive difficulty sampling
"""

import random
from collections import defaultdict
from typing import Iterator, List, Optional, Dict
import torch
from torch.utils.data import Sampler


class PKDomainSampler(Sampler):
    """
    P-K Sampler with Domain Balancing.

    Each batch contains:
    - P different products (classes)
    - K samples per product
    - Balanced synthetic/real ratio

    This ensures:
    - Multiple views of same product (for positives in triplet)
    - Multiple products (for negatives in triplet)
    - Mix of synthetic and real (for domain adaptation)

    Batch size = P × K
    """

    def __init__(
        self,
        labels: List[int],
        domains: Optional[List[int]] = None,
        products_per_batch: int = 8,
        samples_per_product: int = 4,
        synthetic_ratio: float = 0.5,
        min_samples_per_class: int = 2,
        drop_last: bool = True,
    ):
        """
        Args:
            labels: List of class indices for each sample
            domains: List of domain labels (0=synthetic, 1=real), optional
            products_per_batch: Number of different products per batch (P)
            samples_per_product: Number of samples per product per batch (K)
            synthetic_ratio: Target ratio of synthetic samples (0.0 to 1.0)
            min_samples_per_class: Minimum samples required per class
            drop_last: Drop incomplete batches
        """
        self.labels = labels
        self.domains = domains if domains is not None else [1] * len(labels)
        self.products_per_batch = products_per_batch
        self.samples_per_product = samples_per_product
        self.synthetic_ratio = synthetic_ratio
        self.min_samples_per_class = min_samples_per_class
        self.drop_last = drop_last

        self.batch_size = products_per_batch * samples_per_product

        # Build indices
        self._build_indices()

    def _build_indices(self):
        """Build class and domain index mappings."""
        # Class to sample indices
        self.class_to_indices = defaultdict(list)
        for idx, label in enumerate(self.labels):
            self.class_to_indices[label].append(idx)

        # Filter classes with enough samples
        self.valid_classes = [
            c for c, indices in self.class_to_indices.items()
            if len(indices) >= self.min_samples_per_class
        ]

        # Separate by domain
        self.synthetic_classes = set()
        self.real_classes = set()

        for idx, (label, domain) in enumerate(zip(self.labels, self.domains)):
            if label in self.valid_classes:
                if domain == 0:  # Synthetic
                    self.synthetic_classes.add(label)
                else:  # Real
                    self.real_classes.add(label)

        # Convert to lists
        self.synthetic_classes = list(self.synthetic_classes)
        self.real_classes = list(self.real_classes)

        # Total classes
        self.all_classes = list(set(self.synthetic_classes + self.real_classes))

        print(f"PKDomainSampler initialized:")
        print(f"  Total classes: {len(self.all_classes)}")
        print(f"  Synthetic classes: {len(self.synthetic_classes)}")
        print(f"  Real classes: {len(self.real_classes)}")
        print(f"  Batch size: {self.batch_size} (P={self.products_per_batch}, K={self.samples_per_product})")

    def __iter__(self) -> Iterator[List[int]]:
        """Generate batches."""
        # Shuffle class sample lists
        class_indices = {
            c: random.sample(indices, len(indices))
            for c, indices in self.class_to_indices.items()
            if c in self.all_classes
        }
        class_pointers = {c: 0 for c in self.all_classes}

        # Calculate products per domain
        num_synthetic = int(self.products_per_batch * self.synthetic_ratio)
        num_real = self.products_per_batch - num_synthetic

        # Adjust if not enough classes
        if len(self.synthetic_classes) < num_synthetic:
            num_synthetic = len(self.synthetic_classes)
            num_real = min(len(self.real_classes), self.products_per_batch - num_synthetic)
        if len(self.real_classes) < num_real:
            num_real = len(self.real_classes)
            num_synthetic = min(len(self.synthetic_classes), self.products_per_batch - num_real)

        # Available classes
        available_synthetic = set(self.synthetic_classes)
        available_real = set(self.real_classes)

        syn_list = list(available_synthetic)
        real_list = list(available_real)
        random.shuffle(syn_list)
        random.shuffle(real_list)
        syn_ptr, real_ptr = 0, 0

        while True:
            batch = []
            batch_classes = []

            # Sample synthetic classes
            for _ in range(num_synthetic):
                if syn_ptr >= len(syn_list):
                    # Refresh and reshuffle
                    syn_list = [c for c in self.synthetic_classes if c in available_synthetic]
                    if not syn_list:
                        break
                    random.shuffle(syn_list)
                    syn_ptr = 0

                cls = syn_list[syn_ptr]
                syn_ptr += 1
                batch_classes.append(cls)

            # Sample real classes
            for _ in range(num_real):
                if real_ptr >= len(real_list):
                    real_list = [c for c in self.real_classes if c in available_real]
                    if not real_list:
                        break
                    random.shuffle(real_list)
                    real_ptr = 0

                cls = real_list[real_ptr]
                real_ptr += 1
                batch_classes.append(cls)

            if len(batch_classes) < self.products_per_batch:
                # Fill remaining with any available class
                remaining = self.products_per_batch - len(batch_classes)
                other_classes = [c for c in self.all_classes if c not in batch_classes]
                if other_classes:
                    batch_classes.extend(random.sample(other_classes, min(remaining, len(other_classes))))

            if len(batch_classes) < 2:
                break

            # Sample K instances per class
            for cls in batch_classes:
                indices = class_indices.get(cls, [])
                ptr = class_pointers.get(cls, 0)

                sampled = []
                for _ in range(self.samples_per_product):
                    if ptr >= len(indices):
                        # Reshuffle
                        indices = random.sample(self.class_to_indices[cls], len(self.class_to_indices[cls]))
                        class_indices[cls] = indices
                        ptr = 0
                        available_synthetic.discard(cls)
                        available_real.discard(cls)

                    sampled.append(indices[ptr])
                    ptr += 1

                class_pointers[cls] = ptr
                batch.extend(sampled)

            if len(batch) >= self.batch_size:
                yield batch[:self.batch_size]
            elif not self.drop_last and len(batch) > 0:
                yield batch

            # Check if exhausted
            if not available_synthetic and not available_real:
                break

    def __len__(self) -> int:
        """Approximate number of batches."""
        total_samples = sum(len(v) for v in self.class_to_indices.values())
        return total_samples // self.batch_size


class CurriculumSampler(Sampler):
    """
    Curriculum Learning Sampler.

    Progressively increases training difficulty:
    1. Warmup: Random easy samples
    2. Easy: Wide margin triplets (easy negatives)
    3. Hard: Tight margin triplets (hard negatives)
    4. Finetune: Focus on hardest cases

    Requires pre-computed triplet difficulties.
    """

    def __init__(
        self,
        labels: List[int],
        difficulties: Optional[List[str]] = None,
        batch_size: int = 32,
        current_phase: str = "warmup",
    ):
        """
        Args:
            labels: List of class indices
            difficulties: List of difficulty labels ('easy', 'semi_hard', 'hard')
            batch_size: Batch size
            current_phase: Current curriculum phase
        """
        self.labels = labels
        self.difficulties = difficulties or ['easy'] * len(labels)
        self.batch_size = batch_size
        self.current_phase = current_phase

        # Index by difficulty
        self.easy_indices = [i for i, d in enumerate(self.difficulties) if d == 'easy']
        self.semi_hard_indices = [i for i, d in enumerate(self.difficulties) if d == 'semi_hard']
        self.hard_indices = [i for i, d in enumerate(self.difficulties) if d == 'hard']

        # All indices for warmup/random
        self.all_indices = list(range(len(labels)))

    def set_phase(self, phase: str):
        """Set curriculum phase."""
        self.current_phase = phase
        print(f"Curriculum phase set to: {phase}")

    def __iter__(self) -> Iterator[List[int]]:
        """Generate batches based on current phase."""
        if self.current_phase == "warmup":
            # Random sampling
            indices = self.all_indices.copy()
            random.shuffle(indices)

        elif self.current_phase == "easy":
            # Mostly easy + some semi-hard
            indices = (
                self.easy_indices +
                self.semi_hard_indices[:len(self.easy_indices) // 2]
            )
            random.shuffle(indices)

        elif self.current_phase == "hard":
            # All with emphasis on hard
            indices = (
                self.hard_indices * 2 +  # Double weight for hard
                self.semi_hard_indices +
                self.easy_indices[:len(self.hard_indices)]
            )
            random.shuffle(indices)

        elif self.current_phase == "finetune":
            # Focus on hardest
            indices = (
                self.hard_indices +
                self.semi_hard_indices[:len(self.hard_indices) // 2]
            )
            random.shuffle(indices)

        else:
            indices = self.all_indices.copy()
            random.shuffle(indices)

        # Yield batches
        for i in range(0, len(indices), self.batch_size):
            batch = indices[i:i + self.batch_size]
            if len(batch) == self.batch_size:
                yield batch
            elif len(batch) > 0:
                yield batch

    def __len__(self) -> int:
        return len(self.all_indices) // self.batch_size


class TripletBatchSampler(Sampler):
    """
    Sampler that creates batches with pre-mined triplets.

    Each batch contains complete triplets (anchor, positive, negative).
    Useful when using pre-computed triplets from triplet mining.
    """

    def __init__(
        self,
        triplets: List[Dict],
        samples_per_batch: int = 32,
        shuffle: bool = True,
    ):
        """
        Args:
            triplets: List of triplet dicts with keys:
                - anchor_idx: Index of anchor sample
                - positive_idx: Index of positive sample
                - negative_idx: Index of negative sample
            samples_per_batch: Number of triplets per batch
            shuffle: Shuffle triplets each epoch
        """
        self.triplets = triplets
        self.samples_per_batch = samples_per_batch
        self.shuffle = shuffle

    def __iter__(self) -> Iterator[List[int]]:
        """Generate batches of triplet indices."""
        triplet_indices = list(range(len(self.triplets)))
        if self.shuffle:
            random.shuffle(triplet_indices)

        for i in range(0, len(triplet_indices), self.samples_per_batch):
            batch_triplet_indices = triplet_indices[i:i + self.samples_per_batch]

            # Collect all unique sample indices
            batch_sample_indices = set()
            for ti in batch_triplet_indices:
                triplet = self.triplets[ti]
                batch_sample_indices.add(triplet["anchor_idx"])
                batch_sample_indices.add(triplet["positive_idx"])
                batch_sample_indices.add(triplet["negative_idx"])

            yield list(batch_sample_indices)

    def __len__(self) -> int:
        return len(self.triplets) // self.samples_per_batch


class StratifiedDomainSampler(Sampler):
    """
    Sampler that stratifies by both class and domain.

    Ensures each batch has:
    - Multiple classes
    - Balanced representation from each domain
    - Stratified sampling within each class-domain combination
    """

    def __init__(
        self,
        labels: List[int],
        domains: List[int],
        batch_size: int = 32,
        domain_balance: str = "equal",  # "equal", "proportional"
    ):
        """
        Args:
            labels: Class labels
            domains: Domain labels (0=synthetic, 1=real)
            batch_size: Batch size
            domain_balance: How to balance domains
        """
        self.labels = labels
        self.domains = domains
        self.batch_size = batch_size
        self.domain_balance = domain_balance

        # Build class-domain mapping
        self.class_domain_indices = defaultdict(lambda: defaultdict(list))
        for idx, (label, domain) in enumerate(zip(labels, domains)):
            self.class_domain_indices[label][domain].append(idx)

        self.classes = list(self.class_domain_indices.keys())

    def __iter__(self) -> Iterator[List[int]]:
        """Generate stratified batches."""
        # Shuffle within each class-domain group
        shuffled = {}
        for cls in self.classes:
            shuffled[cls] = {}
            for domain in [0, 1]:
                indices = self.class_domain_indices[cls][domain]
                if indices:
                    shuffled[cls][domain] = random.sample(indices, len(indices))

        pointers = {cls: {0: 0, 1: 0} for cls in self.classes}
        available_classes = set(self.classes)

        while available_classes:
            batch = []
            sampled_classes = random.sample(
                list(available_classes),
                min(len(available_classes), self.batch_size // 2)
            )

            for cls in sampled_classes:
                # Try to get one from each domain
                for domain in [0, 1]:
                    indices = shuffled[cls].get(domain, [])
                    ptr = pointers[cls][domain]

                    if ptr < len(indices):
                        batch.append(indices[ptr])
                        pointers[cls][domain] = ptr + 1
                    else:
                        # Domain exhausted
                        pass

                # Check if class exhausted
                all_exhausted = all(
                    pointers[cls][d] >= len(shuffled[cls].get(d, []))
                    for d in [0, 1]
                )
                if all_exhausted:
                    available_classes.discard(cls)

            if len(batch) > 0:
                yield batch

    def __len__(self) -> int:
        return len(self.labels) // self.batch_size
