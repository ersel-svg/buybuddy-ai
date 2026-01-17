"""
Model Evaluator for Embedding Models.

Evaluates trained models on test set with metrics:
- Recall@K (K=1, 5, 10)
- Mean Average Precision (mAP)
- Accuracy
- AUC

NOTE: Evaluation is done using product_id as ground truth.
The model should retrieve images of the same product_id.
"""

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Optional
from collections import defaultdict
from tqdm import tqdm


class ModelEvaluator:
    """
    Evaluate embedding model quality.

    Computes retrieval metrics by:
    1. Extracting embeddings for all test images
    2. For each query, finding nearest neighbors
    3. Checking if neighbors have the same product_id
    """

    def __init__(
        self,
        model: torch.nn.Module,
        model_type: str,
        device: Optional[torch.device] = None,
        batch_size: int = 64,
    ):
        """
        Initialize the evaluator.

        Args:
            model: The trained model
            model_type: Model type identifier
            device: Device to run on
            batch_size: Batch size for embedding extraction
        """
        self.model = model
        self.model_type = model_type
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size

        self.model.to(self.device)
        self.model.eval()

    def extract_embeddings(self, dataset) -> tuple[np.ndarray, list[str], list[int]]:
        """
        Extract embeddings for all samples in dataset.

        Args:
            dataset: The dataset to extract from

        Returns:
            Tuple of (embeddings, product_ids, frame_indices)
        """
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )

        all_embeddings = []
        all_product_ids = []
        all_frame_indices = []

        with torch.no_grad():
            for batch in tqdm(loader, desc="Extracting embeddings"):
                images = batch["image"].to(self.device)
                product_ids = batch["product_id"]
                frame_indices = batch["frame_idx"]

                # Get embeddings
                if hasattr(self.model, "get_embedding"):
                    embeddings = self.model.get_embedding(images)
                else:
                    outputs = self.model(images)
                    if isinstance(outputs, dict):
                        embeddings = outputs["embeddings"]
                    else:
                        embeddings = outputs

                embeddings = F.normalize(embeddings, p=2, dim=1)

                all_embeddings.append(embeddings.cpu().numpy())
                all_product_ids.extend(product_ids)
                all_frame_indices.extend(frame_indices.tolist())

        embeddings = np.vstack(all_embeddings)
        return embeddings, all_product_ids, all_frame_indices

    def evaluate(
        self,
        test_dataset,
        k_values: list[int] = [1, 5, 10],
    ) -> dict:
        """
        Evaluate model on test dataset.

        Args:
            test_dataset: Test dataset
            k_values: K values for Recall@K

        Returns:
            Dictionary of metrics
        """
        print(f"Evaluating on {len(test_dataset)} samples...")

        # Extract embeddings
        embeddings, product_ids, frame_indices = self.extract_embeddings(test_dataset)
        n_samples = len(embeddings)

        print(f"Extracted {n_samples} embeddings")

        # Build product_id to indices mapping
        product_to_indices = defaultdict(list)
        for idx, product_id in enumerate(product_ids):
            product_to_indices[product_id].append(idx)

        # Compute similarity matrix
        print("Computing similarity matrix...")
        similarity_matrix = self._compute_similarity_matrix(embeddings)

        # Compute metrics
        metrics = {}

        # Recall@K
        for k in k_values:
            recall = self._compute_recall_at_k(
                similarity_matrix,
                product_ids,
                product_to_indices,
                k=k,
            )
            metrics[f"recall@{k}"] = recall

        # Mean Average Precision
        map_score = self._compute_map(
            similarity_matrix,
            product_ids,
            product_to_indices,
        )
        metrics["mAP"] = map_score

        # Accuracy (top-1 classification accuracy)
        accuracy = self._compute_accuracy(test_dataset)
        metrics["accuracy"] = accuracy

        # Print metrics
        print("\nEvaluation Results:")
        for name, value in metrics.items():
            print(f"  {name}: {value:.4f}")

        return metrics

    def _compute_similarity_matrix(self, embeddings: np.ndarray) -> np.ndarray:
        """Compute cosine similarity matrix."""
        # Normalize (should already be normalized, but ensure)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        normalized = embeddings / (norms + 1e-8)

        # Cosine similarity
        similarity = np.dot(normalized, normalized.T)

        return similarity

    def _compute_recall_at_k(
        self,
        similarity_matrix: np.ndarray,
        product_ids: list[str],
        product_to_indices: dict,
        k: int,
    ) -> float:
        """
        Compute Recall@K.

        For each query, check if any of the top-K neighbors
        (excluding self) has the same product_id.
        """
        n_samples = len(product_ids)
        hits = 0

        for i in range(n_samples):
            query_product_id = product_ids[i]

            # Get top K+1 (to exclude self)
            similarities = similarity_matrix[i]
            top_indices = np.argsort(similarities)[::-1][:k + 1]

            # Check if any neighbor (excluding self) has same product_id
            for idx in top_indices:
                if idx != i and product_ids[idx] == query_product_id:
                    hits += 1
                    break

        return hits / n_samples

    def _compute_map(
        self,
        similarity_matrix: np.ndarray,
        product_ids: list[str],
        product_to_indices: dict,
    ) -> float:
        """
        Compute Mean Average Precision.

        For each query, compute AP considering all positives.
        """
        n_samples = len(product_ids)
        ap_sum = 0.0

        for i in range(n_samples):
            query_product_id = product_ids[i]

            # Get all positive indices (same product_id, excluding self)
            positive_indices = set(product_to_indices[query_product_id]) - {i}

            if not positive_indices:
                continue

            # Sort by similarity (descending)
            similarities = similarity_matrix[i]
            sorted_indices = np.argsort(similarities)[::-1]

            # Compute AP
            ap = 0.0
            num_positives_found = 0

            for rank, idx in enumerate(sorted_indices):
                if idx == i:
                    continue

                if idx in positive_indices:
                    num_positives_found += 1
                    precision_at_rank = num_positives_found / (rank + 1)
                    ap += precision_at_rank

            if num_positives_found > 0:
                ap /= len(positive_indices)
                ap_sum += ap

        return ap_sum / n_samples

    def _compute_accuracy(self, test_dataset) -> float:
        """
        Compute classification accuracy using the model's classifier head.
        """
        loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )

        correct = 0
        total = 0

        with torch.no_grad():
            for batch in loader:
                images = batch["image"].to(self.device)
                labels = batch["label"].to(self.device)

                outputs = self.model(images, labels)

                if isinstance(outputs, dict) and "logits" in outputs:
                    logits = outputs["logits"]
                    preds = logits.argmax(dim=1)
                    correct += (preds == labels).sum().item()
                    total += labels.size(0)

        return correct / total if total > 0 else 0.0


class CrossModalEvaluator:
    """
    Evaluate cross-modal retrieval (products vs cutouts).

    Used when you have product embeddings and want to
    match against cutout embeddings.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        device: Optional[torch.device] = None,
    ):
        self.model = model
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

    def evaluate_retrieval(
        self,
        product_embeddings: np.ndarray,
        product_ids: list[str],
        cutout_embeddings: np.ndarray,
        cutout_product_ids: list[str],
        k_values: list[int] = [1, 5, 10],
    ) -> dict:
        """
        Evaluate product-to-cutout retrieval.

        Args:
            product_embeddings: Embeddings of products
            product_ids: Product IDs for products
            cutout_embeddings: Embeddings of cutouts
            cutout_product_ids: Product IDs for cutouts (ground truth)
            k_values: K values for Recall@K

        Returns:
            Dictionary of metrics
        """
        # Normalize
        product_embeddings = product_embeddings / (
            np.linalg.norm(product_embeddings, axis=1, keepdims=True) + 1e-8
        )
        cutout_embeddings = cutout_embeddings / (
            np.linalg.norm(cutout_embeddings, axis=1, keepdims=True) + 1e-8
        )

        # Compute cross-similarity: products x cutouts
        similarity = np.dot(product_embeddings, cutout_embeddings.T)

        metrics = {}

        # For each product, find matching cutouts
        for k in k_values:
            hits = 0
            for i, prod_id in enumerate(product_ids):
                # Get top-K cutouts
                top_indices = np.argsort(similarity[i])[::-1][:k]

                # Check if any matches
                for idx in top_indices:
                    if cutout_product_ids[idx] == prod_id:
                        hits += 1
                        break

            metrics[f"product_to_cutout_recall@{k}"] = hits / len(product_ids)

        # Reverse direction: cutout to product
        for k in k_values:
            hits = 0
            for j, cutout_prod_id in enumerate(cutout_product_ids):
                # Get top-K products
                cutout_sim = similarity[:, j]
                top_indices = np.argsort(cutout_sim)[::-1][:k]

                for idx in top_indices:
                    if product_ids[idx] == cutout_prod_id:
                        hits += 1
                        break

            metrics[f"cutout_to_product_recall@{k}"] = hits / len(cutout_product_ids)

        return metrics


class DomainAwareEvaluator:
    """
    Evaluator that separately assesses cross-domain performance.

    Computes:
    - Real → Synthetic retrieval (using real as query, find synthetic)
    - Synthetic → Real retrieval (using synthetic as query, find real)
    - Per-category breakdown
    - Hard example detection
    """

    def __init__(
        self,
        model: torch.nn.Module,
        device: Optional[torch.device] = None,
        batch_size: int = 64,
    ):
        self.model = model
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.model.to(self.device)
        self.model.eval()

    def extract_embeddings_with_domain(
        self,
        dataset
    ) -> tuple[np.ndarray, list[str], list[str], list[str]]:
        """
        Extract embeddings with domain and category information.

        Returns:
            (embeddings, product_ids, domains, categories)
        """
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )

        all_embeddings = []
        all_product_ids = []
        all_domains = []
        all_categories = []

        with torch.no_grad():
            for batch in tqdm(loader, desc="Extracting embeddings"):
                images = batch["image"].to(self.device)
                product_ids = batch["product_id"]

                # Get domain if available
                if "domain" in batch:
                    domains = batch["domain"]
                else:
                    domains = ["unknown"] * len(product_ids)

                # Get category if available
                if "category" in batch:
                    categories = batch["category"]
                else:
                    categories = ["unknown"] * len(product_ids)

                # Get embeddings
                if hasattr(self.model, "get_embedding"):
                    embeddings = self.model.get_embedding(images)
                else:
                    outputs = self.model(images)
                    if isinstance(outputs, dict):
                        embeddings = outputs["embeddings"]
                    else:
                        embeddings = outputs

                embeddings = F.normalize(embeddings, p=2, dim=1)

                all_embeddings.append(embeddings.cpu().numpy())
                all_product_ids.extend(product_ids)
                all_domains.extend(domains if isinstance(domains, list) else domains.tolist())
                all_categories.extend(categories if isinstance(categories, list) else categories.tolist())

        embeddings = np.vstack(all_embeddings)
        return embeddings, all_product_ids, all_domains, all_categories

    def evaluate_cross_domain(
        self,
        dataset,
        k_values: list[int] = [1, 5, 10],
    ) -> dict:
        """
        Evaluate with cross-domain breakdown.

        Returns metrics for:
        - Overall performance
        - Synthetic → Real (query synthetic, retrieve from real)
        - Real → Synthetic (query real, retrieve from synthetic)
        - Per-category breakdown
        """
        print(f"Evaluating {len(dataset)} samples with cross-domain analysis...")

        embeddings, product_ids, domains, categories = self.extract_embeddings_with_domain(dataset)
        n_samples = len(embeddings)

        # Build indices
        product_to_indices = defaultdict(list)
        domain_indices = {"synthetic": [], "real": [], "unknown": []}
        category_indices = defaultdict(list)

        for idx, (pid, domain, category) in enumerate(zip(product_ids, domains, categories)):
            product_to_indices[pid].append(idx)
            domain_indices[domain].append(idx)
            category_indices[category].append(idx)

        # Compute similarity matrix
        print("Computing similarity matrix...")
        similarity_matrix = np.dot(embeddings, embeddings.T)

        metrics = {}

        # Overall metrics
        for k in k_values:
            recall = self._compute_recall_at_k(
                similarity_matrix, product_ids, product_to_indices, k
            )
            metrics[f"recall@{k}"] = recall

        # Cross-domain: Synthetic → Real
        synth_indices = domain_indices["synthetic"]
        real_indices = domain_indices["real"]

        if synth_indices and real_indices:
            for k in k_values:
                recall = self._compute_cross_domain_recall(
                    embeddings, product_ids,
                    synth_indices, real_indices, k
                )
                metrics[f"synth_to_real_recall@{k}"] = recall

            for k in k_values:
                recall = self._compute_cross_domain_recall(
                    embeddings, product_ids,
                    real_indices, synth_indices, k
                )
                metrics[f"real_to_synth_recall@{k}"] = recall

        # Per-category metrics
        per_category = {}
        for category, cat_indices in category_indices.items():
            if len(cat_indices) < 10:  # Skip categories with too few samples
                continue

            cat_products = [product_ids[i] for i in cat_indices]
            cat_product_to_indices = defaultdict(list)
            for i, idx in enumerate(cat_indices):
                cat_product_to_indices[product_ids[idx]].append(i)

            # Get submatrix
            cat_embeddings = embeddings[cat_indices]
            cat_similarity = np.dot(cat_embeddings, cat_embeddings.T)

            recall_1 = self._compute_recall_at_k(
                cat_similarity, cat_products, cat_product_to_indices, 1
            )
            per_category[category] = {"recall@1": recall_1, "count": len(cat_indices)}

        metrics["per_category"] = per_category

        # Find hard examples (worst performing products)
        hard_examples = self._find_hard_examples(
            embeddings, product_ids, product_to_indices, top_n=20
        )
        metrics["hard_examples"] = hard_examples

        # Find most confused pairs
        confused_pairs = self._find_confused_pairs(
            embeddings, product_ids, top_n=10
        )
        metrics["confused_pairs"] = confused_pairs

        return metrics

    def _compute_recall_at_k(
        self,
        similarity_matrix: np.ndarray,
        product_ids: list[str],
        product_to_indices: dict,
        k: int,
    ) -> float:
        n_samples = len(product_ids)
        hits = 0

        for i in range(n_samples):
            query_product_id = product_ids[i]
            similarities = similarity_matrix[i]
            top_indices = np.argsort(similarities)[::-1][:k + 1]

            for idx in top_indices:
                if idx != i and product_ids[idx] == query_product_id:
                    hits += 1
                    break

        return hits / n_samples

    def _compute_cross_domain_recall(
        self,
        embeddings: np.ndarray,
        product_ids: list[str],
        query_indices: list[int],
        gallery_indices: list[int],
        k: int,
    ) -> float:
        """Compute recall when querying from one domain against another."""
        query_embeddings = embeddings[query_indices]
        gallery_embeddings = embeddings[gallery_indices]
        query_product_ids = [product_ids[i] for i in query_indices]
        gallery_product_ids = [product_ids[i] for i in gallery_indices]

        # Cross-similarity
        similarity = np.dot(query_embeddings, gallery_embeddings.T)

        hits = 0
        for i, query_pid in enumerate(query_product_ids):
            top_indices = np.argsort(similarity[i])[::-1][:k]

            for idx in top_indices:
                if gallery_product_ids[idx] == query_pid:
                    hits += 1
                    break

        return hits / len(query_indices) if query_indices else 0.0

    def _find_hard_examples(
        self,
        embeddings: np.ndarray,
        product_ids: list[str],
        product_to_indices: dict,
        top_n: int = 20,
    ) -> list[dict]:
        """Find products with worst retrieval performance."""
        # Compute per-product recall@1
        similarity_matrix = np.dot(embeddings, embeddings.T)

        product_recalls = {}
        for pid, indices in product_to_indices.items():
            if len(indices) < 2:
                continue

            hits = 0
            for idx in indices:
                similarities = similarity_matrix[idx]
                top_idx = np.argsort(similarities)[::-1][1]  # Exclude self
                if product_ids[top_idx] == pid:
                    hits += 1

            product_recalls[pid] = hits / len(indices)

        # Sort by recall (ascending = worst first)
        sorted_products = sorted(product_recalls.items(), key=lambda x: x[1])

        return [
            {"product_id": pid, "recall@1": recall}
            for pid, recall in sorted_products[:top_n]
        ]

    def _find_confused_pairs(
        self,
        embeddings: np.ndarray,
        product_ids: list[str],
        top_n: int = 10,
    ) -> list[dict]:
        """Find pairs of different products that are often confused."""
        similarity_matrix = np.dot(embeddings, embeddings.T)

        # Find high-similarity pairs with different product_ids
        confused = []
        n = len(embeddings)

        for i in range(n):
            for j in range(i + 1, n):
                if product_ids[i] != product_ids[j]:
                    sim = similarity_matrix[i, j]
                    if sim > 0.7:  # Only high similarity
                        confused.append({
                            "product_id_1": product_ids[i],
                            "product_id_2": product_ids[j],
                            "similarity": float(sim),
                        })

        # Sort by similarity (descending)
        confused.sort(key=lambda x: x["similarity"], reverse=True)

        # Deduplicate (same pair can appear multiple times)
        seen_pairs = set()
        unique_confused = []
        for item in confused:
            pair = tuple(sorted([item["product_id_1"], item["product_id_2"]]))
            if pair not in seen_pairs:
                seen_pairs.add(pair)
                unique_confused.append(item)
                if len(unique_confused) >= top_n:
                    break

        return unique_confused


def compute_embedding_quality_metrics(
    embeddings: np.ndarray,
    labels: np.ndarray,
) -> dict:
    """
    Compute embedding quality metrics.

    Args:
        embeddings: Embedding vectors
        labels: Class labels

    Returns:
        Dictionary with metrics:
        - intra_class_distance: Average distance within same class
        - inter_class_distance: Average distance between classes
        - silhouette_score: Clustering quality measure
    """
    from sklearn.metrics import silhouette_score

    # Normalize
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / (norms + 1e-8)

    unique_labels = np.unique(labels)
    n_classes = len(unique_labels)

    # Compute class centroids
    centroids = np.zeros((n_classes, embeddings.shape[1]))
    for i, label in enumerate(unique_labels):
        mask = labels == label
        centroids[i] = embeddings[mask].mean(axis=0)

    # Intra-class distance (average distance to centroid)
    intra_distances = []
    for i, label in enumerate(unique_labels):
        mask = labels == label
        class_embeddings = embeddings[mask]
        distances = 1 - np.dot(class_embeddings, centroids[i])
        intra_distances.extend(distances.tolist())

    intra_class_distance = np.mean(intra_distances)

    # Inter-class distance (average distance between centroids)
    centroid_similarities = np.dot(centroids, centroids.T)
    np.fill_diagonal(centroid_similarities, 0)
    inter_class_distance = 1 - centroid_similarities.sum() / (n_classes * (n_classes - 1))

    # Silhouette score
    if n_classes > 1 and len(embeddings) > n_classes:
        try:
            silhouette = silhouette_score(embeddings, labels)
        except Exception:
            silhouette = 0.0
    else:
        silhouette = 0.0

    return {
        "intra_class_distance": float(intra_class_distance),
        "inter_class_distance": float(inter_class_distance),
        "silhouette_score": float(silhouette),
        "class_separation_ratio": float(inter_class_distance / (intra_class_distance + 1e-8)),
    }
