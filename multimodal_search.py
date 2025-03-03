import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

class MultiModalSearch:
    def __init__(self, embedding_dim=200):
        """
        Initialize the multi-modal search system.

        Parameters:
        -----------
        embedding_dim : int
            Dimension of the embedding vectors
        """
        self.embedding_dim = embedding_dim
        self.modality_data = {}
        self.normalization_vectors = {}

    def add_modality(self, modality_name, embeddings):
        """
        Add a modality with its embeddings to the system.

        Parameters:
        -----------
        modality_name : str
            Name of the modality
        embeddings : numpy.ndarray
            Embedding vectors for this modality (shape: n_items x embedding_dim)
        """
        self.modality_data[modality_name] = embeddings
        self.normalization_vectors[modality_name] = self._compute_normalization_vector(embeddings)

    def _compute_normalization_vector(self, embeddings):
        """
        Compute the normalization vector for a set of embeddings.

        Parameters:
        -----------
        embeddings : numpy.ndarray
            Embedding vectors

        Returns:
        --------
        numpy.ndarray
            Normalization vector (standard deviation for each dimension)
        """
        return np.std(embeddings, axis=0)

    def _normalize_embeddings(self, embeddings, norm_vector):
        """
        Normalize embeddings using the provided normalization vector.

        Parameters:
        -----------
        embeddings : numpy.ndarray
            Embedding vectors to normalize
        norm_vector : numpy.ndarray
            Normalization vector

        Returns:
        --------
        numpy.ndarray
            Normalized embeddings
        """
        # Avoid division by zero
        safe_norm_vector = np.where(norm_vector > 1e-10, norm_vector, 1.0)
        return embeddings / safe_norm_vector

    def _probabilistic_scaling(self, embeddings, pf):
        """
        Apply probabilistic scaling to embeddings.

        Parameters:
        -----------
        embeddings : numpy.ndarray
            Embedding vectors to scale
        pf : float
            Probability factor [0, 1]

        Returns:
        --------
        numpy.ndarray
            Probabilistically scaled embeddings
        """
        # Generate random scaling factors between 0 and pf for each dimension
        scaling_factors = np.random.uniform(0, pf, self.embedding_dim)
        return embeddings * scaling_factors

    def search(self, query, query_modality, pf=1.0, num_trials=10, top_k=10):
        """
        Search across all modalities with the given query.

        Parameters:
        -----------
        query : numpy.ndarray
            The query embedding vector
        query_modality : str
            The modality of the query
        pf : float
            Probability factor [0, 1]
        num_trials : int
            Number of random scaling trials to average over
        top_k : int
            Number of top results to return

        Returns:
        --------
        list of tuples
            (modality, index, similarity score) for the top_k results
        """
        if query_modality not in self.normalization_vectors:
            raise ValueError(f"Unknown modality: {query_modality}")

        # Normalize the query
        norm_query = self.normalization_vectors[query_modality]
        normalized_query = self._normalize_embeddings(query, norm_query)

        # Prepare to collect results from multiple trials
        all_results = []

        for _ in range(num_trials):
            # Apply probabilistic scaling to the query
            scaled_query = self._probabilistic_scaling(normalized_query, pf)

            # Search in each modality
            trial_results = []

            for modality, embeddings in self.modality_data.items():
                # Get normalization vector for this modality
                norm_vector = self.normalization_vectors[modality]

                # Normalize the embeddings
                normalized_embeddings = self._normalize_embeddings(embeddings, norm_vector)

                # Apply the same probabilistic scaling factors
                scaled_embeddings = self._probabilistic_scaling(normalized_embeddings, pf)

                # Compute similarities
                similarities = cosine_similarity(scaled_query.reshape(1, -1), scaled_embeddings)[0]

                # Store results with modality info
                for idx, sim in enumerate(similarities):
                    trial_results.append((modality, idx, sim))

            # Sort by similarity for this trial
            trial_results.sort(key=lambda x: x[2], reverse=True)
            all_results.append(trial_results[:top_k])

        # Aggregate results across trials
        result_counts = {}
        for trial_top_k in all_results:
            for modality, idx, _ in trial_top_k:
                key = (modality, idx)
                result_counts[key] = result_counts.get(key, 0) + 1

        # Get the most consistently high-ranking items
        final_results = [(modality, idx, count / num_trials) for (modality, idx), count in result_counts.items()]
        final_results.sort(key=lambda x: x[2], reverse=True)

        return final_results[:top_k]

    def compute_raw_similarity(self, query, query_modality):
        """
        Compute raw similarity scores without normalization or probabilistic scaling.
        Used for comparison in evaluation.
        """
        results = []
        for modality, embeddings in self.modality_data.items():
            similarities = cosine_similarity(query.reshape(1, -1), embeddings)[0]
            for idx, sim in enumerate(similarities):
                results.append((modality, idx, sim))

        results.sort(key=lambda x: x[2], reverse=True)
        return results

def calculate_precision(search_results, relevance_data, top_k=10):
    """
    Calculate precision for search results.
    """
    results = search_results[:top_k]
    if not results:
        return 0.0

    relevant_count = 0
    for modality, idx, _ in results:
        if modality in relevance_data and idx < len(relevance_data[modality]):
            relevant_count += relevance_data[modality][idx]

    return relevant_count / len(results)

def generate_synthetic_data(num_modalities=2, items_per_modality=100, embedding_dim=200, seed=42):
    """
    Generate synthetic data with different feature distributions across modalities.
    """
    np.random.seed(seed)

    modalities = {}
    relevance = {}
    queries = {}

    # Create modalities with different feature distributions
    for i in range(num_modalities):
        # Each modality has high variance in a different subset of dimensions
        embeddings = np.random.normal(0, 0.1, (items_per_modality, embedding_dim))

        # Calculate which dimensions have high variance for this modality
        start_idx = i * embedding_dim // num_modalities
        end_idx = (i + 1) * embedding_dim // num_modalities

        # Set high variance for these dimensions
        embeddings[:, start_idx:end_idx] = np.random.normal(0, 1, (items_per_modality, end_idx - start_idx))

        modality_name = f"modality_{i}"
        modalities[modality_name] = embeddings

        # Create a query for this modality
        query = np.random.normal(0, 0.1, (1, embedding_dim))
        query[0, start_idx:end_idx] = np.random.normal(0, 1, (1, end_idx - start_idx))
        queries[modality_name] = query

        # Create ground truth relevance
        # 10% of items in each modality are relevant to the query
        num_relevant = items_per_modality // 10
        relevant_indices = np.random.choice(items_per_modality, size=num_relevant, replace=False)
        relevance_vector = np.zeros(items_per_modality)
        relevance_vector[relevant_indices] = 1

        # Make relevant items more similar to the query
        boost_factor = 0.5
        for idx in relevant_indices:
            modalities[modality_name][idx, start_idx:end_idx] += boost_factor * query[0, start_idx:end_idx]

        # Normalize embeddings
        norms = np.linalg.norm(modalities[modality_name], axis=1, keepdims=True)
        modalities[modality_name] /= norms

        relevance[modality_name] = relevance_vector

    # Create a multi-modal query that should match with items from all modalities
    multimodal_query = np.zeros((1, embedding_dim))
    for modality_name, query in queries.items():
        multimodal_query += query

    # Normalize the multimodal query
    multimodal_query /= np.linalg.norm(multimodal_query)
    queries["multimodal"] = multimodal_query

    return modalities, queries, relevance

def evaluate_approach(pf_values, num_trials=10, top_k=10):
    """
    Evaluate the multi-modal search approach with different probability factors.
    """
    # Generate synthetic data
    modalities, queries, relevance = generate_synthetic_data()

    # Initialize the search system
    search_system = MultiModalSearch()

    # Add modalities to the search system
    for modality_name, embeddings in modalities.items():
        search_system.add_modality(modality_name, embeddings)

    # Results for each query type
    results = {}
    baseline_results = {}

    # Evaluate for each query type
    for query_name, query in tqdm(queries.items(), desc="Evaluating queries"):
        query_results = []

        # If multimodal query, use modality_0 as the query modality (arbitrary choice)
        query_modality = "modality_0" if query_name == "multimodal" else query_name

        # Compute baseline (raw similarity without our approach)
        baseline_search_results = search_system.compute_raw_similarity(query, query_modality)
        baseline_precision = calculate_precision(baseline_search_results, relevance, top_k)
        baseline_results[query_name] = baseline_precision

        for pf in tqdm(pf_values, desc=f"Evaluating {query_name}", leave=False):
            # Search with this probability factor
            search_results = search_system.search(
                query, query_modality, pf=pf, num_trials=num_trials, top_k=top_k
            )

            # Calculate precision for this probability factor
            precision = calculate_precision(search_results, relevance, top_k)
            query_results.append(precision)

        results[query_name] = query_results

    return results, baseline_results

def plot_results(pf_values, results, baseline_results, top_k=10):
    """
    Plot evaluation results.
    """
    plt.figure(figsize=(12, 7))

    for query_name, precisions in results.items():
        plt.plot(pf_values, precisions, marker='o', linewidth=2, label=f"Query: {query_name}")
        # Add horizontal line for baseline
        plt.axhline(y=baseline_results[query_name], linestyle='--',
                    color=plt.gca().lines[-1].get_color(), alpha=0.7,
                    label=f"{query_name} baseline")

    plt.xlabel("Probability Factor (pf)", fontsize=12)
    plt.ylabel(f"Precision@{top_k}", fontsize=12)
    plt.title("Effect of Probability Factor on Multi-Modal Search", fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("multimodal_search_evaluation.png")

    # Print best probability factors
    print("\nBaseline results:")
    for query_name, precision in baseline_results.items():
        print(f"  {query_name}: {precision:.4f}")

    print("\nBest probability factors:")
    for query_name, precisions in results.items():
        best_pf_idx = np.argmax(precisions)
        best_pf = pf_values[best_pf_idx]
        best_precision = precisions[best_pf_idx]
        baseline = baseline_results[query_name]
        improvement = (best_precision - baseline) / baseline * 100 if baseline > 0 else float('inf')
        print(f"  {query_name}: pf={best_pf:.2f} (Precision: {best_precision:.4f}, Improvement: {improvement:.1f}%)")

def main():
    """Main function to run the evaluation."""
    # Parameters
    pf_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    num_trials = 10
    top_k = 10

    # Evaluate the approach
    results, baseline_results = evaluate_approach(pf_values, num_trials, top_k)

    # Plot the results
    plot_results(pf_values, results, baseline_results, top_k)

if __name__ == "__main__":
    main()