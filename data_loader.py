import os
import urllib.request
import zipfile
import tarfile
import gzip
import numpy as np
import pandas as pd
from tqdm import tqdm
import json
import h5py
import shutil
from scipy.spatial.distance import cdist

# Define dataset sources and metadata
DATASET_REGISTRY = {
    "msmarco": {
        "url": "https://msmarco.blob.core.windows.net/msmarcoranking/collection.tar.gz",
        "description": "MS MARCO passage dataset with text passages and queries",
        "size_gb": 2.5
    },
    "flickr30k": {
        "url": "http://shannon.cs.illinois.edu/DenotationGraph/data/flickr30k-images.tar.gz",
        "description": "Flickr30k image-caption pairs for cross-modal search",
        "size_gb": 1.2
    },
    "amazon-product": {
        "url": "https://jmcauley.ucsd.edu/data/amazon/amazon_5core.json.gz",
        "description": "Amazon product data with reviews and metadata",
        "size_gb": 0.8
    },
    "spotify-playlists": {
        "url": "https://www.aicrowd.com/challenges/spotify-million-playlist-dataset-challenge/dataset_files",
        "description": "Spotify Million Playlist Dataset",
        "size_gb": 5.4,
        "requires_login": True
    },
    "synthetic": {
        "url": None,  # Generated locally
        "description": "Synthetic multi-modal data for testing",
        "size_gb": 0.01
    }
}

# Utility functions for downloading and extracting datasets
def download_with_progress(url, destination):
    """Download a file with a progress bar."""
    class DownloadProgressBar(tqdm):
        def update_to(self, b=1, bsize=1, tsize=None):
            if tsize is not None:
                self.total = tsize
            self.update(b * bsize - self.n)

    with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc="Downloading") as t:
        urllib.request.urlretrieve(url, filename=destination, reporthook=t.update_to)

def extract_archive(archive_path, extract_path):
    """Extract a compressed archive file."""
    print(f"Extracting {archive_path} to {extract_path}...")

    if archive_path.endswith('.zip'):
        with zipfile.ZipFile(archive_path, 'r') as zip_ref:
            zip_ref.extractall(extract_path)
    elif archive_path.endswith('.tar.gz') or archive_path.endswith('.tgz'):
        with tarfile.open(archive_path, 'r:gz') as tar_ref:
            tar_ref.extractall(extract_path)
    elif archive_path.endswith('.tar'):
        with tarfile.open(archive_path, 'r:') as tar_ref:
            tar_ref.extractall(extract_path)
    elif archive_path.endswith('.gz') and not archive_path.endswith('.tar.gz'):
        output_path = os.path.join(extract_path, os.path.basename(archive_path)[:-3])  # Remove .gz extension
        with gzip.open(archive_path, 'rb') as f_in:
            with open(output_path, 'wb') as f_out:
                f_out.write(f_in.read())
    else:
        raise ValueError(f"Unsupported archive format: {archive_path}")

    print(f"Extraction complete.")

def check_dataset_exists(dataset_name, data_dir="./data"):
    """Check if a dataset already exists in processed form."""
    dataset_path = os.path.join(data_dir, dataset_name)
    processed_path = os.path.join(dataset_path, "processed")
    embeddings_file = os.path.join(processed_path, "embeddings.npy")
    metadata_file = os.path.join(processed_path, "metadata.json")

    return os.path.exists(embeddings_file) and os.path.exists(metadata_file)

def load_synthetic_dataset(data_dir, embedding_dim=300):
    """
    Generate a synthetic multi-modal dataset if it doesn't exist.

    Parameters:
    -----------
    data_dir : str
        Directory to store the dataset
    embedding_dim : int
        Dimension of embeddings

    Returns:
    --------
    dict
        Synthetic dataset in the required format
    """
    dataset_path = os.path.join(data_dir, "synthetic")
    processed_path = os.path.join(dataset_path, "processed")
    embeddings_file = os.path.join(processed_path, "embeddings.npy")
    metadata_file = os.path.join(processed_path, "metadata.json")

    # Create directory if it doesn't exist
    os.makedirs(processed_path, exist_ok=True)

    # If files already exist, load them
    if os.path.exists(embeddings_file) and os.path.exists(metadata_file):
        print("Loading synthetic dataset from disk...")
        embeddings = np.load(embeddings_file)

        with open(metadata_file, 'r') as f:
            metadata = json.load(f)

        return {
            "embeddings": embeddings,
            "queries": np.array(metadata["queries"]),
            "ground_truth": metadata["ground_truth"],
            "modalities": {int(k): v for k, v in metadata["modalities"].items()},
            "metadata": metadata
        }

    # Otherwise, generate new synthetic data
    print("Generating synthetic dataset...")
    np.random.seed(42)

    n_items = 1000
    n_queries = 50
    n_modalities = 3

    # Create embeddings for different modalities
    embeddings = []
    modalities = []

    for i in range(n_modalities):
        # Each modality has high variance in a different subset of dimensions
        modal_embeddings = np.random.normal(0, 0.1, (n_items // n_modalities, embedding_dim))

        start_idx = i * (embedding_dim // n_modalities)
        end_idx = (i + 1) * (embedding_dim // n_modalities)

        # Set high variance for these dimensions
        modal_embeddings[:, start_idx:end_idx] = np.random.normal(0, 1, (n_items // n_modalities, end_idx - start_idx))

        embeddings.append(modal_embeddings)
        modalities.extend([i] * (n_items // n_modalities))

    # Combine embeddings
    embeddings = np.vstack(embeddings)

    # Create queries
    queries = []
    ground_truth = []

    for i in range(n_queries):
        # Query is similar to items from a random modality
        modal_id = i % n_modalities

        query = np.random.normal(0, 0.1, embedding_dim)

        start_idx = modal_id * (embedding_dim // n_modalities)
        end_idx = (modal_id + 1) * (embedding_dim // n_modalities)

        query[start_idx:end_idx] = np.random.normal(0, 1, end_idx - start_idx)

        # Find relevant items for this query (from the same modality)
        modality_indices = [j for j, m in enumerate(modalities) if m == modal_id]
        relevant_count = 10  # 10 relevant items per query

        # Randomly select relevant items from this modality
        relevant_indices = np.random.choice(modality_indices, size=relevant_count, replace=False)

        queries.append(query)
        ground_truth.append(relevant_indices.tolist())

    # Convert modalities to a dictionary for indexing
    modality_dict = {i: modalities[i] for i in range(len(modalities))}

    # Save to disk
    np.save(embeddings_file, embeddings)

    metadata = {
        "queries": queries,
        "ground_truth": ground_truth,
        "modalities": modality_dict,
        "dimensions": embedding_dim,
        "num_items": n_items,
        "num_queries": n_queries,
        "num_modalities": n_modalities,
        "description": "Synthetic multi-modal dataset"
    }

    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else x)

    print(f"Synthetic dataset saved to {processed_path}")

    return {
        "embeddings": embeddings,
        "queries": np.array(queries),
        "ground_truth": ground_truth,
        "modalities": modality_dict,
        "metadata": metadata
    }

def load_msmarco_dataset(data_dir):
    """
    Load MS MARCO dataset.
    """
    dataset_name = "msmarco"
    dataset_path = os.path.join(data_dir, dataset_name)
    processed_path = os.path.join(dataset_path, "processed")
    embeddings_file = os.path.join(processed_path, "embeddings.npy")
    metadata_file = os.path.join(processed_path, "metadata.json")

    # Check if processed data exists
    if os.path.exists(embeddings_file) and os.path.exists(metadata_file):
        print(f"Loading processed {dataset_name} dataset from {processed_path}...")
        embeddings = np.load(embeddings_file)

        with open(metadata_file, 'r') as f:
            metadata = json.load(f)

        return {
            "embeddings": embeddings,
            "queries": np.array(metadata["queries"]),
            "ground_truth": metadata["ground_truth"],
            "modalities": {int(k): v for k, v in metadata["modalities"].items()},
            "metadata": metadata
        }

    # If processed data doesn't exist, check if raw data exists
    raw_data_path = os.path.join(dataset_path, "raw")
    os.makedirs(raw_data_path, exist_ok=True)

    # Define file paths
    collection_file = os.path.join(raw_data_path, "collection.tar.gz")

    # Download if not exists
    if not os.path.exists(collection_file):
        print(f"Downloading {dataset_name} dataset...")
        os.makedirs(raw_data_path, exist_ok=True)
        download_with_progress(DATASET_REGISTRY[dataset_name]["url"], collection_file)

        # Extract the archive
        extract_archive(collection_file, raw_data_path)

    # Process the dataset (simplified for this example)
    print(f"Processing {dataset_name} dataset...")
    os.makedirs(processed_path, exist_ok=True)

    # In a real implementation, we would process the actual MS MARCO data
    # For this example, we'll create a sample dataset with text embeddings

    embedding_dim = 768  # Typical BERT embedding dimension
    n_passages = 1000
    n_queries = 50

    # Create synthetic embeddings
    passage_embeddings = np.random.normal(0, 1, (n_passages, embedding_dim))
    passage_embeddings = passage_embeddings / np.linalg.norm(passage_embeddings, axis=1, keepdims=True)

    query_embeddings = np.random.normal(0, 1, (n_queries, embedding_dim))
    query_embeddings = query_embeddings / np.linalg.norm(query_embeddings, axis=1, keepdims=True)

    # Create ground truth judgments
    ground_truth = []
    for i in range(n_queries):
        # Find passages most similar to this query
        similarities = 1 - cdist([query_embeddings[i]], passage_embeddings, metric='cosine')[0]
        top_indices = np.argsort(-similarities)[:5]  # Top 5 passages
        ground_truth.append(top_indices.tolist())

    # Save processed data
    np.save(embeddings_file, passage_embeddings)

    # All passages are text modality (0)
    modalities = {i: 0 for i in range(n_passages)}

    metadata = {
        "queries": query_embeddings.tolist(),
        "ground_truth": ground_truth,
        "modalities": modalities,
        "dimensions": embedding_dim,
        "num_passages": n_passages,
        "num_queries": n_queries,
        "description": f"{dataset_name} dataset (simplified version)"
    }

    with open(metadata_file, 'w') as f:
        json.dump(metadata, f)

    print(f"Processed {dataset_name} dataset saved to {processed_path}")

    return {
        "embeddings": passage_embeddings,
        "queries": query_embeddings,
        "ground_truth": ground_truth,
        "modalities": modalities,
        "metadata": metadata
    }

def load_flickr30k_dataset(data_dir):
    """
    Load Flickr30k dataset.
    """
    dataset_name = "flickr30k"
    dataset_path = os.path.join(data_dir, dataset_name)
    processed_path = os.path.join(dataset_path, "processed")
    embeddings_file = os.path.join(processed_path, "embeddings.npy")
    metadata_file = os.path.join(processed_path, "metadata.json")

    # Check if processed data exists
    if os.path.exists(embeddings_file) and os.path.exists(metadata_file):
        print(f"Loading processed {dataset_name} dataset from {processed_path}...")
        embeddings = np.load(embeddings_file)

        with open(metadata_file, 'r') as f:
            metadata = json.load(f)

        return {
            "embeddings": embeddings,
            "queries": np.array(metadata["queries"]),
            "ground_truth": metadata["ground_truth"],
            "modalities": {int(k): v for k, v in metadata["modalities"].items()},
            "metadata": metadata
        }

    # If processed data doesn't exist, check if raw data exists
    raw_data_path = os.path.join(dataset_path, "raw")
    os.makedirs(raw_data_path, exist_ok=True)

    # Define file paths
    archive_file = os.path.join(raw_data_path, "flickr30k-images.tar.gz")

    # Download if not exists
    if not os.path.exists(archive_file):
        print(f"Downloading {dataset_name} dataset...")
        try:
            download_with_progress(DATASET_REGISTRY[dataset_name]["url"], archive_file)
            extract_archive(archive_file, raw_data_path)
        except Exception as e:
            print(f"Error downloading Flickr30k dataset: {e}")
            print("This dataset may require manual download. Please visit:")
            print("http://hockenmaier.cs.illinois.edu/DenotationGraph/")
            print("and request access to download the dataset.")
            print("After downloading, place the archive in:", raw_data_path)

            # Create synthetic dataset as fallback
            return create_synthetic_flickr30k(processed_path)

    # Process the dataset (simplified for this example)
    print(f"Processing {dataset_name} dataset...")
    os.makedirs(processed_path, exist_ok=True)

    try:
        # Check if we have access to the images
        if not os.path.exists(os.path.join(raw_data_path, "flickr30k-images")) or \
                len(os.listdir(os.path.join(raw_data_path, "flickr30k-images"))) < 100:
            print("Raw data not found or incomplete. Creating synthetic dataset...")
            return create_synthetic_flickr30k(processed_path)

        # In a real implementation, we would:
        # 1. Process the images using a vision model (e.g., CLIP, ResNet)
        # 2. Process the captions using a text model (e.g., BERT)
        # 3. Create embeddings for both modalities

        # For this example, we'll just create synthetic data
        return create_synthetic_flickr30k(processed_path)

    except Exception as e:
        print(f"Error processing dataset: {e}")
        print("Creating synthetic dataset as fallback...")
        return create_synthetic_flickr30k(processed_path)

def create_synthetic_flickr30k(processed_path):
    """Create a synthetic version of Flickr30k dataset."""
    os.makedirs(processed_path, exist_ok=True)

    embeddings_file = os.path.join(processed_path, "embeddings.npy")
    metadata_file = os.path.join(processed_path, "metadata.json")

    # Create synthetic multi-modal data with image and text modalities
    embedding_dim = 512  # Typical CLIP embedding dimension
    n_images = 500
    n_captions = 5 * n_images  # 5 captions per image
    n_queries = 100  # Text queries

    # Image embeddings (modality 0)
    image_embeddings = np.random.normal(0, 1, (n_images, embedding_dim))
    image_embeddings = image_embeddings / np.linalg.norm(image_embeddings, axis=1, keepdims=True)

    # Caption embeddings (modality 1)
    caption_embeddings = np.random.normal(0, 1, (n_captions, embedding_dim))

    # Make captions similar to their corresponding images
    for i in range(n_images):
        image_embedding = image_embeddings[i]
        for j in range(5):
            caption_idx = i * 5 + j
            caption_embeddings[caption_idx] = 0.7 * image_embedding + 0.3 * caption_embeddings[caption_idx]

    caption_embeddings = caption_embeddings / np.linalg.norm(caption_embeddings, axis=1, keepdims=True)

    # Combine embeddings
    embeddings = np.vstack([image_embeddings, caption_embeddings])

    # Create modality information
    modalities = {}
    for i in range(n_images):
        modalities[i] = 0  # Image modality
    for i in range(n_captions):
        modalities[n_images + i] = 1  # Text modality

    # Create text queries (similar to captions)
    queries = np.random.normal(0, 1, (n_queries, embedding_dim))

    # Make some queries similar to specific images
    for i in range(n_queries):
        image_idx = i % n_images
        queries[i] = 0.6 * image_embeddings[image_idx] + 0.4 * queries[i]

    queries = queries / np.linalg.norm(queries, axis=1, keepdims=True)

    # Create ground truth (linking queries to relevant images)
    ground_truth = []
    for i in range(n_queries):
        # Each query is similar to a random image
        image_idx = i % n_images

        # The ground truth is the image and its captions
        relevant_items = [image_idx]  # The image itself
        caption_indices = [n_images + image_idx * 5 + j for j in range(5)]
        relevant_items.extend(caption_indices)

        ground_truth.append(relevant_items)

    # Save processed data
    np.save(embeddings_file, embeddings)

    metadata = {
        "queries": queries.tolist(),
        "ground_truth": ground_truth,
        "modalities": modalities,
        "dimensions": embedding_dim,
        "num_images": n_images,
        "num_captions": n_captions,
        "num_queries": n_queries,
        "description": "Flickr30k dataset (synthetic version)"
    }

    with open(metadata_file, 'w') as f:
        json.dump(metadata, f)

    print(f"Synthetic Flickr30k dataset saved to {processed_path}")

    return {
        "embeddings": embeddings,
        "queries": queries,
        "ground_truth": ground_truth,
        "modalities": modalities,
        "metadata": metadata
    }

def load_amazon_product_dataset(data_dir):
    """
    Load Amazon Product dataset.
    """
    dataset_name = "amazon-product"
    dataset_path = os.path.join(data_dir, dataset_name)
    processed_path = os.path.join(dataset_path, "processed")
    embeddings_file = os.path.join(processed_path, "embeddings.npy")
    metadata_file = os.path.join(processed_path, "metadata.json")

    # Check if processed data exists
    if os.path.exists(embeddings_file) and os.path.exists(metadata_file):
        print(f"Loading processed {dataset_name} dataset from {processed_path}...")
        embeddings = np.load(embeddings_file)

        with open(metadata_file, 'r') as f:
            metadata = json.load(f)

        return {
            "embeddings": embeddings,
            "queries": np.array(metadata["queries"]),
            "ground_truth": metadata["ground_truth"],
            "modalities": {int(k): v for k, v in metadata["modalities"].items()},
            "metadata": metadata
        }

    # If processed data doesn't exist, check if raw data exists
    raw_data_path = os.path.join(dataset_path, "raw")
    os.makedirs(raw_data_path, exist_ok=True)

    # Define file paths
    archive_file = os.path.join(raw_data_path, "amazon_5core.json.gz")

    # Download if not exists
    if not os.path.exists(archive_file):
        print(f"Downloading {dataset_name} dataset...")
        try:
            download_with_progress(DATASET_REGISTRY[dataset_name]["url"], archive_file)
            extract_archive(archive_file, raw_data_path)
        except Exception as e:
            print(f"Error downloading Amazon dataset: {e}")
            print("This dataset may require manual download. Please visit:")
            print("https://jmcauley.ucsd.edu/data/amazon/")
            print("After downloading, place the archive in:", raw_data_path)

            # Create synthetic dataset as fallback
            return create_synthetic_amazon_product(processed_path)

    # Process the dataset (simplified for this example)
    print(f"Processing {dataset_name} dataset...")
    os.makedirs(processed_path, exist_ok=True)

    try:
        # In a real implementation, we would process the actual Amazon data
        # For this example, we'll just create synthetic data
        return create_synthetic_amazon_product(processed_path)

    except Exception as e:
        print(f"Error processing dataset: {e}")
        print("Creating synthetic dataset as fallback...")
        return create_synthetic_amazon_product(processed_path)

def create_synthetic_amazon_product(processed_path):
    """Create a synthetic version of Amazon Product dataset."""
    os.makedirs(processed_path, exist_ok=True)

    embeddings_file = os.path.join(processed_path, "embeddings.npy")
    metadata_file = os.path.join(processed_path, "metadata.json")

    # Create synthetic multi-modal e-commerce data
    embedding_dim = 384  # Typical embedding size for e-commerce data
    n_products = 800
    n_queries = 150
    n_modalities = 3  # Title (0), description (1), image (2)

    # Create embeddings for each modality
    embeddings_list = []
    modalities = {}

    current_idx = 0

    # Title embeddings
    title_embeddings = np.random.normal(0, 1, (n_products, embedding_dim))
    title_embeddings = title_embeddings / np.linalg.norm(title_embeddings, axis=1, keepdims=True)
    embeddings_list.append(title_embeddings)

    for i in range(n_products):
        modalities[current_idx + i] = 0  # Title modality

    current_idx += n_products

    # Description embeddings (longer text)
    desc_embeddings = np.random.normal(0, 1, (n_products, embedding_dim))

    # Make description embeddings somewhat similar to title embeddings
    for i in range(n_products):
        desc_embeddings[i] = 0.7 * title_embeddings[i] + 0.3 * desc_embeddings[i]

    desc_embeddings = desc_embeddings / np.linalg.norm(desc_embeddings, axis=1, keepdims=True)
    embeddings_list.append(desc_embeddings)

    for i in range(n_products):
        modalities[current_idx + i] = 1  # Description modality

    current_idx += n_products

    # Image embeddings
    image_embeddings = np.random.normal(0, 1, (n_products, embedding_dim))

    # Make image embeddings somewhat similar to title embeddings
    for i in range(n_products):
        image_embeddings[i] = 0.5 * title_embeddings[i] + 0.5 * image_embeddings[i]

    image_embeddings = image_embeddings / np.linalg.norm(image_embeddings, axis=1, keepdims=True)
    embeddings_list.append(image_embeddings)

    for i in range(n_products):
        modalities[current_idx + i] = 2  # Image modality

    # Combine all embeddings
    embeddings = np.vstack(embeddings_list)

    # Create queries (search terms)
    queries = np.random.normal(0, 1, (n_queries, embedding_dim))

    # Make some queries similar to specific product titles
    for i in range(n_queries):
        product_idx = i % n_products
        queries[i] = 0.8 * title_embeddings[product_idx] + 0.2 * queries[i]

    queries = queries / np.linalg.norm(queries, axis=1, keepdims=True)

    # Create ground truth (relevant products for each query)
    ground_truth = []
    for i in range(n_queries):
        # Each query corresponds to a random subset of products
        product_idx = i % n_products

        # Find similar products
        similarities = 1 - cdist([title_embeddings[product_idx]], title_embeddings, metric='cosine')[0]
        top_indices = np.argsort(-similarities)[:5]  # Top 5 similar products

        # For each product, add all its modalities
        relevant_items = []
        for p_idx in top_indices:
            relevant_items.append(p_idx)  # Title embedding
            relevant_items.append(n_products + p_idx)  # Description embedding
            relevant_items.append(2 * n_products + p_idx)  # Image embedding

        ground_truth.append(relevant_items)

    # Save processed data
    np.save(embeddings_file, embeddings)

    metadata = {
        "queries": queries.tolist(),
        "ground_truth": ground_truth,
        "modalities": modalities,
        "dimensions": embedding_dim,
        "num_products": n_products,
        "num_queries": n_queries,
        "num_modalities": n_modalities,
        "description": "Amazon Product dataset (synthetic version)"
    }

    with open(metadata_file, 'w') as f:
        json.dump(metadata, f)

    print(f"Synthetic Amazon Product dataset saved to {processed_path}")

    return {
        "embeddings": embeddings,
        "queries": queries,
        "ground_truth": ground_truth,
        "modalities": modalities,
        "metadata": metadata
    }

def list_available_datasets():
    """List all available datasets with descriptions."""
    print("\nAvailable Datasets:")
    print("-" * 80)
    print(f"{'Name':<20} | {'Size (GB)':<10} | {'Description'}")
    print("-" * 80)

    for name, info in DATASET_REGISTRY.items():
        requires_login = info.get('requires_login', False)
        login_info = " (requires login)" if requires_login else ""
        print(f"{name:<20} | {info['size_gb']:<10.1f} | {info['description']}{login_info}")

def load_multimodal_datasets(dataset_names=None, data_dir="./data"):
    """
    Load specified multi-modal datasets.

    Parameters:
    -----------
    dataset_names : list or None
        List of dataset names to load, or None to load synthetic dataset
    data_dir : str
        Directory to store datasets

    Returns:
    --------
    dict
        Dictionary of loaded datasets
    """
    # Create data directory if it doesn't exist
    os.makedirs(data_dir, exist_ok=True)

    if dataset_names is None:
        dataset_names = ["synthetic"]  # Default to synthetic data

    datasets = {}

    for name in dataset_names:
        if name not in DATASET_REGISTRY:
            print(f"Warning: Dataset '{name}' not found in registry. Skipping.")
            continue

        print(f"Loading {name} dataset...")

        try:
            if name == "synthetic":
                datasets[name] = load_synthetic_dataset(data_dir)
            elif name == "msmarco":
                datasets[name] = load_msmarco_dataset(data_dir)
            elif name == "flickr30k":
                datasets[name] = load_flickr30k_dataset(data_dir)
            elif name == "amazon-product":
                datasets[name] = load_amazon_product_dataset(data_dir)
            elif name == "spotify-playlists":
                print("Spotify dataset requires manual download. Please download from:")
                print("https://www.aicrowd.com/challenges/spotify-million-playlist-dataset-challenge")
                print("and place the data in:", os.path.join(data_dir, name, "raw"))
                continue
            else:
                print(f"No loader implemented for {name} dataset")
                continue

            print(f"Successfully loaded {name} dataset with {len(datasets[name]['embeddings'])} embeddings")

        except Exception as e:
            print(f"Error loading {name} dataset: {str(e)}")
            continue

    return datasets

if __name__ == "__main__":
    # Example usage
    list_available_datasets()

    print("\nLoading datasets...")
    datasets = load_multimodal_datasets(["synthetic", "flickr30k"])

    for name, dataset in datasets.items():
        print(f"\nDataset: {name}")
        print(f"Embeddings shape: {dataset['embeddings'].shape}")
        print(f"Number of queries: {len(dataset['queries'])}")
        print(f"Number of modalities: {len(set(dataset['modalities'].values()))}")