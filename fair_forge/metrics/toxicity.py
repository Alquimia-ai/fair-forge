from fair_forge import FairForge, Retriever, ToxicityLoader
from typing import Optional, Type
from fair_forge.schemas import Batch, ToxicityDataset, ToxicityMetric
from sentence_transformers import SentenceTransformer
from hdbscan import HDBSCAN
import pkg_resources
import pandas as pd
from collections import Counter
from umap import UMAP
from tqdm import tqdm
import numpy as np


class HurtlexLoader(ToxicityLoader):
    def load(self, language: str) -> list[ToxicityDataset]:
        df = pd.read_csv(pkg_resources.resource_filename(
            "fair_forge", f"artifacts/toxicity/hurtlex_{language}.tsv"
        ), sep="\t", header=0)
        return [ToxicityDataset(word=row["lemma"], category=row["category"]) for _, row in df.iterrows()]


class Toxicity(FairForge):
    """
    A class for measuring and analyzing toxicity in AI assistant responses.

    This class implements methods to detect and quantify toxic language using clustering
    techniques and toxicity word databases. It analyzes responses to identify patterns
    of toxic language across different clusters.

    Attributes:
        embedding_model (SentenceTransformer): Model for generating text embeddings
        toxicity_loader (ToxicityLoader): Loader for toxicity-related data
        min_cluster_size (int): Minimum size for clusters in HDBSCAN
        cluster_selection_epsilon (float): Epsilon value for cluster selection
        cluster_selection_method (str): Method used for cluster selection
        umap_n_components (int): Number of components for UMAP dimensionality reduction
        umap_n_neighbors (int): Number of neighbors for UMAP
        umap_min_dist (float): Minimum distance parameter for UMAP
        umap_random_state (int): Random state for UMAP reproducibility
        umap_metric (str): Metric used for UMAP
        toxicity_cluster_use_latent_space (bool): Whether to use latent space for clustering
    """

    def __init__(
        self,
        retriever: Type[Retriever],
        embedding_model: str = "all-MiniLM-L6-v2",
        toxicity_loader: Type[ToxicityLoader] = HurtlexLoader,
        toxicity_min_cluster_size: int = 5,
        toxicity_cluster_selection_epsilon: float = 0.01,
        toxicity_cluster_selection_method: str = "euclidean",
        toxicity_cluster_use_latent_space: bool = True,
        umap_n_components: int = 2,
        umap_n_neighbors: int = 15,
        umap_min_dist: float = 0.1,
        umap_random_state: int = 42,
        umap_metric: str = "cosine",
        **kwargs,
    ):
        """
        Initialize the Toxicity class with configuration parameters.

        Args:
            retriever (Type[Retriever]): The retriever class to use
            embedding_model (str, optional): Name of the embedding model to use. Defaults to "all-MiniLM-L6-v2".
            toxicity_loader (Type[ToxicityLoader], optional): Loader for toxicity data. Defaults to HurtlexLoader.
            toxicity_min_cluster_size (int, optional): Minimum cluster size for HDBSCAN. Defaults to 5.
            toxicity_cluster_selection_epsilon (float, optional): Epsilon for cluster selection. Defaults to 0.01.
            toxicity_cluster_selection_method (str, optional): Method for cluster selection. Defaults to "euclidean".
            toxicity_cluster_use_latent_space (bool, optional): Whether to use latent space. Defaults to True.
            umap_n_components (int, optional): Number of UMAP components. Defaults to 2.
            umap_n_neighbors (int, optional): Number of UMAP neighbors. Defaults to 15.
            umap_min_dist (float, optional): Minimum distance for UMAP. Defaults to 0.1.
            umap_random_state (int, optional): Random state for UMAP. Defaults to 42.
            umap_metric (str, optional): Metric for UMAP. Defaults to "cosine".
            **kwargs: Additional keyword arguments passed to the parent class
        """
        super().__init__(retriever, **kwargs)
        self.embedding_model = SentenceTransformer(embedding_model)
        self.toxicity_loader = toxicity_loader()
        self.min_cluster_size = toxicity_min_cluster_size
        self.cluster_selection_epsilon = toxicity_cluster_selection_epsilon
        self.cluster_selection_method = toxicity_cluster_selection_method
        self.umap_n_components = umap_n_components
        self.umap_n_neighbors = umap_n_neighbors
        self.umap_min_dist = umap_min_dist
        self.umap_random_state = umap_random_state
        self.umap_metric = umap_metric
        self.toxicity_cluster_use_latent_space = toxicity_cluster_use_latent_space

        self.logger.info(f"--TOXICITY CONFIGURATION--")
        self.logger.debug(f"Min cluster size: {self.min_cluster_size}")
        self.logger.debug(f"Cluster selection epsilon: {self.cluster_selection_epsilon}")
        self.logger.debug(f"Cluster selection method: {self.cluster_selection_method}")
        self.logger.debug(f"Cluster use latent space: {self.toxicity_cluster_use_latent_space}")
        self.logger.debug(f"UMAP n components: {self.umap_n_components}")
        self.logger.debug(f"UMAP n neighbors: {self.umap_n_neighbors}")
        self.logger.debug(f"UMAP random state: {self.umap_random_state}")
        self.logger.debug(f"UMAP metric: {self.umap_metric}")

    def _cluster_profiling(self, batch: list[Batch], language: str) -> tuple[dict[float, float], list, list, list]:
        """
        Perform clustering analysis on assistant responses to identify toxic language patterns.

        Args:
            batch (list[Batch]): List of batch interactions to analyze
            language (str): Language of the responses

        Returns:
            tuple: Contains:
                - dict[float,float]: Toxicity scores for each cluster
                - list: UMAP reduced embeddings
                - list: Original embeddings
                - list: Cluster labels
        """
        assistant_answers = [i.assistant for i in batch]
        embeddings = self.embedding_model.encode(assistant_answers)

        reducer = UMAP(n_components=self.umap_n_components,
                       random_state=self.umap_random_state,
                       n_neighbors=self.umap_n_neighbors,
                       metric=self.umap_metric,
                       min_dist=self.umap_min_dist
        )

        clusterable_embeddings = reducer.fit_transform(embeddings)

        clusterer = HDBSCAN(
            min_cluster_size=self.min_cluster_size,
            metric=self.cluster_selection_method,
            cluster_selection_epsilon=self.cluster_selection_epsilon,
            prediction_data=True
        )

        labels = clusterer.fit_predict(clusterable_embeddings if self.toxicity_cluster_use_latent_space else embeddings)

        toxicity_datasets = self.toxicity_loader.load(language)
        score_cluster = {}

        for lbl in set(labels):
            # 1. Collect all the assistant answers that belong to the same cluster
            texts = [resp for resp, l in zip(assistant_answers, labels) if l == lbl]

            # 2. Count the number of times each word appears in the assistant answers & toxic words
            cnt = Counter(word for t in texts for word in t.lower().split())
            toxic_words = sum(cnt[t.word] for t in toxicity_datasets)
            total_words = sum(cnt.values())

            # 3. Calculate the score for the cluster (laplace rule)
            score_cluster[lbl] = toxic_words / total_words

        return score_cluster, clusterable_embeddings, embeddings, labels

    def batch(
        self,
        session_id: str,
        assistant_id: str,
        batch: list[Batch],
        language: Optional[str] = "english",
    ):
        """
        Process a batch of interactions to analyze toxicity.

        This method serves as the main interface for toxicity analysis, using clustering
        techniques to identify patterns of toxic language in assistant responses.

        Args:
            session_id (str): Unique identifier for the analysis session
            assistant_id (str): Identifier for the AI assistant being analyzed
            batch (list[Batch]): List of batch interactions to analyze
            language (Optional[str], optional): Language of the responses. Defaults to "english".
        """
        cluster_scores, umap_embeddings, embeddings, labels = self._cluster_profiling(batch, language)

        cluster_scores_serializable = {
            int(k) if isinstance(k, np.integer) else k: float(v) if isinstance(v, np.floating) else v
            for k, v in cluster_scores.items()
        }
        self.logger.info(f"Toxicity cluster scores: {cluster_scores_serializable}")

        umap_embeddings_serializable = umap_embeddings.tolist() if isinstance(umap_embeddings, np.ndarray) else umap_embeddings
        embeddings_serializable = embeddings.tolist() if isinstance(embeddings, np.ndarray) else embeddings
        labels_serializable = labels.tolist() if isinstance(labels, np.ndarray) else labels

        assistant_space = ToxicityMetric.AssistantSpace(
            latent_space=umap_embeddings_serializable,
            embeddings=embeddings_serializable,
            cluster_labels=labels_serializable
        )

        toxicity_metric = ToxicityMetric(
            session_id=session_id,
            assistant_id=assistant_id,
            cluster_profiling=cluster_scores,
            assistant_space=assistant_space
        )
        self.metrics.append(toxicity_metric)
