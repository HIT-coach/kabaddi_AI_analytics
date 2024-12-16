from typing import Generator, Iterable, List, TypeVar

import numpy as np
import supervision as sv
import torch
import umap
from sklearn.cluster import KMeans
from tqdm import tqdm
from transformers import AutoProcessor, SiglipVisionModel

V = TypeVar("V")

SIGLIP_MODEL_PATH = 'google/siglip-base-patch16-224'


def create_batches(
    sequence: Iterable[V], batch_size: int
) -> Generator[List[V], None, None]:
    """
    Generate batches from a sequence with a specified batch size.

    Args:
        sequence (Iterable[V]): The input sequence to be batched.
        batch_size (int): The size of each batch.

    Yields:
        Generator[List[V], None, None]: A generator yielding batches of the input
            sequence.
    """
    batch_size = max(batch_size, 1)
    current_batch = []
    for element in sequence:
        if len(current_batch) == batch_size:
            yield current_batch
            current_batch = []
        current_batch.append(element)
    if current_batch:
        yield current_batch


class TeamClassifier:
    """
    A classifier that uses a pre-trained SiglipVisionModel for feature extraction,
    UMAP for dimensionality reduction, and KMeans for clustering.
    """
    def __init__(self, device: str = 'cpu', batch_size: int = 32):
        """
       Initialize the TeamClassifier with device and batch size.

       Args:
           device (str): The device to run the model on ('cpu' or 'cuda').
           batch_size (int): The batch size for processing images.
       """
        self.device = device
        self.batch_size = batch_size
        self.features_model = SiglipVisionModel.from_pretrained(
            SIGLIP_MODEL_PATH).to(device)
        self.processor = AutoProcessor.from_pretrained(SIGLIP_MODEL_PATH)
        self.reducer = umap.UMAP(n_components=3)
        self.cluster_model = KMeans(n_clusters=2)

    def extract_features(self, crops: List[np.ndarray]) -> np.ndarray:
        """
        Extract features from a list of image crops using the pre-trained
            SiglipVisionModel.

        Args:
            crops (List[np.ndarray]): List of image crops.

        Returns:
            np.ndarray: Extracted features as a numpy array.
        """
        crops = [sv.cv2_to_pillow(crop) for crop in crops]
        batches = create_batches(crops, self.batch_size)
        data = []
        with torch.no_grad():
            for batch in tqdm(batches, desc='Embedding extraction'):
                inputs = self.processor(
                    images=batch, return_tensors="pt").to(self.device)
                outputs = self.features_model(**inputs)
                embeddings = torch.mean(outputs.last_hidden_state, dim=1).cpu().numpy()
                data.append(embeddings)

        return np.concatenate(data)

    def fit(self, crops: List[np.ndarray]) -> None:
        """
        Fit the classifier model on a list of image crops.

        Args:
            crops (List[np.ndarray]): List of image crops.
        """
        data = self.extract_features(crops)
        projections = self.reducer.fit_transform(data)
        self.cluster_model.fit(projections)

    def predict(self, crops: List[np.ndarray]) -> np.ndarray:
        """
        Predict the team roles for each player crop in a Kabaddi game.

        Args:
            crops (List[np.ndarray]): List of image crops.

        Returns:
            np.ndarray: Predicted team roles (0 = defenders, 1 = attacker).
        """
        if len(crops) == 0:
            return np.array([])

        data = self.extract_features(crops)

        # Transform features using UMAP
        projections = self.reducer.transform(data)

        # Custom heuristic: Use distance to centroid for role assignment
        # Calculate distances from cluster centroids
        distances = self.cluster_model.transform(projections)

        # Assign closest centroid as team
        team_assignments = np.argmin(distances, axis=1)

        # Kabaddi-specific rule: If one team has a single player, mark them as attacker
        unique_teams, counts = np.unique(team_assignments, return_counts=True)
        if len(counts) == 2 and 1 in counts:
            attacker_team = unique_teams[np.argmin(counts)]
            corrected_roles = [
                attacker_team if team == attacker_team else 1 - attacker_team
                for team in team_assignments
            ]
            return np.array(corrected_roles)

        return team_assignments


class TeamClassifierMultiFrame:
    def __init__(self, buffer_size: int = 5):
        """
        Initialize the multi-frame team classifier.

        Args:
            buffer_size (int): Number of frames to aggregate for classification.
        """
        self.buffer_size = buffer_size
        self.frame_buffer = []
        self.team_id_buffer = []

    def update_buffer(self, frame, detections, team_ids):
        """
        Update the buffer with new frame and team IDs.

        Args:
            frame (np.ndarray): Current frame.
            detections (sv.Detections): Detections in the current frame.
            team_ids (np.ndarray): Team IDs predicted for the current frame.
        """
        if len(self.frame_buffer) >= self.buffer_size:
            self.frame_buffer.pop(0)
            self.team_id_buffer.pop(0)

        self.frame_buffer.append((frame, detections))
        self.team_id_buffer.append(team_ids)

    def compute_consensus(self) -> np.ndarray:
        """
        Compute team classification consensus over buffered frames.

        Returns:
            np.ndarray: Final team IDs based on multi-frame consensus.
        """
        # Combine all team ID predictions
        combined_team_ids = np.concatenate(self.team_id_buffer)

        # Compute majority vote for each player
        unique_ids, counts = np.unique(combined_team_ids, return_counts=True)
        return unique_ids[np.argmax(counts)]
