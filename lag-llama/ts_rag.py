import numpy as np
import faiss
import torch
from typing import List, Tuple, Dict
from scipy.stats import skew, kurtosis

import numpy as np
from scipy.stats import skew, kurtosis
import torch
from gluonts.model.forecast import SampleForecast

class TimeSeriesRAG:
    def __init__(
        self,
        embedding_dim: int = 8,
        num_neighbors: int = 5,
        similarity_threshold: float = 0.8
    ):
        """Initialize RAG for time series forecasting"""
        self.embedding_dim = embedding_dim
        self.num_neighbors = num_neighbors
        self.similarity_threshold = similarity_threshold
        
        # Initialize FAISS index for similarity search
        self.index = faiss.IndexFlatL2(embedding_dim)
        
        # Storage for embeddings and metadata
        self.embeddings = []
        self.metadata = []
    
    def encode_time_series(self, series: torch.Tensor) -> np.ndarray:
        """Convert time series to embedding vector"""
        x = series if isinstance(series, np.ndarray) else series.numpy()
        x = x.flatten()  # Ensure 1D array

        # Check if data contains NaN or empty values
        if np.any(np.isnan(x)) or len(x) == 0:
            print("Warning: Time series contains NaN values or is empty.")
            return np.zeros(self.embedding_dim)

        features = []

        # Basic statistics with normalization
        features.extend([
            np.mean(x),
            np.std(x),
            (np.min(x) - np.mean(x)) / (np.std(x) + 1e-8),
            (np.max(x) - np.mean(x)) / (np.std(x) + 1e-8),
            np.percentile(x, 25),
            np.percentile(x, 50),
            np.percentile(x, 75)
        ])

        # Normalized differences
        if len(x) > 1:
            diff = np.diff(x)
            features.extend([
                np.mean(diff),
                np.std(diff) / (np.std(x) + 1e-8)
            ])

            # Normalized skewness and kurtosis
            if len(diff) > 2:
                features.extend([
                    skew(diff) / (1 + abs(skew(diff))),
                    kurtosis(diff) / (1 + abs(kurtosis(diff)))
                ])

        # Recent patterns
        window_sizes = [5, 10, 20]
        for w in window_sizes:
            if len(x) >= w:
                window = x[-w:]
                features.extend([
                    np.mean(window) / (np.std(x) + 1e-8),
                    np.std(window) / (np.std(x) + 1e-8)
                ])

        # Convert to numpy array and normalize
        features = np.array(features, dtype=np.float32)

        # Normalize entire feature vector
        feature_std = np.std(features)
        if feature_std > 0:
            features = (features - np.mean(features)) / feature_std

        # Pad or truncate
        if len(features) < self.embedding_dim:
            features = np.pad(features, (0, self.embedding_dim - len(features)))
        else:
            features = features[:self.embedding_dim]

        return features
    # def encode_time_series(self, series: torch.Tensor) -> np.ndarray:
        # x = series if isinstance(series, np.ndarray) else series.numpy()
        # print(np.isnan(x))

        # # Check if data contains NaN or empty values
        # if np.any(np.isnan(x)) or len(x) == 0:
        #     print("Warning: Time series contains NaN values or is empty.")
        #     return np.zeros(self.embedding_dim)  # Return a zero vector if data is invalid

        # features = []

        # # Basic statistics: mean, std, min, max, percentiles
        # features.extend([np.mean(x), np.std(x), np.min(x), np.max(x), np.percentile(x, 25), np.percentile(x, 50), np.percentile(x, 75)])

        # # Calculate difference (diff) as a feature to capture trend
        # diff = np.diff(x)
        # features.extend([np.mean(diff), np.std(diff), np.sum(diff > 0) / len(diff)])

        # # Adding more features, for example: skew and kurtosis
        # features.append(skew(x))  # Adding skewness
        # features.append(kurtosis(x))  # Adding kurtosis

        # features = [f if not np.isnan(f) else 0.0 for f in features]

        # features = np.array(features)

        # if features.ndim > 1:
        #     features = features.flatten()

        # # Ensure that features length is consistent with embedding_dim
        # if len(features) < self.embedding_dim:
        #     features = np.pad(features, (0, self.embedding_dim - len(features)), mode='constant')
        # elif len(features) > self.embedding_dim:
        #     features = features[:self.embedding_dim]

        # return features.astype(np.float32)
        
        # Skewness and Kurtosis to capture the distribution of the values
        # if len(diff) > 1:  # Ensure there is enough data to calculate skew/kurtosis
        #     features.extend([skew(diff), kurtosis(diff)])
        
        # # Momentum: Percentage change in values over a period
        # for period in [5, 10, 20]:
        #     if len(x) >= period:
        #         momentum = x[-1] / x[-period] - 1
        #         features.append(momentum)

        # # Volatility: Standard deviation of recent values
        # for period in [5, 10, 20]:
        #     if len(x) >= period:
        #         vol = np.std(x[-period:])
        #         features.append(vol)

        # # Pad or truncate to embedding_dim
        # features = np.array(features).flatten()
        
        # # Handle the case when features length is less than embedding_dim
        # if len(features) < self.embedding_dim:
        #     features = np.pad(features, (0, self.embedding_dim - len(features)))
        # else:
        #     features = features[:self.embedding_dim]
        
        # return features.astype(np.float32)

    def add_to_index(self, series: torch.Tensor, metadata: dict):
        """Add a time series and its metadata to the index"""
        embedding = self.encode_time_series(series)
        self.index.add(embedding.reshape(1, -1))
        self.embeddings.append(embedding)
        self.metadata.append(metadata)
        
    def retrieve_similar(self, query_series: torch.Tensor) -> list:
        """Retrieve similar time series and their metadata"""
        print(f"Current index size: {self.index.ntotal}")
        if self.index.ntotal == 0:
            print("Warning: Index is empty, no similar sequences to retrieve")
            return []
        
        query_embedding = self.encode_time_series(query_series)
        print("Query embedding shape:", query_embedding.shape)
        print("Query embedding dtype:", query_embedding.dtype)
        print("Query embedding sample:", query_embedding)

        query_embedding = query_embedding.astype(np.float32)
        if len(query_embedding.shape) == 1:
            query_embedding = query_embedding.reshape(1, -1)
        
        # Search nearest neighbors
        try:
            k = min(3, self.index.ntotal)  # Start with just 3 neighbors
            print(f"Searching for {k} neighbors")
            distances, indices = self.index.search(query_embedding, k)

            print("Raw distances:", distances)
            print("Raw indices:", indices)

            # Check for valid results
            results = []
            for dist, idx in zip(distances[0], indices[0]):
                if idx != -1 and dist != float('inf') and dist != 3.4028235e+38:
                    similarity = 1 / (1 + dist)
                    print(f"Distance: {dist}, Similarity: {similarity}")
                    if similarity >= self.similarity_threshold:
                        results.append((self.metadata[idx], similarity))

            return results
        
        except Exception as e:
            print(f"Error during search: {str(e)}")
            print(f"Query embedding stats - min: {query_embedding.min()}, max: {query_embedding.max()}")
            return []
    
    def augment_prediction(self, base_forecast: SampleForecast, similar_sequences: list) -> SampleForecast:
        """Augment base prediction using retrieved similar sequences"""
        if not similar_sequences:
            return base_forecast
            
        # Get base samples
        base_samples = base_forecast.samples  # Shape: (num_samples x prediction_length)
            
        # Weight predictions by similarity
        total_weight = 0
        weighted_sum = np.zeros_like(base_samples)
        
        for metadata, similarity in similar_sequences:
            # Get historical prediction for similar sequence
            hist_pred = metadata['historical_prediction']
            
            # Weight by similarity score
            weight = similarity
            weighted_sum += hist_pred * weight
            total_weight += weight
            
        # Combine with base prediction
        if total_weight > 0:
            weighted_avg = weighted_sum / total_weight
            # Blend predictions (0.7 for base prediction, 0.3 for retrieved predictions)
            augmented_samples = 0.7 * base_samples + 0.3 * weighted_avg
            
            # Create new forecast object
            augmented_forecast = SampleForecast(
                samples=augmented_samples,
                start_date=base_forecast.start_date,
                item_id=base_forecast.item_id,
                info=base_forecast.info
            )
            return augmented_forecast
            
        return base_forecast