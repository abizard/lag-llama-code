import torch
import numpy as np
from gluonts.model.forecast import SampleForecast
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances

# def find_best_matches_full_series_batch(
#     dataset,
#     context_tensor_matrix,
#     length,
#     prediction_length,
#     top_n,
# ):
#     best_matches = []

#     # Menggabungkan semua time series dalam train_df menjadi satu array
#     train_series = np.concatenate([entry["target"] for entry in dataset])

#     for context_tensor in context_tensor_matrix:
#         context = context_tensor[-length:].numpy()

#         # Inisialisasi variabel untuk menyimpan jarak dan segmen terbaik
#         distances = []
#         segments = []

#         # Iterasi melalui train_series untuk menemukan segmen yang cocok
#         for i in range(len(train_series) - length - prediction_length + 1):
#             train_segment = train_series[i : i + length]
#             distance = np.linalg.norm(context - train_segment)
#             distances.append(distance)
#             segments.append(train_series[i + length : i + length + prediction_length])

#         # Mengurutkan segmen berdasarkan jarak terpendek
#         sorted_indices = np.argsort(distances)[:top_n]
#         best_segments = [segments[idx] for idx in sorted_indices]
#         best_matches.extend(best_segments)

#     return best_matches

def find_best_matches_full_series_batch(
    dataset,
    context_tensor_matrix,
    test_length,
    prediction_length,
    top_n,
):
    best_matches = []

    # Menggabungkan semua time series dalam dataset menjadi satu array
    train_series = np.concatenate([entry["target"] for entry in dataset])

    for context_tensor in context_tensor_matrix:
        context = context_tensor[-test_length:].numpy().reshape(1, -1)

        # Inisialisasi variabel untuk menyimpan jarak dan segmen terbaik
        distances = []
        segments = []

        # Iterasi melalui train_series untuk menemukan segmen yang cocok
        for i in range(len(train_series) - test_length - prediction_length + 1):
            train_segment = train_series[i : i + test_length].reshape(1, -1)
            distance = euclidean_distances(context, train_segment)[0][0]
            distances.append(distance)
            segments.append(train_series[i + test_length : i + test_length + prediction_length])

        # Mengurutkan segmen berdasarkan jarak terpendek
        sorted_indices = np.argsort(distances)[:top_n]
        best_segments = [segments[idx] for idx in sorted_indices]
        best_matches.extend(best_segments)

    return best_matches

def augment_time_series(dataset, context_tensor_matrix, prediction_length, top_n):
    """
    Fungsi ini melakukan augmentasi pada time series dengan menambahkan segmen terbaik dari data latih ke konteks.
    """
    test_length = len(context_tensor_matrix[0])
    best_matches = find_best_matches_full_series_batch(
        dataset, context_tensor_matrix, test_length, prediction_length, top_n
    )

    augmented_matrix = []
    mean_std_values = []

    for idx, context_tensor in enumerate(context_tensor_matrix):
        context_tensor = torch.tensor(context_tensor, dtype=torch.float32)
        elements = best_matches[idx * top_n : (idx + 1) * top_n]
        avg_best_segment = np.mean(elements, axis=0)
        avg_segment_tensor = torch.tensor(avg_best_segment, dtype=torch.float32)

        # Normalisasi segmen terbaik
        mask = ~torch.isnan(avg_segment_tensor)
        avg_mean = avg_segment_tensor[mask].mean()
        avg_std = torch.sqrt(((avg_segment_tensor[mask] - avg_mean) ** 2).mean()) + 1e-7
        avg_segment_tensor = normalize(avg_segment_tensor, avg_mean, avg_std)

        # Normalisasi konteks
        mask = ~torch.isnan(context_tensor)
        context_mean = context_tensor[mask].mean()
        context_std = torch.sqrt(((context_tensor[mask] - context_mean) ** 2).mean()) + 1e-7
        context_tensor = normalize(context_tensor, context_mean, context_std)

        # Penyesuaian perbedaan antara awal konteks dan akhir segmen terbaik
        context_start = context_tensor[0] if not torch.isnan(context_tensor[0]) else 0
        best_segment_start = avg_segment_tensor[-1]
        difference = context_start - best_segment_start
        avg_segment_tensor += difference

        # Penggabungan segmen terbaik dengan konteks
        augmented_tensor = torch.cat((avg_segment_tensor, context_tensor))
        augmented_matrix.append(augmented_tensor)
        mean_std_values.append((context_mean, context_std))

    return augmented_matrix, mean_std_values

def normalize_context(context_tensor_matrix):
    """
    Fungsi ini melakukan normalisasi pada setiap tensor dalam context_tensor_matrix.
    """
    normalized_context = []
    mean_std_values = []

    for idx, context_tensor in enumerate(context_tensor_matrix):
        context_tensor = torch.tensor(context_tensor, dtype=torch.float32)

        mask = ~torch.isnan(context_tensor)
        context_mean = context_tensor[mask].mean()
        context_std = torch.sqrt(((context_tensor[mask] - context_mean) ** 2).mean()) + 1e-7
        normalized_tensor = normalize(context_tensor, context_mean, context_std) 

        normalized_context.append(normalized_tensor)
        mean_std_values.append((context_mean, context_std))

    return normalized_context, mean_std_values

def denormalize_predictions(predictions, mean_std_values):
    """
    Fungsi ini mengembalikan prediksi yang telah dinormalisasi ke skala aslinya.
    """
    denormalized_predictions = []

    for idx, prediction in enumerate(predictions):
        mean, std = mean_std_values[idx]
        samples = torch.tensor(prediction.samples)
        denormalized_prediction = samples * std + mean
        denormalized_forecast = SampleForecast(
            samples=denormalized_prediction.numpy(),
            start_date=prediction.start_date,
            item_id=prediction.item_id,
            info=prediction.info
        )
        denormalized_predictions.append(denormalized_forecast)

    return denormalized_predictions

def normalize(tensor, mean, std):
    return (tensor - mean) / std
