"""
Copyright 2024 BlackRock, Inc.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import traceback
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from ripser import Rips


def calculate_scaled_distance_distribution(sub_df: pd.DataFrame, *args) -> Dict[str, Any]:
    """
    Calculates the minimum value, the mean and the interquartile range for the similarity
    metric (e.g. L2/Cosine/Inner Product)
    """
    scale = (np.sort(sub_df.score) / np.sort(sub_df.score)[0])[1:]

    scale_distribution = {
        "scale_mean": np.mean(scale),
        "scale_min": np.min(scale),
        "iq25-75_scale": np.quantile(scale, 0.75) - np.quantile(scale, 0.25),
    }
    return scale_distribution


def calculate_first_order_homology_distribution(sub_df: pd.DataFrame, depsilon,  *args) -> Dict[str, Any]:
    """
    Calculates the topological features for the embeddings of both the query and retrieved chunks
    """
    try:
        rips = Rips()
        mask = (np.sort(sub_df.score) / np.sort(sub_df.score)[0]) <= depsilon
        if sum(mask) < 5:
            raise ValueError
        chunks_embed = np.array(sub_df["chunk_embeddings"])[mask, :]
        query_embed = np.array(sub_df["query_embedding"])
    
        renormalised_embeddings = chunks_embed - query_embed[np.newaxis, :]
        renormalised_embeddings = (
            renormalised_embeddings / np.linalg.norm(renormalised_embeddings, axis=-1)[:, np.newaxis]
        )

        diagrams = rips.fit_transform(renormalised_embeddings[:, :])

        neighbour_0th_homology = diagrams[0][:-1, 1]
        neighbour_1st_homology = diagrams[1][:, :2]

        holes_lifetimes = neighbour_1st_homology[:, 1] - neighbour_1st_homology[:, 0]
        homology_distribution = {
            "max_homology_birth": np.max(neighbour_0th_homology),
            "mean_homology_birth": np.mean(neighbour_0th_homology),
            "w1_h0": np.sum(np.abs(neighbour_0th_homology - neighbour_0th_homology/2.0))
            / (len(neighbour_0th_homology)-1),
            "std_homology_birth": np.std(neighbour_0th_homology),
            "mean_homology1st_birth": np.mean(neighbour_1st_homology, axis=0)[0],
            "mean_homology1st_lifetime": np.mean(holes_lifetimes),
            "ltmax_h1": np.max(holes_lifetimes),
        }
        return homology_distribution
    except Exception:
        print(traceback.format_exc())
        homology_distribution = {
            "max_homology_birth": np.nan,
            "mean_homology_birth": np.nan,
            "w1_h0": np.nan,
            "std_homology_birth": np.nan,
            "mean_homology1st_birth": np.nan,
            "mean_homology1st_lifetime": np.nan,
            "ltmax_h1": np.nan,
        }
        return homology_distribution
        


def calculate_silhouette_score_distribution(sub_df: pd.DataFrame, *args) -> Dict[str, Any]:
    """
    Calculates the mean of the standard deviation of the silhouette score
    """
    silhouette_score_distribution = {
        "silhouette_score_mean": sub_df["silhouette_score"].mean(),
        "silhouette_score_std": sub_df["silhouette_score"].std(),
    }
    return silhouette_score_distribution


def calculate_doc_spread(sub_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Calculates the number of number of distinct documents of the retrieved chunks over
    the total number of total chunks retrieved
    """
    return {"top_k_doc_spread": len(set(sub_df["docname"])) / len(sub_df)}


def calculate_topic_spread(sub_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Calculates the number of number of distinct topics contained in the the retrieved
    chunks over the total number of total chunks retrieved
    """
    return {"top_k_topic_spread": len(set(sub_df["topic_label"])) / len(sub_df)}


def calculate_relevant_features(sub_df: pd.DataFrame, kde_features_order: List[str]) -> Dict[str, Any]:
    """
    Computates all features and combines the relevant ones into a dictionary
    """
    feature_set = set(kde_features_order)

    features = calculate_scaled_distance_distribution(sub_df)
    features.update(calculate_first_order_homology_distribution(sub_df))
    features.update(calculate_silhouette_score_distribution(sub_df))
    features.update(calculate_doc_spread(sub_df))
    features.update(calculate_topic_spread(sub_df))

    return {k: v for k, v in features.items() if k in feature_set}
