import pandas as pd
import numpy as np
from typing import Callable

from config import DEFAULT_K, USER_COL, ITEM_COL


def hit_rate(rating_true: list, rating_pred: list, k: int = DEFAULT_K) -> int:
    """
    calculate hit rate for single user
    :param rating_true: list order doesn't matter
    :param rating_pred: list order matters
    :param k: number of ratings to compare
    :return: 1 if hit 0 if no
    """
    prediction = rating_pred[0:k]
    for it in prediction:
        if it in rating_true:
            return 1
    return 0


def precision_at_k(rating_true: list, rating_pred: list, k: int = DEFAULT_K) -> float:
    """
    calculate precision@k
    :param rating_true: list order doesn't matter
    :param rating_pred: list order matters
    :param k: number of ratings to compare
    :return: float precision at k
    """
    prediction = rating_pred[0:k]
    relevant = [rating for rating in prediction if rating in rating_true]

    return len(relevant) / len(prediction)


def recall_at_k(rating_true: list, rating_pred: list, k: int = DEFAULT_K) -> float:
    """
    calculate recall@k
    :param rating_true: list order doesn't matter
    :param rating_pred: list order matters
    :param k: number of ratings to compare
    :return: float recall at k
    """
    prediction = rating_pred[0:k]
    relevant = [rating for rating in prediction if rating in rating_true]

    return len(relevant) / len(rating_true)


def average_precision_at_k(rating_true: list, rating_pred: list, k: int = DEFAULT_K) -> float:
    """
    calculate average precision@k
    :param rating_true: list order doesn't matter
    :param rating_pred: list order matters
    :param k: number of ratings to compare
    :return: float average precision at k
    """
    prediction = rating_pred[0:k]
    hits = 0
    average_precision = 0
    for i, rating in enumerate(prediction):
        if rating in rating_true:
            hits += 1
            average_precision += hits / (i + 1)
    if hits == 0:
        return 0
    return average_precision / hits


def ndcg_at_k(rating_true: list, rating_pred: list, k: int = DEFAULT_K) -> float:
    """
    Calculate Normalized cumulative gain at k
    :param k: number of ratings to compare
    :param rating_true: list order matters
    :param rating_pred: list order matters
    :return: ndcg at k
    """
    prediction = rating_pred[0:k]
    ideal_gain = sum([1 / np.log2(i + 2) for i in range(k)])
    discounted_cumulative_gain = sum([1 / np.log2(i + 2) for i, rating
                                      in enumerate(prediction) if rating in rating_true])
    return discounted_cumulative_gain / ideal_gain


def mean_metrics(true_df: pd.DataFrame,
                 pred_df: pd.DataFrame,
                 rating_func: Callable,
                 col_user: str = USER_COL,
                 col_rating: str = ITEM_COL,
                 k: int = DEFAULT_K) -> np.ndarray:
    """
    calculate chosen mean metric from the metric list over the dataframe
    :param true_df: dataframe with ground truth values
    :param pred_df: dataframe with predicted values
    :param rating_func: one of metric functions
    :param col_user: name of column for user in dataframe
    :param col_rating: name of column for items in dataframes
    :param k: number of ratings to compare
    :return:
    """
    merged_results = true_df.merge(pred_df,
                                   on=col_user,
                                   suffixes=('_true', '_pred'))
    return np.mean([rating_func(merged_results[col_rating + '_true'][i],
                                merged_results[col_rating + '_pred'][i], k) for i in range(merged_results.shape[0])])


METRIC_LIST = [hit_rate, precision_at_k, recall_at_k, average_precision_at_k, ndcg_at_k]


def evaluate_model(truth_df: pd.DataFrame, predicted_df: pd.DataFrame, k: int = DEFAULT_K) -> None:
    """
    calculate all presented metrics to evaluate the model of recommender system
    :param truth_df: dataframe with ground truth values
    :param predicted_df: dataframe with predicted values
    :param k: number of ratings to compare
    :return: None
    """
    for metric in METRIC_LIST:
        print("{}@{}: {:.3f}".format(metric.__name__, k,
                                     mean_metrics(truth_df,
                                                  predicted_df,
                                                  metric,
                                                  k=k)))
