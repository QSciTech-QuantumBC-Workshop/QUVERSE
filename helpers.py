import os
import numpy as np
from typing import List, Tuple
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def get_dataset(dataset_dir: str) -> tuple:
    """
    Load the train and test datasets from the given directory.

    Args:
        dataset_dir (str): The directory path where the datasets are located.

    Returns:
        tuple: A tuple containing the train and test datasets.
    """
    train_data = pd.read_csv(os.path.join(dataset_dir, "train.csv"))
    test_data = pd.read_csv(
        os.path.join(dataset_dir, "test.csv"),
    )
    return train_data, test_data


def create_animation(train_data: pd.DataFrame, animation_name: str) -> None:
    """
    Creates an animation of a simple harmonic oscillator using the given train_data.

    Parameters:
    train_data (DataFrame): The training data containing 'time' and 'displacement' columns.
    animation_name (str): The name of the animation file to be saved.

    Returns:
    None
    """

    sample_x, sample_y = (
        train_data["time"].to_list(),
        train_data["displacement"].to_list(),
    )

    # First set up the figure, the axis, and the plot element we want to animate
    fig = plt.figure()
    ax = plt.axes(xlim=(0, 150), ylim=(-1, 1))
    ax.set_title("Simple Harmonic Oscillator")
    ax.set_xlabel("Time")
    ax.set_ylabel("Displacement")
    (line,) = ax.plot([], [], lw=2)

    # initialization function: plot the background of each frame
    def init():
        line.set_data([], [])
        return (line,)

    def animate(i):
        line.set_data(sample_x[:i], sample_y[:i])
        return (line,)

    anim = FuncAnimation(
        fig, animate, init_func=init, frames=150, blit=True, interval=20
    )
    anim.save(animation_name, fps=20, extra_args=["-vcodec", "libx264"])
    # plt.show()


def normalize_feat(features):
    """
    Normalize the features by scaling each feature to the range [0, 1]. Expects a numpy NDArray and returns NDArray

    Parameters:
    - features: numpy array of shape (n_samples, n_features)
        The input features to be normalized.

    Returns:
    - normalized_features: numpy array of shape (n_samples, n_features)
        The normalized features.
    """
    num_features = features.shape[1]

    for i in range(num_features):
        features[:, i] = features[:, i] / (
            np.max(features[:, i]) - np.min(features[:, i])
        )

    return features


def time_window_batch(input_data, window, batch_size) -> List[Tuple[np.ndarray]]:
    """
    Splits the input data into batches of time windows.

    Args:
        input_data (array-like): The input data to be split into batches.
        window (int): The size of the time window.
        batch_size (int): The number of time windows in each batch.

    Returns:
        list: A list of tuples, where each tuple contains a batch of time windows and their corresponding labels.
    """
    window_data = []
    L = len(input_data)
    for i in range(0, L - window, batch_size):
        train_series_batch = []
        train_label_batch = []
        for j in range(batch_size):
            train_series = input_data[i + j : i + j + window]
            train_label = input_data[i + j + window : i + j + window + 1]
            train_series_batch.append(train_series)
            train_label_batch.append(train_label)
        window_data.append((np.array(train_series_batch), np.array(train_label_batch)))
    return window_data
