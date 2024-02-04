import torch
import numpy as np
import torch.nn as nn
from qiskit_algorithms.utils import algorithm_globals

from qrnn import QRNN


class QuantumModel(nn.Module):
    """
    QuantumModel is a PyTorch module that represents a quantum model.

    Args:
        n_qubits (int): Number of qubits in the quantum model.
        input_size (int): Size of the input data.
        num_hiddens (int): Number of hidden units in the quantum model.
        batch_size (int): Batch size for training.
        target_size (int): Size of the target output.

    Attributes:
        qrnn (QRNN): Quantum recurrent neural network module.
        dense (nn.Linear): Linear layer for final output.

    Methods:
        forward(x): Performs forward pass through the quantum model.

    """

    def __init__(
        self, n_qubits, input_size, num_hiddens, batch_size, target_size
    ) -> None:
        super(QuantumModel, self).__init__()

        seed = 71
        np.random.seed = seed
        algorithm_globals.random_seed = seed
        torch.manual_seed = seed

        self.qrnn = QRNN(n_qubits, input_size, num_hiddens, batch_size)
        self.dense = nn.Linear(num_hiddens, target_size)

    def forward(self, x):
        """
        Performs forward pass through the quantum model.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output scores.

        """
        qlstm_out, _ = self.qrnn(x)
        dense_out = self.dense(qlstm_out)
        out_scores = dense_out

        return out_scores
