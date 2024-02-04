import numpy as np

from typing import Optional
from functools import cache
import torch
import torch.nn as nn
from torch import Tensor

from qiskit_aer import AerSimulator
from qiskit.primitives import BackendEstimator
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.connectors import TorchConnector

from encoder import Endoding


class QRNN(nn.Module):
    def __init__(
        self,
        n_qubits,
        input_size,
        num_hiddens,
        batch_size,
        *,
        ansatz_rotations=["ry", "rz"]
    ) -> None:
        """
        Quantum Recurrent Neural Network (QRNN) module.

        Args:
            n_qubits (int): Number of qubits.
            input_size (int): Size of the input.
            num_hiddens (int): Number of hidden units.
            batch_size (int): Batch size.
            ansatz_rotations (list, optional): List of ansatz rotations. Defaults to ['ry', 'rz'].
        """
        super(QRNN, self).__init__()

        encoder = Endoding(n_qubits)
        fmap = encoder.get_feature_map()
        ansatz = encoder.get_ansatz(rotations=ansatz_rotations)

        self.n_qubits = n_qubits
        self.input_size = input_size
        self.num_hiddens = num_hiddens
        self.batch_size = batch_size
        self.input_params = fmap.parameters
        self.weight_params = ansatz.parameters
        self.q_circuit = encoder.get_circuit(fmap, ansatz)
        # self.obs = encoder.get_observable(n_qubits)
        self.obs = encoder.get_pauli_obs(n_qubits)
        self.gates = ("forget", "input", "update", "output")

        self.create_layers()

    @cache
    def get_estimator(self) -> BackendEstimator:
        """
        Get the estimator for quantum circuit simulation. For now, we will be using a
        `simulator` *rather than* an actual quantum hardware to execute our circuit,
        `qml_circuit` for the different inputs

        Returns:
            BackendEstimator: The backend estimator.
        """
        options = {}
        qasm_sim = AerSimulator()
        return BackendEstimator(backend=qasm_sim, options=options)

    def create_layers(self) -> None:
        """
        Create quantum and classical layers.
        """
        # quantum layers
        self.quantum_layers = {k: self.construct_layer() for k in self.gates}

        # classical layers
        self.classical_layer_in = nn.Linear(
            self.input_size + self.num_hiddens, self.n_qubits, bias=False
        )
        self.classical_layer_out = nn.Linear(
            self.n_qubits, self.num_hiddens, bias=False
        )

    def construct_layer(self) -> TorchConnector:
        """
        Construct a quantum layer. We will be using tha `Estimator` primitive rather than sampler.
        The `Estimator` primitive in Qiskit is designed to estimate expectation values of observables
        given a parameterized quantum circuit. It is particularly useful for our QCL where we are interested
        in the expectation values of the operators that correspond to our system's physical quantities.

        Returns:
            TorchConnector: The quantum layer.
        """
        estiamator = self.get_estimator()

        layer = EstimatorQNN(
            estimator=estiamator,
            circuit=self.q_circuit,
            input_params=self.input_params,
            weight_params=self.weight_params,
            input_gradients=True,
            observables=self.obs,
        )

        initial_weights = Tensor(np.random.normal(0, 0.01, len(self.weight_params)))

        return TorchConnector(layer, initial_weights=initial_weights)

    def forward(self, x: Tensor, H_C: Optional[Tensor] = None) -> tuple:
        """
        Forward pass of the QRNN. We are implementing the Long Short Term Memory (LSTM) variant
        of the Recurrent Neural Network (RNN) architecture. This takes care of the vanishing gradient problem.

        Args:
            x (Tensor): Input tensor.
            H_C (Optional[Tensor], optional): Hidden and cell state tensor. Defaults to None.

        Returns:
            Tuple[Tensor, Tuple[Tensor, Tensor]]: Output tensor and updated hidden and cell state.
        """
        if H_C is None:
            H = torch.zeros(self.batch_size, self.num_hiddens)
            C = torch.zeros(self.batch_size, self.num_hiddens)
        else:
            H, C = H_C.detach()

        outputs = []
        for X in x:
            X = X.reshape((x.shape[1], x.shape[2]))
            V_t = torch.cat((H, X), dim=1)
            x_in = self.classical_layer_in(V_t)

            I = torch.sigmoid(
                self.classical_layer_out(self.quantum_layers["input"](x_in))
            )
            F = torch.sigmoid(
                self.classical_layer_out(self.quantum_layers["forget"](x_in))
            )
            O = torch.sigmoid(
                self.classical_layer_out(self.quantum_layers["output"](x_in))
            )
            C_tilde = torch.tanh(
                self.classical_layer_out(self.quantum_layers["update"](x_in))
            )
            C = F * C + I * C_tilde
            H = O * torch.tanh(C)
            outputs.append(H.unsqueeze(0))

        outputs = torch.cat(outputs, dim=0)
        return outputs, (H, C)
