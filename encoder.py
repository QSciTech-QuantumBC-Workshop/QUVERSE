from typing import List, Union
from dataclasses import dataclass
from qiskit.quantum_info import Pauli, SparsePauliOp
from qiskit.circuit.library import TwoLocal, ZZFeatureMap

import params


@dataclass
class Endoding:
    """
    A class representing an encoding scheme for quantum machine learning.

    Attributes:
        feature_dim (int): The dimension of the input feature space.
        entanglement (str): The type of entanglement to be used in the feature map and ansatz.
        feature_map_rep (int): The number of times the feature map is repeated.
        var_form_rep (int): The number of times the variational form is repeated.
    """

    feature_dim: int
    entanglement: str = "linear"
    feature_map_rep: int = params.feature_map_rep
    var_form_rep: int = params.var_form_rep

    def get_feature_map(self) -> ZZFeatureMap:
        """
        Get the feature map circuit for the encoding scheme of our input. For our specific case, Angle Encoding
        might be the most straightforward and effective method for encoding our dataset into a quantum state
        for several reasons:

            - Direct Mapping of Continuous Variables: Each of our features can be directly mapped to angles in
            quantum gates, making the encoding process intuitive and preserving the continuous nature of our data.
            - Scalability: This method scales linearly with the number of features, making it suitable for datasets
            with a moderate number of features.
            - Hardware Efficiency: Compared to amplitude encoding, angle encoding requires fewer quantum operations,
            making it more feasible on current noisy intermediate-scale quantum (NISQ) devices.

        The following code does not direcetly perform the *Angle Encoding* but sets up a quantum circuit than can be used
        for angle encoding when combined with our actual data. See details following:

            - **feature_dim**: This refers to the number of features that we wish to encode. In this scenario,
            `three` - displacement, velocity, acceleration
            - **feature_map_rep**: This is the number of repetitions of our feature map layer. Increasing it increases the expressivity
            of our model but also increases the circuit depth. We will stick to `1` for now.
            - **ent**: This is the entanglement strategy among our qubits. We can go for `linear`, `circular`, `full` etc. But we will
            choose `linear` for now since it is computationally less expensive.

        Returns:
            ZZFeatureMap: The feature map circuit.
        """
        return ZZFeatureMap(
            feature_dimension=self.feature_dim,
            reps=self.feature_map_rep,
            entanglement=self.entanglement,
            name="feature encoding block",
        )

    def get_ansatz(self, rotations) -> TwoLocal:
        """
        Get the ansatz circuit for the encoding scheme. We will use the `TwoLocal` circuit
        as the variational form (ansatz) of our quantum model. This ansatz is a parameterized circuit
        that we will train to learn the mapping from the encoded inputs to our desired outputs.

        Args:
            rotations (list): A list of rotation gates to be applied in the ansatz.

        Returns:
            TwoLocal: The ansatz circuit.
        """
        return TwoLocal(
            num_qubits=self.feature_dim,
            rotation_blocks=rotations,
            entanglement_blocks="cx",
            entanglement=self.entanglement,
            reps=self.var_form_rep,
            name="variational block",
        )

    def get_circuit(self, feature_map, ansatz):
        """
        Get the full circuit for the encoding scheme. Here, we will combine our input encoding circuit
        `feature_map` with our variational circuit `ansatz` into a single quantum circuit, which can then
        be used for training.

        Args:
            feature_map (ZZFeatureMap): The feature map circuit.
            ansatz (TwoLocal): The ansatz circuit.

        Returns:
            QuantumCircuit: The full circuit.
        """
        return feature_map.compose(ansatz)

    def get_observable(self, num_features) -> Pauli:
        """
        Get the observable for the encoding scheme.

        Args:
            num_features (int): The number of features.

        Returns:
            Pauli: The observable.
        """
        return Pauli("Z" * num_features)

    def get_pauli_obs(self, n_qubits) -> Union[List[SparsePauliOp], ValueError]:
        """
        Get a list of Pauli operators for the encoding scheme.

        Args:
            n_qubits (int): The number of qubits.

        Returns:
            list: A list of SparsePauliOp objects representing the Pauli operators.

        Raises:
            ValueError: If n_qubits is not a positive integer.
        """
        if not isinstance(n_qubits, int) or n_qubits <= 0:
            raise ValueError("n_qubits must be a positive integer")

        pauli_ops = []
        for i in range(n_qubits):
            pauli_str = "I" * i + "Z" + "I" * (n_qubits - i - 1)
            pauli_op = SparsePauliOp.from_list([(pauli_str, 1)])
            pauli_ops.append(pauli_op)

        return pauli_ops
