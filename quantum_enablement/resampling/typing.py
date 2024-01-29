# This code is part of Qiskit.
#
# (C) Copyright IBM 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Typing definitions and tools for resampling."""


from collections import Counter
from collections.abc import Sequence
from typing import Any, Union

from qiskit.primitives import SamplerResult
from qiskit.primitives.base.base_result import BasePrimitiveResult
from qiskit.result import ProbDistribution, QuasiDistribution

# TODO: define Memory class with casting methods
Memory = Sequence  # Note: Sequence[int], although historically it has been Sequence[str]
# TODO: add Counts
SamplesLike = Union[Memory, ProbDistribution, QuasiDistribution, SamplerResult]


################################################################################
## CASTING
################################################################################
def memory_to_distribution(memory: Memory) -> ProbDistribution:
    """Cast memory to probability distribution object."""
    if not isinstance(memory, Memory):
        raise TypeError(f"Expected type 'Memory', got '{type(memory)}' instead.")
    counts = Counter(memory)
    total = len(memory)
    distribution = {key: value / total for key, value in counts.items()}
    return ProbDistribution(distribution, shots=total)


def memory_to_quasi_dist(memory: Memory) -> QuasiDistribution:
    """Cast memory to quasi-distribution object."""
    distribution = memory_to_distribution(memory)
    return distribution_to_quasi_dist(distribution)


def memory_to_result(memory: Memory) -> SamplerResult:
    """Cast memory to sampler result object."""
    quasi_dist = memory_to_quasi_dist(memory)
    return quasi_dist_to_result(quasi_dist)


def distribution_to_quasi_dist(distribution: ProbDistribution) -> QuasiDistribution:
    """Cast probability distribution to quasi-distribution object."""
    if not isinstance(distribution, ProbDistribution):
        raise TypeError(f"Expected type 'ProbDistribution', got '{type(distribution)}' instead.")
    return QuasiDistribution(
        distribution,
        shots=distribution.shots,
        stddev_upper_bound=None,  # TODO: compute stddev_upper_bound
    )  # TODO: add class-method from ProbDistribution to qiskit


def distribution_to_result(distribution: ProbDistribution) -> SamplerResult:
    """Cast probability distribution to sampler result object."""
    quasi_dist = distribution_to_quasi_dist(distribution)
    return quasi_dist_to_result(quasi_dist)


def quasi_dist_to_result(quasi_dist: QuasiDistribution) -> SamplerResult:
    """Cast quasi-distribution to sampler result object."""
    if not isinstance(quasi_dist, QuasiDistribution):
        raise TypeError(f"Expected type 'QuasiDistribution', got '{type(quasi_dist)}' instead.")
    metadatum: dict[str, Any] = {}
    if quasi_dist.shots:
        metadatum["shots"] = quasi_dist.shots
    if quasi_dist.stddev_upper_bound:
        pass  # TODO: add new metadata
    return SamplerResult(quasi_dists=[quasi_dist], metadata=[metadatum])


################################################################################
## MISC
################################################################################
# TODO: add to qiskit.primitives.base.base_result
def compose_results(*results) -> BasePrimitiveResult:
    """Compose primitive results into a single result."""
    result_type = type(results[0])
    if not issubclass(result_type, BasePrimitiveResult):
        raise TypeError(f"Expected type 'BasePrimitiveResult', got '{result_type}' instead.")
    if not all(isinstance(result, result_type) for result in results):
        raise TypeError(f"Type mismatch, expected all inputs to be of type '{result_type}'")
    experiment_values = (
        experiment.values()
        for result in results
        for experiment in result._generate_experiments()  # pylint: disable=protected-access
    )
    return result_type(*zip(*experiment_values))
