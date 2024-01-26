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

"""Collection of resampling techniques."""

from collections import Counter
from collections.abc import Collection
from functools import singledispatchmethod
from typing import Any

from numpy.random import Generator, default_rng
from qiskit.primitives import SamplerResult
from qiskit.primitives.base.base_result import BasePrimitiveResult
from qiskit.result import ProbDistribution, QuasiDistribution


# TODO: internal methods
# TODO: warn if requested size exceeds input size/shots
# TODO: add counts
# FIXME: memory is made out of bitstrings, int conversion is needed or renaming
# FIXME: memory needs ordering -> Sequence better than Collection,
# FIXME: but Collection makes sense -> rename
class Bootstrap:
    """Bootstrap resampling utility class."""

    def __init__(self, *, default_size: int = 10_000, seed: int | None = None) -> None:
        self._rng: Generator = default_rng(seed)
        self._default_size = default_size

    ################################################################################
    ## PROPERTIES
    ################################################################################
    @property
    def rng(self) -> Generator:
        """Random number generator."""
        return self._rng

    @property
    def default_size(self) -> int:
        """Default resample size."""
        return self._default_size

    @default_size.setter
    def default_size(self, size: int) -> None:
        self._default_size = int(size)

    ################################################################################
    ## TYPE CASTING
    ################################################################################
    @staticmethod
    def memory_to_distribution(memory: Collection) -> ProbDistribution:
        """Cast memory to probability distribution object."""
        counts = Counter(memory)
        total = len(memory)
        distribution = {key: value / total for key, value in counts.items()}
        return ProbDistribution(distribution, shots=total)

    @classmethod
    def memory_to_quasi_dist(cls, memory: Collection) -> ProbDistribution:
        """Cast memory to quasi-distribution object."""
        distribution = cls.memory_to_distribution(memory)
        return cls.distribution_to_quasi_dist(distribution)

    @classmethod
    def memory_to_result(cls, memory: Collection) -> ProbDistribution:
        """Cast memory to sampler result object."""
        quasi_dist = cls.memory_to_quasi_dist(memory)
        return cls.quasi_dist_to_result(quasi_dist)

    @staticmethod
    def distribution_to_quasi_dist(distribution: ProbDistribution) -> QuasiDistribution:
        """Cast probability distribution to quasi-distribution object."""
        return QuasiDistribution(
            distribution,
            shots=distribution.shots,
            stddev_upper_bound=None,  # TODO: compute stddev_upper_bound
        )  # TODO: add class-method from ProbDistribution to qiskit

    @classmethod
    def distribution_to_result(cls, distribution: ProbDistribution) -> SamplerResult:
        """Cast probability distribution to sampler result object."""
        quasi_dist = cls.distribution_to_quasi_dist(distribution)
        return cls.quasi_dist_to_result(quasi_dist)

    @staticmethod
    def quasi_dist_to_result(quasi_dist: QuasiDistribution) -> SamplerResult:
        """Cast quasi-distribution to sampler result object."""
        metadatum: dict[str, Any] = {}
        if quasi_dist.shots:
            metadatum["shots"] = quasi_dist.shots
        if quasi_dist.stddev_upper_bound:
            pass  # TODO: add variance adn std_error
        return SamplerResult(quasi_dists=[quasi_dist], metadata=[metadatum])

    ################################################################################
    ## METHODS
    ################################################################################
    @singledispatchmethod
    def resample_memory(self, data, *, size: int | None = None) -> list[int]:
        """Resample memory randomly."""
        raise TypeError(f"Invalid data type input: {type(data)}.")

    @resample_memory.register
    def _(self, data: SamplerResult, *, size: int | None = None) -> list[int]:
        if data.num_experiments != 1:
            raise ValueError(
                f"Number of experiments in input result ({data.num_experiments}) "
                "exceeds the maximum number of experiments allowed (1)."
            )
        quasi_dist = data.quasi_dists[0]
        metadatum = data.metadata[0]
        size = size or metadatum.get("shots", None)
        return self.resample_memory(quasi_dist, size=size)

    @resample_memory.register
    def _(self, data: QuasiDistribution, *, size: int | None = None) -> list[int]:
        distribution = data.nearest_probability_distribution(return_distance=False)
        return self.resample_memory(distribution, size=size)

    @resample_memory.register
    def _(self, data: ProbDistribution, *, size: int | None = None) -> list[int]:
        population = list(data.keys())
        probabilities = list(data.values())
        size = size or data.shots or self.default_size
        resamples = self.rng.choice(population, p=probabilities, size=size, replace=True)
        return resamples.tolist()

    @resample_memory.register
    def _(self, data: Collection, *, size: int | None = None) -> list:
        size = size or self.default_size
        resamples = self.rng.choice(data, p=None, size=size, replace=True)
        return resamples.tolist()

    def resample_distribution(
        self,
        data: SamplerResult | QuasiDistribution | ProbDistribution | Collection,
        *,
        size: int | None = None,
    ) -> ProbDistribution:
        """Resample probability distribution randomly."""
        memory = self.resample_memory(data, size=size)
        return self.memory_to_distribution(memory)

    def resample_quasi_dist(
        self,
        data: SamplerResult | QuasiDistribution | ProbDistribution | Collection,
        *,
        size: int | None = None,
    ) -> QuasiDistribution:
        """Resample quasi-distribution randomly."""
        memory = self.resample_memory(data, size=size)
        return self.memory_to_quasi_dist(memory)

    @singledispatchmethod
    def resample_result(self, data, *, size: int | None = None) -> SamplerResult:
        """Resample sampler result randomly."""
        memory = self.resample_memory(data, size=size)
        return self.memory_to_result(memory)

    @resample_result.register  # Note: to account for multi-experiment results
    def _(self, data: SamplerResult, *, size: int | None = None) -> SamplerResult:
        resamples: list[SamplerResult] = []
        for result in data.decompose():
            memory = self.resample_memory(result, size=size)
            resample = self.memory_to_result(memory)
            resamples.append(resample)
        return compose_results(*resamples)


# TODO: add to qiskit.primitives.base.base_result
def compose_results(*results) -> BasePrimitiveResult:
    """Compose primitive results into a single result."""
    result_type = type(results[0])
    # if not issubclass(result_type, BasePrimitiveResult):
    #     raise TypeError(...)
    # if not all(isinstance(result, result_type) for result in results):
    #     raise TypeError(...)
    experiment_values = (
        experiment.values()
        for result in results
        for experiment in result._generate_experiments()  # pylint: disable=protected-access
    )
    return result_type(*zip(*experiment_values))
