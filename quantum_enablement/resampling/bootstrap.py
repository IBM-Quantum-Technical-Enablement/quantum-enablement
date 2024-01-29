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

"""Bootstrapping tools and techniques."""


from functools import singledispatchmethod

from numpy.random import Generator, default_rng
from qiskit.primitives import SamplerResult
from qiskit.result import ProbDistribution, QuasiDistribution

from .typing import (
    Memory,
    SamplesLike,
    compose_results,
    memory_to_distribution,
    memory_to_quasi_dist,
    memory_to_result,
)


# TODO: warn if requested size exceeds input size/shots
class Bootstrap:
    """Bootstrap resampling utility class.

    Args:
        default_size: the number of (re)samples to collect randomly
            in the absence of any other provided or inferred `size`.
            If `None`, it will default to `10_000`.
        seed: an optional seed for the internal (pseudo-)random number generator.
    """

    def __init__(self, *, default_size: int | None = None, seed: int | None = None) -> None:
        self._rng: Generator = default_rng(seed)
        self.default_size = default_size or 10_000

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
        self._default_size: int = self._validate_size(size)

    ################################################################################
    ## PUBLIC API
    ################################################################################
    @singledispatchmethod
    def resample_memory(self, samples, *, size: int | None = None) -> Memory:
        """Resample memory randomly.

        Args:
            samples: original `SamplesLike` object to resample from.
            size: the number of (re)samples to collect randomly.
                If `None`, it will try to be inferred from `samples`
                and fall back to the default value otherwise.

        Returns:
            Memory object randomly resampled from input samples.
        """
        raise TypeError(f"Expected type 'SamplesLike', got '{type(samples)}' instead.")

    @resample_memory.register
    def _(self, samples: SamplerResult, *, size: int | None = None) -> Memory:
        if samples.num_experiments != 1:
            raise ValueError(
                f"Number of experiments in input result ({samples.num_experiments}) "
                "exceeds the maximum number of experiments allowed (1)."
            )
        quasi_dist = samples.quasi_dists[0]
        metadatum = samples.metadata[0]
        size = size or metadatum.get("shots", None)
        return self.resample_memory(quasi_dist, size=size)

    @resample_memory.register
    def _(self, samples: QuasiDistribution, *, size: int | None = None) -> Memory:
        distribution = samples.nearest_probability_distribution(return_distance=False)
        return self.resample_memory(distribution, size=size)

    @resample_memory.register
    def _(self, samples: ProbDistribution, *, size: int | None = None) -> Memory:
        population = list(samples.keys())
        probabilities = list(samples.values())
        size = size or samples.shots or self.default_size
        size = self._validate_size(size)
        resamples = self.rng.choice(population, p=probabilities, size=size, replace=True)
        return resamples.tolist()

    @resample_memory.register
    def _(self, samples: Memory, *, size: int | None = None) -> Memory:
        size = size or self.default_size
        size = self._validate_size(size)
        resamples = self.rng.choice(samples, p=None, size=size, replace=True)
        return resamples.tolist()

    def resample_distribution(
        self, samples: SamplesLike, *, size: int | None = None
    ) -> ProbDistribution:
        """Resample probability distribution randomly.

        Args:
            samples: original `SamplesLike` object to resample from.
            size: the number of (re)samples to collect randomly.
                If `None`, it will try to be inferred from `samples`
                and fall back to the default value otherwise.

        Returns:
            ProbDistribution object randomly resampled from input samples.
        """
        memory = self.resample_memory(samples, size=size)
        return memory_to_distribution(memory)

    def resample_quasi_dist(
        self, samples: SamplesLike, *, size: int | None = None
    ) -> QuasiDistribution:
        """Resample quasi-distribution randomly.

        Args:
            samples: original `SamplesLike` object to resample from.
            size: the number of (re)samples to collect randomly.
                If `None`, it will try to be inferred from `samples`
                and fall back to the default value otherwise.

        Returns:
            QuasiDistribution object randomly resampled from input samples.
        """
        memory = self.resample_memory(samples, size=size)
        return memory_to_quasi_dist(memory)

    @singledispatchmethod
    def resample_result(self, samples, *, size: int | None = None) -> SamplerResult:
        """Resample sampler result randomly.

        Args:
            samples: original `SamplesLike` object to resample from.
            size: the number of (re)samples to collect randomly.
                If `None`, it will try to be inferred from `samples`
                and fall back to the default value otherwise.

        Returns:
            SamplerResult object randomly resampled from input samples.
        """
        memory = self.resample_memory(samples, size=size)
        return memory_to_result(memory)

    @resample_result.register  # Note: to account for multi-experiment results
    def _(self, samples: SamplerResult, *, size: int | None = None) -> SamplerResult:
        resamples: list[SamplerResult] = []
        for result in samples.decompose():
            memory = self.resample_memory(result, size=size)
            resample = memory_to_result(memory)
            resamples.append(resample)
        return compose_results(*resamples)

    ################################################################################
    ## INTERNAL API
    ################################################################################
    def _validate_size(self, size: int) -> int:
        try:
            size = int(size)
        except (TypeError, ValueError) as error:
            raise TypeError(f"Expected type 'int', got '{type(size)}' instead.") from error
        if size < 1:
            raise ValueError(f"Resampling size must be greater than zero, got '{size}' instead.")
        return size
