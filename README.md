<!-- SHIELDS -->
<div align="left">

  ![Platform](https://img.shields.io/badge/Platform-Linux%20%7C%20macOS%20%7C%20Windows-informational)
  [![Python](https://img.shields.io/badge/Python-3.8%20%7C%203.9%20%7C%203.10%20%7C%203.11%20%7C%203.12-informational)](https://www.python.org/)
  [![Qiskit](https://img.shields.io/badge/Qiskit-%E2%89%A5%200.45.1-6133BD)](https://github.com/Qiskit/qiskit) <br />
  [![Tests](https://github.com/IBM-Quantum-Technical-Enablement/quantum-enablement/actions/workflows/test.yml/badge.svg)](https://github.com/IBM-Quantum-Technical-Enablement/quantum-enablement/actions/workflows/test.yml)
  [![Coverage](https://coveralls.io/repos/github/IBM-Quantum-Technical-Enablement/quantum-enablement/badge.svg?branch=main)](https://coveralls.io/github/IBM-Quantum-Technical-Enablement/quantum-enablement?branch=main)
  [![Release](https://img.shields.io/github/release/IBM-Quantum-Technical-Enablement/quantum-enablement.svg?include_prereleases&label=Release)](https://github.com/IBM-Quantum-Technical-Enablement/quantum-enablement/releases)
  [![DOI](https://img.shields.io/badge/DOI-zz.nnnn/zenodo.ddddddd-informational)](https://zenodo.org/)
  [![License](https://img.shields.io/github/license/IBM-Quantum-Technical-Enablement/quantum-enablement?label=License)](LICENSE.txt)

</div> <br />

<!-- PROJECT LOGO AND TITLE -->
<p align="center">
  <a href="README.md">
    <img src="https://github.com/IBM-Quantum-Technical-Enablement/quantum-enablement/blob/main/docs/media/cover.png?raw=true" alt="Logo" width="300">
  </a>
  <h1 align="center">Quantum Enablement</h1>
</p>

<!-- QUICK LINKS -->
<!-- <p align="center">
  <a href="https://mybinder.org/">
    <img src="https://ibm.biz/BdPq3s" alt="Launch Demo" hspace="5" vspace="10">
  </a>
  <a href="https://www.youtube.com/c/qiskit">
    <img src="https://img.shields.io/badge/watch-video-FF0000.svg?style=for-the-badge&logo=youtube" alt="Watch Video" hspace="5" vspace="10">
  </a>
</p> -->


<!-- ---------------------------------------------------------------------- -->

## Table of contents

1. [About the Team](#about-the-team)
2. [Installation](#installation)
3. [Documentation](#documentation)
4. [Deprecation Policy](#deprecation-policy)
5. [Contributing](#contributing)
6. [Authors and Citation](#authors-and-citation)
7. [Acknowledgements](#acknowledgements)
8. [References](#references)
9. [License](#license)


<!-- ---------------------------------------------------------------------- -->

## About the Team

IBM's _Quantum Engineering and Enabling Technologies_ (QEET) team is meant to help enable quantum computing practitioners to get better results from quantum hardware, and to identify critical tooling which will help improve such end-user's workflows.

This team's mission is realized under three main threads:
1. _Tutorials and demonstrations_ -- 
   To highlight particular Qiskit features and their measured benefits around two key areas:
   - _Capabilities_ to improve hardware results. This will typically focus on steps 2 and 4 of the _Qiskit Pattern_ steps.
   - _Applications_ to drive user engagement with systems. This will typically focus on steps 1 and 4 of the _Qiskit Pattern_ steps.
2. _Prototype research and development_ --
   Consisting on the evaluation and validation of promising research results or methods leading to software tools and content creation. When Qiskit does not provide the needed functionality or performance, we will assemble and maintain _Quantum Software Prototypes_ while integration into the production stack is evaluated or completed. These will provide early access to solutions from cutting-edge research.
3. _Workshops and Residency Program_ --
   The above content will either be shared broadly through [IBM's Quantum Learning Platform](https://learning.quantum.ibm.com/), workshops, or the _Quantum Residency Program_ for Quantum Computational Centers.

This repository acts as a central _installable library_ for all of the team's assets. See [our issues list](https://github.com/IBM-Quantum-Technical-Enablement/quantum-enablement/issues) for a proposed list of prototypes and tutorials yet to be made.


<!-- ---------------------------------------------------------------------- -->

<details>
<summary><h3>Qiskit Pattern</h3></summary>

All content will adhere to the following structure:
1. _Quantum Encoding_ --
   Translating the target problem to a quantum native format is the critical first step in developing a quantum workflow. To this end, the team will focus on application agnostic methods that can be tested against arbitrary workflows.
2. _Circuit and Measurement Optimization_ --
   Once a given problem has been translated to the desired quantum native format, the resulting circuit and required measurements can usually be optimized in a variety of ways to ensure best performance.
3. _Execute on Quantum Hardware_ --
   Once a given problem is quantum encoded and optimized, users execute it on a quantum backend. We will highlight how to do so through the Qiskit Runtime Primitives whenverever possible. 
4. _Post Process Results_ --
   Once results are executed on quantum hardware a user needs to post-process the results in order to translate into the desired solution. This process can either be related to step 1 (e.g. selecting a given bitstring for many optimization problems) or step 2 (e.g. knitting results together from a cut circuit).

</details>


<!-- ---------------------------------------------------------------------- -->

## Installation

The latest version of this software library can be easily installed, alongside all required dependencies, via `pip`:
```
pip install quantum-enablement
```

For more detailed information and alternative installation options see the [installation guide](https://github.com/IBM-Quantum-Technical-Enablement/quantum-enablement/blob/main/INSTALL.md).


<!-- ---------------------------------------------------------------------- -->

## Documentation

- This project includes a quick [reference guide](https://github.com/IBM-Quantum-Technical-Enablement/quantum-enablement/blob/main/docs/reference_guide.md) to get started with.
- Complete documentation can be found in the code docstrings.
- Check out the [file map](https://github.com/IBM-Quantum-Technical-Enablement/quantum-enablement/blob/main/FILEMAP.md) for more information on the structure of this repository.


<!-- ---------------------------------------------------------------------- -->

## Deprecation Policy

- This software library is meant to evolve rapidly and, as such, follows its own [deprecation policy](DEPRECATION.md) different from Qiskit's.
- Each substantial improvement, breaking change, or deprecation occurring for each release will be documented in [`CHANGELOG.md`](https://github.com/IBM-Quantum-Technical-Enablement/quantum-enablement/blob/main/CHANGELOG.md).
- To avoid dependency issues, exact version specification is encouraged if no upcoming features are needed (e.g. [version pinning](https://www.easypost.com/dependency-pinning-guide)).


<!-- ---------------------------------------------------------------------- -->

## Contributing

- The easiest way to contribute is by [giving feedback](https://github.com/IBM-Quantum-Technical-Enablement/quantum-enablement/blob/main/CONTRIBUTING.md#giving-feedback).
- If you wish to contribute to the development of the software, you must read and follow our [contribution guidelines](https://github.com/IBM-Quantum-Technical-Enablement/quantum-enablement/blob/main/CONTRIBUTING.md).
- By participating, you are expected to uphold our [code of conduct](https://github.com/IBM-Quantum-Technical-Enablement/quantum-enablement/blob/main/CODE_OF_CONDUCT.md).


<!-- ---------------------------------------------------------------------- -->

## Authors and Citation

This project is the work of [many people](https://github.com/IBM-Quantum-Technical-Enablement/quantum-enablement/graphs/contributors) who contribute at different levels. Please cite as per the included [BibTeX file](https://github.com/IBM-Quantum-Technical-Enablement/quantum-enablement/blob/main/CITATION.bib).


<!-- ---------------------------------------------------------------------- -->

## Acknowledgements

- [*Pedro Rivero*](https://github.com/pedrorrivero):
  for the development of [`pyproject-qiskit`](https://github.com/pedrorrivero/pyproject-qiskit), an open-source template repository for Qiskit-based software projects.
- *Paul Nation*:
  for his pioneering work, technical insight, and guidance.


<!-- ---------------------------------------------------------------------- -->

## References

[1] [Qiskit](https://github.com/Qiskit/qiskit): An Open-source Framework for Quantum Computing


<!-- ---------------------------------------------------------------------- -->

## License

[Apache License 2.0](https://github.com/IBM-Quantum-Technical-Enablement/quantum-enablement/blob/main/LICENSE.txt)
