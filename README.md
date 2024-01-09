<!-- SHIELDS -->
<div align="left">

  ![Platform](https://img.shields.io/badge/Platform-Linux%20%7C%20macOS%20%7C%20Windows-informational)
  [![Python](https://img.shields.io/badge/Python-3.8%20%7C%203.9%20%7C%203.10%20%7C%203.11%20%7C%203.12-informational)](https://www.python.org/)
  [![Qiskit](https://img.shields.io/badge/Qiskit-%E2%89%A5%200.45.1-6133BD)](https://github.com/Qiskit/qiskit) <br />
  [![Tests](https://github.com/pedrorrivero/pyproject-qiskit/actions/workflows/test.yml/badge.svg)](https://github.com/pedrorrivero/pyproject-qiskit/actions/workflows/test.yml)
  [![Coverage](https://coveralls.io/repos/github/pedrorrivero/pyproject-qiskit/badge.svg?branch=main)](https://coveralls.io/github/pedrorrivero/pyproject-qiskit?branch=main)
  [![Release](https://img.shields.io/github/release/pedrorrivero/pyproject-qiskit.svg?include_prereleases&label=Release)](https://github.com/pedrorrivero/pyproject-qiskit/releases)
  [![DOI](https://img.shields.io/badge/DOI-zz.nnnn/zenodo.ddddddd-informational)](https://zenodo.org/)
  [![License](https://img.shields.io/github/license/pedrorrivero/pyproject-qiskit?label=License)](LICENSE.txt)

</div> <br />

<!-- PROJECT LOGO AND TITLE -->
<p align="center">
  <a href="README.md">
    <img src="https://github.com/pedrorrivero/pyproject-qiskit/blob/main/docs/media/cover.png?raw=true" alt="Logo" width="300">
  </a>
  <h1 align="center">Pyproject Qiskit</h1>
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

1. [About this Project](#about-this-project)
2. [Installation](#installation)
3. [Documentation](#documentation)
4. [Deprecation Policy](#deprecation-policy)
5. [Contributing](#contributing)
6. [Authors and Citation](#authors-and-citation)
7. [Acknowledgements](#acknowledgements)
8. [References](#references)
9. [License](#license)


<!-- ---------------------------------------------------------------------- -->

## About this Project

This template repository is a tool for creating [Qiskit](https://www.ibm.com/quantum/qiskit)-based Python projects quickly. It provides much of the necessary boilerplate code and configurations needed for a fully functional, professional, software package.

It was originally put together for quick development of _Quantum Software Prototypes_: collaborations between developers and researchers to bring users early access to solutions from cutting-edge research.

Check out the [file map](https://github.com/pedrorrivero/pyproject-qiskit/blob/main/FILEMAP.md) for more information on the structure of the repository.

<details>
<summary>Some projects using this template</summary>

- [Quantum Enablement](https://github.com/IBM-Quantum-Technical-Enablement/quantum-enablement)
- [Prototype ZNE](https://github.com/qiskit-community/prototype-zne)
- [PR Toolbox](https://github.com/pedrorrivero/pr-toolbox)
- [Staged Primitives](https://github.com/Qiskit-Extensions/staged-primitives)
</details>


<!-- ---------------------------------------------------------------------- -->

## Installation

The latest version of this software package can be easily installed, alongside all required dependencies, via `pip`:
```
pip install pyproject-qiskit
```

For more detailed information and alternative installation options see the [installation guide](https://github.com/pedrorrivero/pyproject-qiskit/blob/main/INSTALL.md).


<!-- ---------------------------------------------------------------------- -->

## Documentation

- This project includes a quick [reference guide](https://github.com/pedrorrivero/pyproject-qiskit/blob/main/docs/reference_guide.md) to get started with.
- Complete documentation can be found in the code docstrings.
- Check out the [file map](https://github.com/pedrorrivero/pyproject-qiskit/blob/main/FILEMAP.md) for more information on the structure of this repository.


<!-- ---------------------------------------------------------------------- -->

## Deprecation Policy

This package is meant to evolve rapidly and, as such, does not follow [Qiskit's deprecation policy](https://github.com/Qiskit/qiskit/blob/main/DEPRECATION.md). 

We may occasionally make breaking changes in order to improve the user experience. When possible, we will keep old interfaces and mark them as deprecated, as long as they can co-exist with the new ones. Each substantial improvement, breaking change, or deprecation will be documented in [`CHANGELOG.md`](https://github.com/pedrorrivero/pyproject-qiskit/blob/main/CHANGELOG.md). 

Careful version specification is encouraged (e.g. [version pinning](https://www.easypost.com/dependency-pinning-guide)).


<!-- ---------------------------------------------------------------------- -->

## Contributing

- The easiest way to contribute is by [giving feedback](https://github.com/pedrorrivero/pyproject-qiskit/blob/main/CONTRIBUTING.md#giving-feedback).
- If you wish to contribute to the development of the software, you must read and follow our [contribution guidelines](https://github.com/pedrorrivero/pyproject-qiskit/blob/main/CONTRIBUTING.md).
- By participating, you are expected to uphold our [code of conduct](https://github.com/pedrorrivero/pyproject-qiskit/blob/main/CODE_OF_CONDUCT.md).


<!-- ---------------------------------------------------------------------- -->

## Authors and Citation

This project is the work of [many people](https://github.com/pedrorrivero/pyproject-qiskit/graphs/contributors) who contribute at different levels. Please cite as per the included [BibTeX file](https://github.com/pedrorrivero/pyproject-qiskit/blob/main/CITATION.bib).


<!-- ---------------------------------------------------------------------- -->

## Acknowledgements

- [*Pedro Rivero*](https://github.com/pedrorrivero):
  for the development of [`pyproject-qiskit`](https://github.com/pedrorrivero/pyproject-qiskit), an open-source template repository for Qiskit-based software projects.
- [*Jim Garrison*](https://github.com/garrison):
  for insightful discussions and the original development of scripts for extremal version testing.


<!-- ---------------------------------------------------------------------- -->

## References

[1] [Qiskit](https://github.com/Qiskit/qiskit): An Open-source Framework for Quantum Computing


<!-- ---------------------------------------------------------------------- -->

## License

[Apache License 2.0](https://github.com/pedrorrivero/pyproject-qiskit/blob/main/LICENSE.txt)
