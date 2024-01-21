# Deprecation Policy

This software library is meant to evolve rapidly and, as such, does not follow [Qiskit's deprecation policy](https://github.com/Qiskit/qiskit/blob/main/DEPRECATION.md). Nonetheless, we will always try to give users ample time to adjust to changes without breaking code that they have already written.


## Supported API and pre-releases

Following [Python's naming conventions](https://realpython.com/python-double-underscore/), any module, variable, function, or class whose name starts with a leading underscore `_` will be considered _private_ and not part of this library's supported API.

Some capabilities may be pre-released before reaching a stable state. These will not adhere to the deprecation policies in place and will actively warn users of their unstable, pre-release, condition.

Every other piece of source code conforms the _public-facing_ API of this library and will therefore be subject to the rules outlined in this document.


## Migration guides

In alignment with the _Quantum Software Prototype_ mission, to avoid redundancy with [Qiskit](https://www.ibm.com/quantum/qiskit) and other IBM software products, once a specific capability from this library is integrated into IBM's stable product stack we will proceed to its deprecation here.

Said deprecation process will last for at least three months, and will not begin until a _migration guide_ explaining users how to transition to the new product feature is produced, approved, and published.


## Deprecations, breaking changes, and versioning

This library follows [semantic versioning](https://semver.org/).

In most cases, functionality will not be changed or removed without active warnings for a sufficient period of time. During this period, we will keep old interfaces and mark them as deprecated. Deprecations, changes and removals are considered API changes, and can only occur in _minor_ releases, not _patch_ releases.

We may occasionally make breaking changes (i.e. backwards incompatible) in order to bring new functionality to users more rapidly, simplify existing tooling, or facilitate maintenance. These changes will only be included in _major_ releases.

Major version zero may include breaking changes in minor releases.


## Documenting changes

Each substantial improvement, breaking change, or deprecation occurring on each release will be documented in [`CHANGELOG.md`](https://github.com/IBM-Quantum-Technical-Enablement/quantum-enablement/blob/main/CHANGELOG.md).
