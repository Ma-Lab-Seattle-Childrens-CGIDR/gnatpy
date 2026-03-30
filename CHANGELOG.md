# Changelog

<!--toc:start-->

- [Changelog](#changelog)
  - [Release DEV](#release-dev)
  - [Release 0.2.0](#release-020)
  - [Release 0.1.0](#release-010)

<!--toc:end-->

## Release DEV

- Added a rank normalization step to CRANE prior to centroid calculation
- Added a function for calculating the kendalltau correlation without
  calculating the p-value, the code is taken with modification
  from [SciPy](https://scipy.org/),
  licensed under a [BSD-3-Clause licensed](https://github.com/scipy/scipy/blob/main/LICENSE.txt)
  which is also reproduced in the License file.

## Release 0.2.0

- Moved DIRAC and CRANE classifiers into their own module,
  and updated them to use the scikit-learn interface

## Release 0.1.0

- Moved rank entropy code out of MetworkPy and into separate package
- Initial release
