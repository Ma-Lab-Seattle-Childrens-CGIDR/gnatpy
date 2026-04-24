# Changelog

<!--toc:start-->

- [Changelog](#changelog)
  - [Release DEV](#release-dev)
  - [Release 0.3.0](#release-030)
  - [Release 0.2.0](#release-020)
  - [Release 0.1.0](#release-010)

<!--toc:end-->

## Release DEV

- Updated INFER method to use entropy of DIRAC style pairwise rank comparisons
  rather than ranks directly
- Added wrapper function for finding rank entropy values across a series of
  genes networks and sample groups

## Release 0.3.0

- Added a rank normalization step to CRANE prior to centroid calculation
- Added a function for calculating the Kendall-Tau correlation without
  calculating the p-value, the code is taken with modification
  from [SciPy](https://scipy.org/),
  licensed under a [BSD-3-Clause licensed](https://github.com/scipy/scipy/blob/main/LICENSE.txt)
  which is also reproduced in the License file.
- Added initial implementation of a multi-sample version of the DIRAC
  gene set classification metric. Evaluates the "mismatch" score across
  all samples, and within groups, with a statistic that is the ratio
  of these two values.
- Added initial implementation of a multi-sample version of CRANE
  gene set classification metric, which compares the distance to
  an overall rank centroid to that of the within group rank centroids

## Release 0.2.0

- Moved DIRAC and CRANE classifiers into their own module,
  and updated them to use the scikit-learn interface

## Release 0.1.0

- Moved rank entropy code out of MetworkPy and into separate package
- Initial release
