### Control Matrices
# Descripition
This is a library for creating and handling many of the matrices that one encounter in
common control problems. In particular, it can handle MIMO systems and the block matrices that
comes with those.

Some of the capabilites include (but are not limited to):
* Make Hankel and Toeplitz matrices from a given block vector.
* Make controllability and observability matrices from a given state space.
* Other utility functions for setting up least squares problems and other matrix equations that are common when dealing with linear state space representations.
# Examples
Included are some examples using the library in different ways.

* Ho-Kalman.ipynb
  * System identification using the Ho-Kalman method, where a similar state space is estimated
using the impulse response of the true system.
* Deterministic Subspace ID - Method I
  * System identification using subspace methods. First, the A and C matrices are estimated from
input-output data using orthogonal projections, whereafter the B and D matrices are found using
linear least squares.
* Deterministic Subspace ID - Method I
  * System identification using subspace methods. First, the state space variables are estimated
using oblique projections, whereafter the A, B, C and D matrices are found using linear least squares.
