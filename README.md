# ARPACK-Armadillo

**ARPACK-Armadillo** is a redesign of the [ARPACK](http://www.caam.rice.edu/software/ARPACK/)
software for large scale eigenvalue problems, built on top of
[Armadillo](http://arma.sourceforge.net/), an open source C++ linear algebra library.

**ARPACK-Armadillo** is implemented as a header-only C++ library which relies on
the BLAS and LAPACK libraries. Therefore program that uses **ARPACK-Armadillo**
should also link to those two libraries.

## Common Usage

**ARPACK-Armadillo** is designed to calculate a specified number (`k`) of eigenvalues
of a large square matrix (`A`). Usually `k` is much less than the size of matrix
(`n`), so that only a few eigenvalues and eigenvectors are computed, which
in general is more efficient than calculating the whole spectral decomposition.
Users can choose eigenvalue selection rules to pick up the eigenvalues of interest,
such as the largest `k` eigenvalues, or eigenvalues with largest real parts,
etc.

To use the eigen solvers in this library, the user does not need to directly
provide the whole matrix, but instead, the algorithm only requires certain operations
defined on `A`, and in the basic setting, it is simply the matrix-vector
multiplication. Therefore, if the matrix-vector product `A * x` can be computed
efficiently, which is the case when `A` is sparse, **ARPACK-Armadillo**
will be very powerful for large scale eigenvalue problems.

There are two major steps to use the **ARPACK-Armadillo** library:

1. Define a class that implements a certain matrix operation, for example the
matrix-vector multiplication `y = A * x` or the shift-solve operation
`y = inv(A - σ * I) * x`. **ARPACK-Armadillo** has defined a number of
helper classes to quickly create such operations from a matrix object.
See the documentation of
[DenseGenMatProd](http://yixuan.cos.name/arpack-arma/doc/classDenseGenMatProd.html),
[DenseSymShiftSolve](http://yixuan.cos.name/arpack-arma/doc/classDenseSymShiftSolve.html), etc.
2. Create an object of one of the eigen solver classes, for example
[SymEigsSolver](http://yixuan.cos.name/arpack-arma/doc/classSymEigsSolver.html)
for symmetric matrices, and
[GenEigsSolver](http://yixuan.cos.name/arpack-arma/doc/classGenEigsSolver.html)
for general matrices. Member functions
of this object can then be called to conduct the computation and retrieve the
eigenvalues and/or eigenvectors.

Below is a list of the available eigen solvers in **ARPACK-Armadillo**:
- [SymEigsSolver](http://yixuan.cos.name/arpack-arma/doc/classSymEigsSolver.html):
for real symmetric matrices
- [GenEigsSolver](http://yixuan.cos.name/arpack-arma/doc/classGenEigsSolver.html):
for general real matrices
- [SymEigsShiftSolver](http://yixuan.cos.name/arpack-arma/doc/classSymEigsShiftSolver.html):
for real symmetric matrices using the shift-and-invert mode
- [GenEigsRealShiftSolver](http://yixuan.cos.name/arpack-arma/doc/classGenEigsRealShiftSolver.html):
for general real matrices using the shift-and-invert mode,
with a real-valued shift

## Examples

Below is an example that demonstrates the use of the eigen solver for symmetric
matrices.

```cpp
#include <armadillo>
#include <SymEigsSolver.h>  // Also includes <MatOp/DenseGenMatProd.h>

int main()
{
    // We are going to calculate the eigenvalues of M
    arma::mat A = arma::randu(10, 10);
    arma::mat M = A + A.t();

    // Construct matrix operation object using the wrapper class DenseGenMatProd
    DenseGenMatProd<double> op(M);

    // Construct eigen solver object, requesting the largest three eigenvalues
    SymEigsSolver< double, LARGEST_ALGE, DenseGenMatProd<double> > eigs(&op, 3, 6);

    // Initialize and compute
    eigs.init();
    int nconv = eigs.compute();

    // Retrieve results
    arma::vec evalues;
    if(nconv > 0)
     evalues = eigs.eigenvalues();

    evalues.print("Eigenvalues found:");

    return 0;
}
```

And here is an example for user-supplied matrix operation class.

```cpp
#include <armadillo>
#include <SymEigsSolver.h>

// M = diag(1, 2, ..., 10)
class MyDiagonalTen
{
public:
    int rows() { return 10; }
    int cols() { return 10; }
    // y_out = M * x_in
    void perform_op(double *x_in, double *y_out)
    {
        for(int i = 0; i < rows(); i++)
        {
            y_out[i] = x_in[i] * (i + 1);
        }
    }
};

int main()
{
    MyDiagonalTen op;
    SymEigsSolver<double, LARGEST_ALGE, MyDiagonalTen> eigs(&op, 3, 6);
    eigs.init();
    eigs.compute();
    arma::vec evalues = eigs.eigenvalues();
    evalues.print("Eigenvalues found:");

    return 0;
}
```

## Shift-and-invert Mode

When we want to find eigenvalues that are closest to a number `σ`,
for example to find the smallest eigenvalues of a positive definite matrix
(in which case `σ=0`), it is advised to use the shift-and-invert mode
of eigen solvers.

In the shift-and-invert mode, selection rules are applied to `1/(λ - σ)`
rather than `λ`, where `λ` are eigenvalues of `A`.
To use this mode, users need to define the shift-solve matrix operation. See
the documentation of
[SymEigsShiftSolver](http://yixuan.cos.name/arpack-arma/doc/classSymEigsShiftSolver.html)
for details.

## Documentation

[This page](http://yixuan.cos.name/arpack-arma/doc/) contains the documentation
of **ARPACK-Armadillo** generated by [Doxygen](http://www.doxygen.org/),
including all the background knowledge, example code and class APIs.

## License

**ARPACK-Armadillo** is an open source project licensed under
[MPL2](https://www.mozilla.org/MPL/2.0/), the same license used by **Armadillo**.
