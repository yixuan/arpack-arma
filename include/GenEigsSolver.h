#ifndef GEN_EIGS_SOLVER_H
#define GEN_EIGS_SOLVER_H

#include <armadillo>
#include <vector>
#include <algorithm>
#include <cmath>
#include <complex>
#include <utility>
#include <stdexcept>

#include "SelectionRule.h"
#include "LinAlg/UpperHessenbergQR.h"
#include "LinAlg/UpperHessenbergEigen.h"
#include "MatOp/DenseGenMatProd.h"
#include "MatOp/DenseGenRealShiftSolve.h"


template < typename Scalar = double,
           int SelectionRule = LARGEST_MAGN,
           typename OpType = DenseGenMatProd<double> >
class GenEigsSolver
{
private:
    typedef arma::Mat<Scalar> Matrix;
    typedef arma::Col<Scalar> Vector;
    typedef arma::uvec BoolVector;

    typedef std::complex<Scalar> Complex;
    typedef arma::Mat<Complex> ComplexMatrix;
    typedef arma::Col<Complex> ComplexVector;

    typedef std::pair<Complex, int> SortPair;

protected:
    OpType *op;             // object to conduct matrix operation,
                            // e.g. matrix-vector product

private:
    const int dim_n;        // dimension of matrix A

protected:
    const int nev;          // number of eigenvalues requested

private:
    const int ncv;          // number of ritz values
    int nmatop;             // number of matrix operations called
    int niter;              // number of restarting iterations

    Matrix fac_V;           // V matrix in the Arnoldi factorization
    Matrix fac_H;           // H matrix in the Arnoldi factorization
    Vector fac_f;           // residual in the Arnoldi factorization

protected:
    ComplexVector ritz_val; // ritz values

private:
    ComplexMatrix ritz_vec; // ritz vectors
    BoolVector ritz_conv;   // indicator of the convergence of ritz values

    const Scalar prec;      // precision parameter used to test convergence
                            // prec = epsilon^(2/3)
                            // epsilon is the machine precision,
                            // e.g. ~= 1e-16 for the "double" type

    // Arnoldi factorization starting from step-k
    inline void factorize_from(int from_k, int to_m, const Vector &fk);

    static bool is_complex(Complex v, Scalar eps)
    {
        return std::abs(v.imag()) > eps;
    }

    static bool is_conj(Complex v1, Complex v2, Scalar eps)
    {
        return std::abs(v1 - std::conj(v2)) < eps;
    }

    // Implicitly restarted Arnoldi factorization
    inline void restart(int k);

    // Calculate the number of converged Ritz values
    inline int num_converged(Scalar tol);

    // Return the adjusted nev for restarting
    inline int nev_adjusted(int nconv);

    // Retrieve and sort ritz values and ritz vectors
    inline void retrieve_ritzpair();

protected:
    // Sort the first nev Ritz pairs in decreasing magnitude order
    // This is used to return the final results
    inline virtual void sort_ritzpair();

public:
    GenEigsSolver(OpType *op_, int nev_, int ncv_) :
        op(op_),
        dim_n(op->rows()),
        nev(nev_),
        ncv(ncv_ > dim_n ? dim_n : ncv_),
        nmatop(0),
        niter(0),
        prec(std::pow(std::numeric_limits<Scalar>::epsilon(), Scalar(2.0 / 3)))
    {
        if(nev_ < 1 || nev_ > dim_n - 2)
            throw std::invalid_argument("nev must satisfy 1 <= nev <= n - 2, n is the size of matrix");

        if(ncv_ < nev_ + 2 || ncv_ > dim_n)
            throw std::invalid_argument("ncv must satisfy nev + 2 <= ncv <= n, n is the size of matrix");
    }

    // Initialization and clean-up
    inline void init(Scalar *init_resid);

    // Initialization with random initial coefficients
    inline void init();

    // Compute Ritz pairs and return the number of converged eigenvalues
    inline int compute(int maxit = 1000, Scalar tol = 1e-10);

    // Return the number of restarting iterations
    inline int num_iterations() { return niter; }

    // Return the number of matrix operations
    inline int num_operations() { return nmatop; }

    // Return converged eigenvalues
    inline ComplexVector eigenvalues();

    // Return converged eigenvectors
    inline ComplexMatrix eigenvectors();
};


// Implementations
#include "GenEigsSolver_Impl.h"


template <typename Scalar = double,
          int SelectionRule = LARGEST_MAGN,
          typename OpType = DenseGenRealShiftSolve<double> >
class GenEigsRealShiftSolver: public GenEigsSolver<Scalar, SelectionRule, OpType>
{
private:
    typedef arma::Col<Scalar> Vector;
    typedef std::complex<Scalar> Complex;
    typedef arma::Col<Complex> ComplexVector;

    Scalar sigma;

    // First transform back the ritz values, and then sort
    void sort_ritzpair()
    {
        ComplexVector ritz_val_org = Scalar(1.0) / this->ritz_val.head(this->nev) + sigma;
        this->ritz_val.head(this->nev) = ritz_val_org;
        GenEigsSolver<Scalar, SelectionRule, OpType>::sort_ritzpair();
    }
public:
    GenEigsRealShiftSolver(OpType *op_, int nev_, int ncv_, Scalar sigma_) :
        GenEigsSolver<Scalar, SelectionRule, OpType>(op_, nev_, ncv_),
        sigma(sigma_)
    {
        this->op->set_shift(sigma);
    }
};

#endif // GEN_EIGS_SOLVER_H
