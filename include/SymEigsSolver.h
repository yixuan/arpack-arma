#ifndef SYM_EIGS_SOLVER_H
#define SYM_EIGS_SOLVER_H

#include <armadillo>
#include <vector>
#include <algorithm>
#include <cmath>
#include <utility>
#include <stdexcept>

#include "SelectionRule.h"
#include "LinAlg/UpperHessenbergQR.h"
#include "MatOp/DenseGenMatProd.h"
#include "MatOp/DenseSymShiftSolve.h"


template < typename Scalar = double,
           int SelectionRule = LARGEST_MAGN,
           typename OpType = DenseGenMatProd<double> >
class SymEigsSolver
{
private:
    typedef arma::Mat<Scalar> Matrix;
    typedef arma::Col<Scalar> Vector;
    typedef arma::uvec BoolVector;
    typedef std::pair<Scalar, int> SortPair;

protected:
    OpType *op;           // object to conduct matrix operation,
                          // e.g. matrix-vector product

private:
    const int dim_n;      // dimension of matrix A

protected:
    const int nev;        // number of eigenvalues requested

private:
    const int ncv;        // number of ritz values
    int nmatop;           // number of matrix operations called
    int niter;            // number of restarting iterations

    Matrix fac_V;         // V matrix in the Arnoldi factorization
    Matrix fac_H;         // H matrix in the Arnoldi factorization
    Vector fac_f;         // residual in the Arnoldi factorization

protected:
    Vector ritz_val;      // ritz values

private:
    Matrix ritz_vec;      // ritz vectors
    BoolVector ritz_conv; // indicator of the convergence of ritz values

    const Scalar prec;    // precision parameter used to test convergence
                          // prec = epsilon^(2/3)
                          // epsilon is the machine precision,
                          // e.g. ~= 1e-16 for the "double" type

    // Arnoldi factorization starting from step-k
    inline void factorize_from(int from_k, int to_m, const Vector &fk);

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
    SymEigsSolver(OpType *op_, int nev_, int ncv_) :
        op(op_),
        dim_n(op->rows()),
        nev(nev_),
        ncv(ncv_ > dim_n ? dim_n : ncv_),
        nmatop(0),
        niter(0),
        prec(std::pow(std::numeric_limits<Scalar>::epsilon(), Scalar(2.0 / 3)))
    {
        if(nev_ < 1 || nev_ > dim_n - 1)
            throw std::invalid_argument("nev must satisfy 1 <= nev <= n - 1, n is the size of matrix");

        if(ncv_ <= nev_ || ncv_ > dim_n)
            throw std::invalid_argument("ncv must satisfy nev < ncv <= n, n is the size of matrix");
    }

    ///
    /// Initialization and clean-up
    ///
    inline void init(Scalar *init_resid);

    ///
    /// Initialization with random initial coefficients
    ///
    inline void init();

    ///
    /// Compute Ritz pairs and return the number of converged eigenvalues
    ///
    inline int compute(int maxit = 1000, Scalar tol = 1e-10);

    ///
    /// Return the number of restarting iterations
    ///
    inline int num_iterations() { return niter; }

    ///
    /// Return the number of matrix operations
    ///
    inline int num_operations() { return nmatop; }

    ///
    /// Return converged eigenvalues
    ///
    inline Vector eigenvalues();

    ///
    /// Return converged eigenvectors
    ///
    inline Matrix eigenvectors();
};


// Implementations
#include "SymEigsSolver_Impl.h"


template <typename Scalar = double,
          int SelectionRule = LARGEST_MAGN,
          typename OpType = DenseSymShiftSolve<double> >
class SymEigsShiftSolver: public SymEigsSolver<Scalar, SelectionRule, OpType>
{
private:
    typedef arma::Col<Scalar> Vector;

    Scalar sigma;

    // First transform back the ritz values, and then sort
    inline void sort_ritzpair()
    {
        Vector ritz_val_org = Scalar(1.0) / this->ritz_val.head(this->nev) + sigma;
        this->ritz_val.head(this->nev) = ritz_val_org;
        SymEigsSolver<Scalar, SelectionRule, OpType>::sort_ritzpair();
    }
public:
    SymEigsShiftSolver(OpType *op_, int nev_, int ncv_, Scalar sigma_) :
        SymEigsSolver<Scalar, SelectionRule, OpType>(op_, nev_, ncv_),
        sigma(sigma_)
    {
        this->op->set_shift(sigma);
    }
};



#endif // SYM_EIGS_SOLVER_H
