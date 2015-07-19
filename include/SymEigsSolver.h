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


///
/// \defgroup EigenSolver Eigen Solvers
///
/// Eigen solvers for different types of problems.
///

///
/// \ingroup EigenSolver
///
/// This class implements the eigen solver for real symmetric matrices.
///
/// \tparam Scalar The element type of the matrix.
///                Currently supported types are `float` and `double`.
/// \tparam SelectionRule An enumeration value indicating the selection rule of
///                       the requested eigenvalues, for example `LARGEST_MAGN`
///                       to retrieve eigenvalues with the largest magnitude.
///                       The full list of enumeration values can be found in
///                       SelectionRule.h .
/// \tparam OpType The name of the matrix operation class. See explanations below.
///
/// **ARPACK-Armadillo** is designed to calculate a specified number (\f$k\f$)
/// of eigenvalues of a large square matrix (\f$A\f$). Usually \f$k\f$ is much
/// less than the size of the matrix (\f$n\f$), so that only a few eigenvalues
/// and eigenvectors are computed.
///
/// This class implements the eigen solver of a real symmetric matrix, but
/// rather than providing the whole matrix, the algorithm only requires the
/// matrix-vector multiplication operation of \f$A\f$. Therefore, users of
/// this solver need to supply a class that computes the result of \f$Av\f$
/// for any given vector \f$v\f$. The name of this class should be given to
/// the template parameter `OpType`, and instance of this class passed to
/// the constructor of SymEigsSolver.
///
/// If the matrix \f$A\f$ is already stored as a matrix object in **Armadillo**,
/// for example `arma::mat`, then there is an easy way to construct such
/// matrix operation class, by using the built-in wrapper class DenseGenMatProd
/// which wraps an existing matrix object in **Armadillo**. This is also the
/// default choice of SymEigsSolver. See the example below.
///
/// \code{.cpp}
/// #include <armadillo>
/// #include <SymEigsSolver.h>
/// #include <MatOp/DenseGenMatProd.h>
///
/// int main()
/// {
///     // We are going to calculate the eigenvalues of M
///     arma::mat A = arma::randu(10, 10);
///     arma::mat M = A + A.t();
///
///     // Construct matrix operation object using the wrapper class DenseGenMatProd
///     DenseGenMatProd<double> op(M);
///
///     // Construct eigen solver object, requesting the largest three eigenvalues
///     SymEigsSolver< double, LARGEST_ALGE, DenseGenMatProd<double> > eigs(&op, 3, 6);
///
///     // Initialize and compute
///     eigs.init();
///     int nconv = eigs.compute();
///
///     // Retrieve results
///     arma::vec evalues;
///     if(nconv > 0)
///         evalues = eigs.eigenvalues();
///
///     evalues.print("Eigenvalues found:");
///
///     return 0;
/// }
/// \endcode
///
/// If the users need to define their own matrix operation class, it should
/// impelement all the public member functions as in DenseGenMatProd. Below is
/// a simple example.
///
/// \code{.cpp}
/// #include <armadillo>
/// #include <SymEigsSolver.h>
///
/// // A size-10 diagonal matrix with elements 1, 2, ..., 10
/// class MyDiagonalTen
/// {
/// public:
///     int rows() { return 10; }
///     int cols() { return 10; }
///     void perform_op(double *x_in, double *y_out)
///     {
///         for(int i = 0; i < rows(); i++)
///         {
///             y_out[i] = x_in[i] * (i + 1);
///         }
///     }
/// };
///
/// int main()
/// {
///     MyDiagonalTen op;
///     SymEigsSolver<double, LARGEST_ALGE, MyDiagonalTen> eigs(&op, 3, 6);
///     eigs.init();
///     eigs.compute();
///     arma::vec evalues = eigs.eigenvalues();
///     evalues.print("Eigenvalues found:");
///
///     return 0;
/// }
/// \endcode
///


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
    /// Providing the initial residual vector for the algorithm.
    ///
    /// \param init_resid Pointer to the initial residual vector.
    ///
    /// **ARPACK-Armadillo** (and also **ARPACK**) uses an iterative algorithm
    /// to find eigenvalues. This function allows the user to provide the initial
    /// residual vector.
    ///
    inline void init(Scalar *init_resid);

    ///
    /// Providing a random initial residual vector.
    ///
    /// This overloaded function generates a random initial residual vector
    /// for the algorithm. Elements in the vector follow independent Uniform(-0.5, 0.5)
    /// distributions.
    ///
    inline void init();

    ///
    /// Conducting the major computation procedure.
    ///
    /// \param maxit Maximum number of iterations allowed in the algorithm.
    /// \param tol Precision parameter for the calculated eigenvalues.
    ///
    /// \return Number of converged eigenvalues.
    ///
    inline int compute(int maxit = 1000, Scalar tol = 1e-10);

    ///
    /// Returning the number of iterations used in the computation.
    ///
    inline int num_iterations() { return niter; }

    ///
    /// Returning the number of matrix operations used in the computation.
    ///
    inline int num_operations() { return nmatop; }

    ///
    /// Returning the converged eigenvalues.
    ///
    /// \return A vector containing the eigenvalues.
    ///
    inline Vector eigenvalues();

    ///
    /// Returning the eigenvectors associated with the converged eigenvalues.
    ///
    /// \return A matrix containing the eigenvectors.
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
