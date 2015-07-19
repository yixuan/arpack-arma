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
/// #include <SymEigsSolver.h>  // Also includes <MatOp/DenseGenMatProd.h>
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
/// If the users need to define their own matrix-vector multiplication operation
/// class, it should impelement all the public member functions as in DenseGenMatProd.
/// Below is an example.
///
/// \code{.cpp}
/// #include <armadillo>
/// #include <SymEigsSolver.h>
///
/// // M = diag(1, 2, ..., 10)
/// class MyDiagonalTen
/// {
/// public:
///     int rows() { return 10; }
///     int cols() { return 10; }
///     // y_out = M * x_in
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
    ///
    /// Constructor to create a solver object.
    ///
    /// \param op_  Pointer to the matrix operation object.
    /// \param nev_ Number of eigenvalues requested. This should satisfy \f$1\le nev \le n-1\f$,
    ///             where \f$n\f$ is the size of matrix.
    /// \param ncv_ Parameter that controls the convergence speed of the algorithm.
    ///             Typically a larger `ncv_` means faster convergence, but it may
    ///             also result in greater memory use and more matrix operations
    ///             in each iteration. This parameter must satisfy \f$nev < ncv \le n\f$,
    ///             and is advised to take \f$ncv \ge 2\times nev\f$.
    ///
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


///
/// \ingroup EigenSolver
///
/// This class implements the eigen solver for real symmetric matrices using
/// the **shift-and-invert mode**. The background information of the symmetric
/// eigen solver is documented in the SymEigsSolver class. Here we focus on
/// explaining the shift-and-invert mode.
///
/// The shift-and-invert mode is based on the following fact:
/// If \f$\lambda\f$ and \f$x\f$ are a pair of eigenvalue and eigenvector of
/// matrix \f$A\f$, such that \f$Ax=\lambda x\f$, then for any \f$\sigma\f$,
/// we have
/// \f[(A-\sigma I)^{-1}x=\nu x\f]
/// where
/// \f[\nu=\frac{1}{\lambda-\sigma}\f]
/// which indicates that \f$(\nu, x)\f$ is an eigenpair of the matrix
/// \f$(A-\sigma I)^{-1}\f$.
///
/// Therefore, if we pass the matrix operation \f$(A-\sigma I)^{-1}y\f$
/// (rather than \f$Ay\f$) to the eigen solver, then we would get the desired
/// values of \f$\nu\f$, and \f$\lambda\f$ can also be easily obtained by noting
/// that \f$\lambda=\sigma+\nu^{-1}\f$.
///
/// Why do we want to do this?
///
/// The reason is that the algorithm of **ARPACK-Armadillo** (and also **ARPACK**)
/// is good at finding eigenvalues with large magnitude, but may fail in looking
/// for eigenvalues that are close to zero. However, if we really need them, we
/// can set \f$\sigma=0\f$, find the largest eigenvalues of \f$A^{-1}\f$, and then
/// transform back to \f$\lambda\f$, since in this case largest values of \f$\nu\f$
/// implies smallest values of \f$\lambda\f$.
///
/// As a summary, in the shift-and-invert mode, the selection rule will apply to
/// \f$\nu=1/(\lambda-\sigma)\f$ rather than \f$\lambda\f$. So a selection rule
/// of `LARGEST_MAGN` combined with shift \f$\sigma\f$ will find eigenvalues of
/// \f$A\f$ that are closest to \f$\sigma\f$. But note that the eigenvalues()
/// method will always return the eigenvalues in the original problem (i.e.,
/// returning \f$\lambda\f$ rather than \f$\nu\f$), and eigenvectors are the
/// same for both the original problem and the shifted-and-inverted problem.
///
/// \tparam Scalar The element type of the matrix.
///                Currently supported types are `float` and `double`.
/// \tparam SelectionRule An enumeration value indicating the selection rule of
///                       the shifted-and-inverted eigenvalues.
///                       The full list of enumeration values can be found in
///                       SelectionRule.h .
/// \tparam OpType The name of the matrix operation class. See the documentation
///                of the constructor.
///
/// Example code that illustrates the use of the shift-and-invert mode:
///
/// \code{.cpp}
/// #include <armadillo>
/// #include <SymEigsSolver.h>  // Also includes <MatOp/DenseSymShiftSolve.h>
///
/// int main()
/// {
///     // A size-10 diagonal matrix with elements 1, 2, ..., 10
///     arma::mat M(10, 10, arma::fill::zeros);
///     for(int i = 0; i < M.n_rows; i++)
///         M(i, i) = i + 1;
///
///     // Construct matrix operation object using the wrapper class
///     DenseSymShiftSolve<double> op(M);
///
///     // Construct eigen solver object with shift 0
///     // This will find eigenvalues that are closest to 0
///     SymEigsShiftSolver< double, LARGEST_MAGN,
///                         DenseSymShiftSolve<double> > eigs(&op, 3, 6, 0.0);
///
///     eigs.init();
///     eigs.compute();
///     arma::vec evalues = eigs.eigenvalues();
///     evalues.print("Eigenvalues found:");  // Will get (3.0, 2.0, 1.0)
///
///     return 0;
/// }
/// \endcode
///
/// A user-supplied matrix shift-solve operation class should implement all the
/// public member functions as in DenseSymShiftSolve. Below is an example.
///
/// \code{.cpp}
/// #include <armadillo>
/// #include <SymEigsSolver.h>
///
/// // M = diag(1, 2, ..., 10)
/// class MyDiagonalTenShiftSolve
/// {
/// private:
///     double sigma_;
/// public:
///     int rows() { return 10; }
///     int cols() { return 10; }
///     void set_shift(double sigma) { sigma_ = sigma; }
///     // y_out = inv(A - sigma * I) * x_in
///     // inv(A - sigma * I) = diag(1/(1-sigma), 1/(2-sigma), ...)
///     void perform_op(double *x_in, double *y_out)
///     {
///         for(int i = 0; i < rows(); i++)
///         {
///             y_out[i] = x_in[i] / (i + 1 - sigma_);
///         }
///     }
/// };
///
/// int main()
/// {
///     MyDiagonalTenShiftSolve op;
///     // Find three eigenvalues that are closest to 3.14
///     SymEigsShiftSolver<double, LARGEST_MAGN,
///                        MyDiagonalTenShiftSolve> eigs(&op, 3, 6, 3.14);
///     eigs.init();
///     eigs.compute();
///     arma::vec evalues = eigs.eigenvalues();
///     evalues.print("Eigenvalues found:");  // Will get (4.0, 3.0, 2.0)
///
///     return 0;
/// }
/// \endcode
///
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
    ///
    /// Constructor to create a eigen solver object using the shift-and-invert mode.
    ///
    /// \param op_  Pointer to the matrix operation object. This class should implement
    ///             the shift-solve operation of \f$A\f$: calculating \f$(A-\sigma I)^{-1}y\f$
    ///             for any vector \f$y\f$. **ARPACK-Armadillo** provides a wrapper
    ///             class DenseSymShiftSolve to wrap an existing \f$A\f$ stored as
    ///             an **Armadillo** matrix. See the example code in the introduction part above.
    /// \param nev_ Number of eigenvalues requested. This should satisfy \f$1\le nev \le n-1\f$,
    ///             where \f$n\f$ is the size of matrix.
    /// \param ncv_ Parameter that controls the convergence speed of the algorithm.
    ///             Typically a larger `ncv_` means faster convergence, but it may
    ///             also result in greater memory use and more matrix operations
    ///             in each iteration. This parameter must satisfy \f$nev < ncv \le n\f$,
    ///             and is advised to take \f$ncv \ge 2\times nev\f$.
    /// \param sigma_ The value of the shift.
    ///
    SymEigsShiftSolver(OpType *op_, int nev_, int ncv_, Scalar sigma_) :
        SymEigsSolver<Scalar, SelectionRule, OpType>(op_, nev_, ncv_),
        sigma(sigma_)
    {
        this->op->set_shift(sigma);
    }
};



#endif // SYM_EIGS_SOLVER_H
