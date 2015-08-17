#ifndef GEN_EIGS_SOLVER_H
#define GEN_EIGS_SOLVER_H

#include <armadillo>
#include <vector>     // std::vector
#include <cmath>      // std::abs, std::pow, std::sqrt
#include <algorithm>  // std::max, std::min
#include <complex>    // std::complex, std::conj, std::norm
#include <limits>     // std::numeric_limits
#include <stdexcept>  // std::invalid_argument

#include "SelectionRule.h"
#include "LinAlg/UpperHessenbergQR.h"
#include "LinAlg/DoubleShiftQR.h"
#include "LinAlg/UpperHessenbergEigen.h"
#include "MatOp/DenseGenMatProd.h"
#include "MatOp/DenseGenRealShiftSolve.h"


///
/// \ingroup EigenSolver
///
/// This class implements the eigen solver for general real matrices.
///
/// Most of the background information documented in the SymEigsSolver class
/// also applies to the GenEigsSolver class here, except that the eigenvalues
/// and eigenvectors of a general matrix can now be complex-valued.
///
/// \tparam Scalar        The element type of the matrix.
///                       Currently supported types are `float` and `double`.
/// \tparam SelectionRule An enumeration value indicating the selection rule of
///                       the requested eigenvalues, for example `LARGEST_MAGN`
///                       to retrieve eigenvalues with the largest magnitude.
///                       The full list of enumeration values can be found in
///                       SelectionRule.h .
/// \tparam OpType        The name of the matrix operation class. Users could either
///                       use the DenseGenMatProd wrapper class, or define their
///                       own that impelemnts all the public member functions as in
///                       DenseGenMatProd.
///
/// An example that illustrates the usage of GenEigsSolver is give below:
///
/// \code{.cpp}
/// #include <armadillo>
/// #include <GenEigsSolver.h>  // Also includes <MatOp/DenseGenMatProd.h>
///
/// int main()
/// {
///     // We are going to calculate the eigenvalues of M
///     arma::mat M = arma::randu(10, 10);
///
///     // Construct matrix operation object using the wrapper class
///     DenseGenMatProd<double> op(M);
///
///     // Construct eigen solver object, requesting the largest
///     // (in magnitude, or norm) three eigenvalues
///     GenEigsSolver< double, LARGEST_MAGN, DenseGenMatProd<double> > eigs(&op, 3, 6);
///
///     // Initialize and compute
///     eigs.init();
///     int nconv = eigs.compute();
///
///     // Retrieve results
///     arma::cx_vec evalues;
///     if(nconv > 0)
///         evalues = eigs.eigenvalues();
///
///     evalues.print("Eigenvalues found:");
///
///     return 0;
/// }
/// \endcode
///
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
    ///
    /// Constructor to create a solver object.
    ///
    /// \param op_  Pointer to the matrix operation object, which should implement
    ///             the matrix-vector multiplication operation of \f$A\f$:
    ///             calculating \f$Ay\f$ for any vector \f$y\f$. Users could either
    ///             create the object from the DenseGenMatProd wrapper class, or
    ///             define their own that impelemnts all the public member functions
    ///             as in DenseGenMatProd.
    /// \param nev_ Number of eigenvalues requested. This should satisfy \f$1\le nev \le n-2\f$,
    ///             where \f$n\f$ is the size of matrix.
    /// \param ncv_ Parameter that controls the convergence speed of the algorithm.
    ///             Typically a larger `ncv_` means faster convergence, but it may
    ///             also result in greater memory use and more matrix operations
    ///             in each iteration. This parameter must satisfy \f$nev+2 \le ncv \le n\f$,
    ///             and is advised to take \f$ncv \ge 2\cdot nev + 1\f$.
    ///
    GenEigsSolver(OpType *op_, int nev_, int ncv_) :
        op(op_),
        dim_n(op->rows()),
        nev(nev_),
        ncv(ncv_ > dim_n ? dim_n : ncv_),
        nmatop(0),
        niter(0),
        prec(std::pow(std::numeric_limits<Scalar>::epsilon(), Scalar(2.0) / 3))
    {
        if(nev_ < 1 || nev_ > dim_n - 2)
            throw std::invalid_argument("nev must satisfy 1 <= nev <= n - 2, n is the size of matrix");

        if(ncv_ < nev_ + 2 || ncv_ > dim_n)
            throw std::invalid_argument("ncv must satisfy nev + 2 <= ncv <= n, n is the size of matrix");
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
    /// \return A complex-valued vector containing the eigenvalues.
    /// Returned vector type will be `arma::cx_vec` or `arma::cx_fvec`, depending on
    /// the template parameter `Scalar` defined.
    ///
    inline ComplexVector eigenvalues();

    ///
    /// Returning the eigenvectors associated with the converged eigenvalues.
    ///
    /// \param nvec The number of eigenvectors to return.
    ///
    /// \return A complex-valued matrix containing the eigenvectors.
    /// Returned matrix type will be `arma::cx_mat` or `arma::cx_fmat`, depending on
    /// the template parameter `Scalar` defined.
    ///
    inline ComplexMatrix eigenvectors(int nvec);
    ///
    /// Returning all converged eigenvectors.
    ///
    inline ComplexMatrix eigenvectors() { return eigenvectors(nev); }
};


// Implementations
#include "GenEigsSolver_Impl.h"


///
/// \ingroup EigenSolver
///
/// This class implements the eigen solver for general real matrices with
/// a real shift value in the **shift-and-invert mode**. The background
/// knowledge of the shift-and-invert mode can be found in the documentation
/// of the SymEigsShiftSolver class.
///
/// \tparam Scalar        The element type of the matrix.
///                       Currently supported types are `float` and `double`.
/// \tparam SelectionRule An enumeration value indicating the selection rule of
///                       the shifted-and-inverted eigenvalues.
///                       The full list of enumeration values can be found in
///                       SelectionRule.h .
/// \tparam OpType        The name of the matrix operation class. Users could either
///                       use the DenseGenRealShiftSolve wrapper class, or define their
///                       own that impelemnts all the public member functions as in
///                       DenseGenRealShiftSolve.
///
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
    ///
    /// Constructor to create a eigen solver object using the shift-and-invert mode.
    ///
    /// \param op_    Pointer to the matrix operation object. This class should implement
    ///               the shift-solve operation of \f$A\f$: calculating
    ///               \f$(A-\sigma I)^{-1}y\f$ for any vector \f$y\f$. Users could either
    ///               create the object from the DenseGenRealShiftSolve wrapper class, or
    ///               define their own that impelemnts all the public member functions
    ///               as in DenseGenRealShiftSolve.
    /// \param nev_   Number of eigenvalues requested. This should satisfy \f$1\le nev \le n-2\f$,
    ///               where \f$n\f$ is the size of matrix.
    /// \param ncv_   Parameter that controls the convergence speed of the algorithm.
    ///               Typically a larger `ncv_` means faster convergence, but it may
    ///               also result in greater memory use and more matrix operations
    ///               in each iteration. This parameter must satisfy \f$nev+2 \le ncv \le n\f$,
    ///               and is advised to take \f$ncv \ge 2\cdot nev + 1\f$.
    /// \param sigma_ The real-valued shift.
    ///
    GenEigsRealShiftSolver(OpType *op_, int nev_, int ncv_, Scalar sigma_) :
        GenEigsSolver<Scalar, SelectionRule, OpType>(op_, nev_, ncv_),
        sigma(sigma_)
    {
        this->op->set_shift(sigma);
    }
};

#endif // GEN_EIGS_SOLVER_H
