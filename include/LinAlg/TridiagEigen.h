#ifndef TRIDIAG_EIGEN_H
#define TRIDIAG_EIGEN_H

#include <armadillo>
#include <stdexcept>
#include "LapackWrapperExtra.h"

///
/// \ingroup LinearAlgebra
///
/// Calculate the eigenvalues and eigenvectors of a symmetric tridiagonal matrix.
///
/// \tparam Scalar The element type of the matrix.
/// Currently supported types are `float` and `double`.
///
/// This class is a wrapper of the Lapack functions `_steqr`.
///
template <typename Scalar = double>
class TridiagEigen
{
private:
    typedef arma::Mat<Scalar> Matrix;
    typedef arma::Col<Scalar> Vector;

    int n;
    Vector main_diag;     // Main diagonal elements of the matrix
    Vector sub_diag;      // Sub-diagonal elements of the matrix
    Matrix evecs;         // To store eigenvectors

    bool computed;

public:
    ///
    /// Default constructor. Computation can
    /// be performed later by calling the compute() method.
    ///
    TridiagEigen() :
        n(0), computed(false)
    {}

    ///
    /// Constructor to create an object that calculates the eigenvalues
    /// and eigenvectors of a symmetric tridiagonal matrix `mat`.
    ///
    /// \param mat Matrix type can be `arma::mat` or `arma::fmat`, depending on
    /// the template parameter `Scalar` defined.
    /// Only the main diagonal and the lower sub-diagonal parts of
    /// the matrix are used.
    ///
    TridiagEigen(const Matrix &mat) :
        n(mat.n_rows), computed(false)
    {
        compute(mat);
    }

    ///
    /// Compute the eigenvalue decomposition of a symmetric tridiagonal matrix.
    ///
    /// \param mat Matrix type can be `arma::mat` or `arma::fmat`, depending on
    /// the template parameter `Scalar` defined.
    /// Only the main diagonal and the lower sub-diagonal parts of
    /// the matrix are used.
    ///
    void compute(const Matrix &mat)
    {
        if(!mat.is_square())
            throw std::invalid_argument("TridiagEigen: matrix must be square");

        n = mat.n_rows;
        main_diag = mat.diag();
        sub_diag = mat.diag(-1);
        evecs.set_size(n, n);

        char compz = 'I';
        int lwork = 1 + 4 * n + n * n;
        Scalar *work = new Scalar[lwork];
        int liwork = 3 + 5 * n;
        int *iwork = new int[liwork];
        int info;

        arma::lapack::stedc(&compz, &n, main_diag.memptr(), sub_diag.memptr(),
                            evecs.memptr(), &n, work, &lwork, iwork, &liwork, &info);

        delete [] work;
        delete [] iwork;

        if(info < 0)
            throw std::invalid_argument("Lapack stedc: illegal value");
        if(info > 0)
            throw std::logic_error("Lapack stedc: failed to compute all the eigenvalues");

        computed = true;
    }

    ///
    /// Retrieve the eigenvalues.
    ///
    /// \return Returned vector type will be `arma::vec` or `arma::fvec`, depending on
    /// the template parameter `Scalar` defined.
    ///
    Vector eigenvalues()
    {
        if(!computed)
            throw std::logic_error("TridiagEigen: need to call compute() first");

        // After calling compute(), main_diag will contain the eigenvalues.
        return main_diag;
    }

    ///
    /// Retrieve the eigenvectors.
    ///
    /// \return Returned matrix type will be `arma::mat` or `arma::fmat`, depending on
    /// the template parameter `Scalar` defined.
    ///
    Matrix eigenvectors()
    {
        if(!computed)
            throw std::logic_error("TridiagEigen: need to call compute() first");

        return evecs;
    }
};



#endif // TRIDIAG_EIGEN_H
