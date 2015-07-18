#ifndef SYMMETRIC_LDL_H
#define SYMMETRIC_LDL_H

#include <armadillo>
#include <stdexcept>
#include "LapackWrapperExtra.h"

///
/// Perform LDL decomposition of a symmetric (possibly indefinite) matrix.
///
/// \tparam Scalar The element type of the matrix.
/// Currently supported types are `float` and `double`.
///
/// This class is a wrapper of the Lapack functions `_sytrf` and `_sytrs`.
///
template <typename Scalar = double>
class SymmetricLDL
{
private:
    typedef arma::Mat<Scalar> Matrix;
    typedef arma::Col<Scalar> Vector;
    typedef arma::Col<int> IntVector;

    int dim_n;          // size of the matrix
    char mat_uplo;      // whether using lower triangle or upper triangle
    Matrix mat_fac;     // storing factorization structures
    IntVector vec_fac;  // storing factorization structures
    bool computed;      // whether factorization has been computed

public:
    ///
    /// Default constructor to create an object that stores the
    /// LDL decomposition of a symmetric matrix. Factorization can
    /// be performed later by calling the compute() method.
    ///
    SymmetricLDL() :
        dim_n(0), mat_uplo('L'), computed(false)
    {}

    ///
    /// Constructor to create an object that performs and stores the
    /// LDL decomposition of a symmetric matrix `mat`.
    ///
    /// \param uplo 'L' to indicate using the lower triangular part of
    ///             the matrix, and 'U' for upper triangular part.
    ///
    /// Matrix type can be `arma::mat` or `arma::fmat`, depending on
    /// the template parameter `Scalar` defined.
    ///
    SymmetricLDL(const Matrix &mat, const char uplo = 'L') :
        dim_n(mat.n_rows),
        mat_uplo(uplo),
        computed(false)
    {
        compute(mat, uplo);
    }

    ///
    /// Conduct the LDL factorization of a symmetric matrix.
    ///
    /// \param uplo 'L' to indicate using the lower triangular part of
    ///             the matrix, and 'U' for upper triangular part.
    ///
    /// Matrix type can be `arma::mat` or `arma::fmat`, depending on
    /// the template parameter `Scalar` defined.
    ///
    void compute(const Matrix &mat, const char uplo = 'L')
    {
        if(!mat.is_square())
            throw std::invalid_argument("SymmetricLDL: matrix must be square");

        dim_n = mat.n_rows;
        mat_uplo = (uplo == 'L' ? 'L' : 'U');  // force to be one of 'L' and 'U'
        mat_fac = mat;
        vec_fac.set_size(dim_n);

        Scalar lwork_query;
        int lwork = -1, info;
        arma::lapack::sytrf(&mat_uplo, &dim_n, mat_fac.memptr(), &dim_n,
                            vec_fac.memptr(), &lwork_query, &lwork, &info);
        lwork = int(lwork_query);

        Scalar *work = new Scalar[lwork];
        arma::lapack::sytrf(&mat_uplo, &dim_n, mat_fac.memptr(), &dim_n,
                            vec_fac.memptr(), work, &lwork, &info);
        delete [] work;

        if(info < 0)
            throw std::invalid_argument("Lapack sytrf: illegal value");
        if(info > 0)
            throw std::logic_error("SymmetricLDL: matrix is singular");

        computed = true;
    }

    ///
    /// Use the computed LDL factorization to solve linear equation \f$Ax=b\f$,
    /// where \f$A\f$ is the matrix factorized.
    ///
    /// \param vec_in The vector \f$b\f$.
    /// \param vec_out The vector \f$x\f$ to be solved, which will be overwritten
    ///                by the calculated solution.
    ///
    /// Vector type can be `arma::vec` or `arma::fvec`, depending on
    /// the template parameter `Scalar` defined.
    ///
    void solve(Vector &vec_in, Vector &vec_out)
    {
        if(!computed)
            return;

        vec_out = vec_in;

        int one = 1;
        int info;
        arma::lapack::sytrs(&mat_uplo, &dim_n, &one, mat_fac.memptr(), &dim_n,
                            vec_fac.memptr(), vec_out.memptr(), &dim_n, &info);
        if(info < 0)
            throw std::invalid_argument("Lapack sytrs: illegal value");
    }
};



#endif // SYMMETRIC_LDL_H
