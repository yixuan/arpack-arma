#ifndef GENERAL_LU_H
#define GENERAL_LU_H

#include <armadillo>
#include <stdexcept>
#include "LapackWrapperExtra.h"

///
/// Perform the LU decomposition of a square matrix.
///
/// \tparam Scalar The element type of the matrix.
/// Currently supported types are `float` and `double`.
///
/// This class is a wrapper of the Lapack functions `_getrf` and `_getrs`.
///
template <typename Scalar = double>
class GeneralLU
{
private:
    typedef arma::Mat<Scalar> Matrix;
    typedef arma::Col<Scalar> Vector;
    typedef arma::Col<int> IntVector;

    int dim_n;          // size of the matrix
    Matrix mat_fac;     // storing factorization structures
    IntVector vec_fac;  // storing factorization structures
    bool computed;      // whether factorization has been computed

public:
    ///
    /// Default constructor to create an object that stores the
    /// LU decomposition of a square matrix. Factorization can
    /// be performed later by calling the compute() method.
    ///
    GeneralLU() :
        dim_n(0), computed(false)
    {}

    ///
    /// Constructor to create an object that performs and stores the
    /// LU decomposition of a square matrix `mat`.
    ///
    /// Matrix type can be `arma::mat` or `arma::fmat`, depending on
    /// the template parameter `Scalar` defined.
    ///
    GeneralLU(const Matrix &mat) :
        dim_n(mat.n_rows),
        computed(false)
    {
        compute(mat);
    }

    ///
    /// Conduct the LU factorization of a square matrix.
    ///
    /// Matrix type can be `arma::mat` or `arma::fmat`, depending on
    /// the template parameter `Scalar` defined.
    ///
    void compute(const Matrix &mat)
    {
        if(!mat.is_square())
            throw std::invalid_argument("GeneralLU: matrix must be square");

        dim_n = mat.n_rows;
        mat_fac = mat;
        vec_fac.set_size(dim_n);

        int info;
        arma::lapack::getrf(&dim_n, &dim_n, mat_fac.memptr(), &dim_n,
                            vec_fac.memptr(), &info);

        if(info < 0)
            throw std::invalid_argument("Lapack getrf: illegal value");
        if(info > 0)
            throw std::logic_error("GeneralLU: matrix is singular");

        computed = true;
    }

    ///
    /// Use the computed LU factorization to solve linear equation \f$Ax=b\f$,
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
        char no_trans = 'N';
        int info;
        arma::lapack::getrs(&no_trans, &dim_n, &one, mat_fac.memptr(), &dim_n,
                            vec_fac.memptr(), vec_out.memptr(), &dim_n, &info);
        if(info < 0)
            throw std::invalid_argument("Lapack getrs: illegal value");
    }
};



#endif // GENERAL_LU_H
