#ifndef SYMMETRICLDL_H
#define SYMMETRICLDL_H

#include <armadillo>
#include "LapackWrapperExtra.h"

// LDL decomposition of an symmetric (but possibly indefinite) matrix
template <typename Scalar = double>
class SymmetricLDL
{
private:
    typedef arma::Mat<Scalar> Matrix;
    typedef arma::Col<Scalar> Vector;
    typedef arma::Col<int> IntVector;

protected:
    int dim_n;          // size of the matrix
    char mat_uplo;      // whether using lower triangle or upper triangle
    Matrix mat_fac;     // storing factorization structures
    IntVector vec_fac;  // storing factorization structures
    bool computed;      // whether factorization has been computed
public:
    SymmetricLDL() :
        dim_n(0), mat_uplo('L'), computed(false)
    {}

    SymmetricLDL(const Matrix &mat, const char uplo = 'L') :
        dim_n(mat.n_rows),
        mat_uplo(uplo),
        computed(false)
    {
        compute(mat, uplo);
    }

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



#endif // SYMMETRICLDL_H
