#ifndef GENERALLU_H
#define GENERALLU_H

#include <armadillo>
#include <stdexcept>
#include "LapackWrapperExtra.h"

// LU decomposition of a square matrix
template <typename Scalar = double>
class GeneralLU
{
private:
    typedef arma::Mat<Scalar> Matrix;
    typedef arma::Col<Scalar> Vector;
    typedef arma::Col<int> IntVector;

protected:
    int dim_n;          // size of the matrix
    Matrix mat_fac;     // storing factorization structures
    IntVector vec_fac;  // storing factorization structures
    bool computed;      // whether factorization has been computed
public:
    GeneralLU() :
        dim_n(0), computed(false)
    {}

    GeneralLU(const Matrix &mat) :
        dim_n(mat.n_rows),
        computed(false)
    {
        compute(mat);
    }

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

    void solve(Vector &vec_in, Vector &vec_out)
    {
        if(!computed)
            return;

        vec_out = vec_in;

        int one = 1;
        int info;
        arma::lapack::getrs("N", &dim_n, &one, mat_fac.memptr(), &dim_n,
                            vec_fac.memptr(), vec_out.memptr(), &dim_n, &info);
        if(info < 0)
            throw std::invalid_argument("Lapack getrs: illegal value");
    }
};



#endif // GENERALLU_H
