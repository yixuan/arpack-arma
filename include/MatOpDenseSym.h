#ifndef MATOPDENSESYM_H
#define MATOPDENSESYM_H

#include <armadillo>
#include <stdexcept>
#include "SymmetricLDL.h"

template <typename Scalar>
class MatOpDenseSym
{
private:
    typedef arma::Mat<Scalar> Matrix;
    typedef arma::Col<Scalar> Vector;

    const Matrix mat;
    const int dim_n;
    SymmetricLDL<Scalar> solver;
public:
    MatOpDenseSym(Matrix &mat_) :
        mat(mat_.memptr(), mat_.n_rows, mat_.n_cols, false),
        dim_n(mat_.n_rows)
    {
        if(!mat_.is_square())
            throw std::invalid_argument("MatOpDenseSym: matrix must be square");
    }

    virtual ~MatOpDenseSym() {}

    int rows() { return dim_n; }
    int cols() { return dim_n; }

    // y_out = A * x_in
    void prod(Scalar *x_in, Scalar *y_out)
    {
        Vector x(x_in,  dim_n, false);
        Vector y(y_out, dim_n, false);
        y = mat * x;
    }

    // y_out = A' * x_in
    void trans_prod(Scalar *x_in, Scalar *y_out)
    {
        Vector x(x_in,  dim_n, false);
        Vector y(y_out, dim_n, false);
        y = mat.t() * x;
    }

    // setting real sigma
    void set_shift(Scalar sigma)
    {
        solver.compute(mat - sigma * arma::eye<Matrix>(dim_n, dim_n));
    }

    // y_out = inv(A - sigma * I) * x_in
    void shift_solve(Scalar *x_in, Scalar *y_out)
    {
        Vector x(x_in,  dim_n, false);
        Vector y(y_out, dim_n, false);
        solver.solve(x, y);
    }
};


#endif // MATOPDENSESYM_H
