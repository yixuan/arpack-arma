#ifndef MATOPDENSE_H
#define MATOPDENSE_H

#include <armadillo>
#include <cmath>
#include <complex>
#include <limits>
#include "MatOp.h"

template <typename Scalar>
class MatOpDense:
    public MatOpWithTransProd<Scalar>,
    public MatOpWithComplexShiftSolve<Scalar>
{
private:
    typedef arma::Mat<Scalar> Matrix;
    typedef arma::Col<Scalar> Vector;

    const Matrix mat;
    bool sigma_is_real;

    // shift solve for real sigma
    void real_shift_solve(Scalar *x_in, Scalar *y_out)
    {
        // TODO
    }

    // shift solve for complex sigma
    void complex_shift_solve(Scalar *x_in, Scalar *y_out)
    {
        // TODO
    }
public:
    MatOpDense(Matrix &mat_) :
        mat(mat_.memptr(), mat_.n_rows, mat_.n_cols, false),
        sigma_is_real(false)
    {}

    virtual ~MatOpDense() {}

    int rows() { return mat.n_rows; }
    int cols() { return mat.n_cols; }

    // y_out = A * x_in
    void prod(Scalar *x_in, Scalar *y_out)
    {
        Vector x(x_in, mat.n_cols, false);
        Vector y(y_out, mat.n_rows, false);
        y = mat * x;
    }

    // y_out = A' * x_in
    void trans_prod(Scalar *x_in, Scalar *y_out)
    {
        Vector x(x_in, mat.n_rows, false);
        Vector y(y_out, mat.n_cols, false);
        y = mat.t() * x;
    }

    // setting complex shift
    void set_shift(Scalar sigmar, Scalar sigmai)
    {
        // TODO
    }

    // y_out = inv(A - sigma * I) * x_in
    void shift_solve(Scalar *x_in, Scalar *y_out)
    {
        if(sigma_is_real)
            real_shift_solve(x_in, y_out);
        else
            complex_shift_solve(x_in, y_out);
    }
};


#endif // MATOPDENSE_H
