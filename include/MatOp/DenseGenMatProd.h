#ifndef DENSEGENMATPROD_H
#define DENSEGENMATPROD_H

#include <armadillo>

template <typename Scalar>
class DenseGenMatProd
{
private:
    typedef arma::Mat<Scalar> Matrix;
    typedef arma::Col<Scalar> Vector;

    const Matrix mat;

public:
    DenseGenMatProd(Matrix &mat_) :
        mat(mat_.memptr(), mat_.n_rows, mat_.n_cols, false)
    {}

    int rows() { return mat.n_rows; }
    int cols() { return mat.n_cols; }

    // y_out = A * x_in
    void perform_op(Scalar *x_in, Scalar *y_out)
    {
        Vector x(x_in, mat.n_cols, false);
        Vector y(y_out, mat.n_rows, false);
        y = mat * x;
    }
};


#endif // DENSEGENMATPROD_H
