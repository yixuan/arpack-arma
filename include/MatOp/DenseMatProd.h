#ifndef DENSEMATPROD_H
#define DENSEMATPROD_H

#include <armadillo>

template <typename Scalar>
class DenseMatProd
{
private:
    typedef arma::Mat<Scalar> Matrix;
    typedef arma::Col<Scalar> Vector;

    const Matrix mat;

public:
    DenseMatProd(Matrix &mat_) :
        mat(mat_.memptr(), mat_.n_rows, mat_.n_cols, false)
    {}

    virtual ~DenseMatProd() {}

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


#endif // DENSEMATPROD_H
