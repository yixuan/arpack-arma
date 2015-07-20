#ifndef DENSE_GEN_MAT_PROD_H
#define DENSE_GEN_MAT_PROD_H

#include <armadillo>

///
/// \defgroup MatOp Matrix Operations
///

///
/// \ingroup MatOp
///
/// This class defines the matrix-vector multiplication operation on a
/// general real matrix \f$A\f$, i.e., calculating \f$y=Ax\f$ for any vector
/// \f$x\f$. It is mainly used in the GenEigsSolver and
/// SymEigsSolver eigen solvers.
///
template <typename Scalar>
class DenseGenMatProd
{
private:
    typedef arma::Mat<Scalar> Matrix;
    typedef arma::Col<Scalar> Vector;

    const Matrix mat;

public:
    ///
    /// Constructor to create the matrix operation object.
    ///
    /// \param mat_ An **Armadillo** matrix object, whose type can be `arma::mat`
    ///             or `arma::fmat`, depending on the template parameter `Scalar` defined.
    ///
    DenseGenMatProd(Matrix &mat_) :
        mat(mat_.memptr(), mat_.n_rows, mat_.n_cols, false)
    {}

    ///
    /// Return the number of rows of the underlying matrix.
    ///
    int rows() { return mat.n_rows; }
    ///
    /// Return the number of columns of the underlying matrix.
    ///
    int cols() { return mat.n_cols; }

    ///
    /// Perform the matrix-vector multiplication operation \f$y=Ax\f$.
    ///
    /// \param x_in  Pointer to the \f$x\f$ vector.
    /// \param y_out Pointer to the \f$y\f$ vector.
    ///
    // y_out = A * x_in
    void perform_op(Scalar *x_in, Scalar *y_out)
    {
        Vector x(x_in, mat.n_cols, false);
        Vector y(y_out, mat.n_rows, false);
        y = mat * x;
    }
};


#endif // DENSE_GEN_MAT_PROD_H
