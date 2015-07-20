#ifndef DENSE_GEN_REAL_SHIFT_SOLVE_H
#define DENSE_GEN_REAL_SHIFT_SOLVE_H

#include <armadillo>
#include <stdexcept>
#include "../LinAlg/GeneralLU.h"

///
/// \ingroup MatOp
///
/// This class defines the shift-solve operation on a general real matrix \f$A\f$,
/// i.e., calculating \f$y=(A-\sigma I)^{-1}x\f$ for any real \f$\sigma\f$ and
/// vector \f$x\f$. It is mainly used in the GenEigsRealShiftSolver eigen solver.
///
template <typename Scalar>
class DenseGenRealShiftSolve
{
private:
    typedef arma::Mat<Scalar> Matrix;
    typedef arma::Col<Scalar> Vector;

    const Matrix mat;
    const int dim_n;
    GeneralLU<Scalar> solver;
public:
    ///
    /// Constructor to create the matrix operation object.
    ///
    /// \param mat_ An **Armadillo** matrix object, whose type can be `arma::mat`
    ///             or `arma::fmat`, depending on the template parameter `Scalar` defined.
    ///
    DenseGenRealShiftSolve(Matrix &mat_) :
        mat(mat_.memptr(), mat_.n_rows, mat_.n_cols, false),
        dim_n(mat_.n_rows)
    {
        if(!mat_.is_square())
            throw std::invalid_argument("DenseGenRealShiftSolve: matrix must be square");
    }

    ///
    /// Return the number of rows of the underlying matrix.
    ///
    int rows() { return dim_n; }
    ///
    /// Return the number of columns of the underlying matrix.
    ///
    int cols() { return dim_n; }

    ///
    /// Set the real shift \f$\sigma\f$.
    ///
    void set_shift(Scalar sigma)
    {
        solver.compute(mat - sigma * arma::eye<Matrix>(dim_n, dim_n));
    }

    ///
    /// Perform the shift-solve operation \f$y=(A-\sigma I)^{-1}x\f$.
    ///
    /// \param x_in  Pointer to the \f$x\f$ vector.
    /// \param y_out Pointer to the \f$y\f$ vector.
    ///
    // y_out = inv(A - sigma * I) * x_in
    void perform_op(Scalar *x_in, Scalar *y_out)
    {
        Vector x(x_in,  dim_n, false);
        Vector y(y_out, dim_n, false);
        solver.solve(x, y);
    }
};


#endif // DENSE_GEN_REAL_SHIFT_SOLVE_H
