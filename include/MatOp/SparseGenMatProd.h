// Copyright (C) 2015 Yixuan Qiu
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef SPARSE_GEN_MAT_PROD_H
#define SPARSE_GEN_MAT_PROD_H

#include <armadillo>

///
/// \ingroup MatOp
///
/// This class defines the matrix-vector multiplication operation on a
/// sparse general real matrix \f$A\f$, i.e., calculating \f$y=Ax\f$ for any vector
/// \f$x\f$. It is mainly used in the GenEigsSolver and
/// SymEigsSolver eigen solvers.
///
template <typename Scalar>
class SparseGenMatProd
{
private:
    typedef arma::Mat<Scalar>   Matrix;
    typedef arma::Col<Scalar>   Vector;
    typedef arma::SpMat<Scalar> SpMatrix;

    const SpMatrix* mat;

public:
    ///
    /// Constructor to create the matrix operation object.
    ///
    /// \param mat_ An **Armadillo** sparse matrix object, whose type can be `arma::sp_mat`
    ///             or `arma::sp_fmat`, depending on the template parameter `Scalar` defined.
    ///
    SparseGenMatProd(const SpMatrix &mat_) :
        mat(&mat_)
    {}

    ///
    /// Return the number of rows of the underlying matrix.
    ///
    int rows() { return mat->n_rows; }
    ///
    /// Return the number of columns of the underlying matrix.
    ///
    int cols() { return mat->n_cols; }

    ///
    /// Perform the matrix-vector multiplication operation \f$y=Ax\f$.
    ///
    /// \param x_in  Pointer to the \f$x\f$ vector.
    /// \param y_out Pointer to the \f$y\f$ vector.
    ///
    // y_out = A * x_in
    void perform_op(Scalar *x_in, Scalar *y_out)
    {
        Vector x(x_in, mat->n_cols, false);
        Vector y(y_out, mat->n_rows, false);
        y = (*mat) * x;
    }
};


#endif // SPARSE_GEN_MAT_PROD_H
