// Copyright (C) 2015 Yixuan Qiu
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.


//! Define matrix operations on existing matrix objects
template<typename eT>
class SparseGenMatProd
  {
  private:

  const SpMat<eT>* mat;


  public:

  //! Constructor to create the matrix operation object.
  inline SparseGenMatProd(const SpMat<eT>& mat_)
    : mat(&mat_)
  {}

  //! Return the number of rows of the underlying matrix.
  arma_inline uword rows()
    {
    return mat->n_rows;
    }

  //! Return the number of columns of the underlying matrix.
  arma_inline uword cols()
    {
    return mat->n_cols;
    }

  //! Perform the matrix-vector multiplication operation \f$y=Ax\f$.
  // y_out = A * x_in
  arma_inline void perform_op(eT* x_in, eT* y_out)
    {
    Col<eT> x(x_in , mat->n_cols, false);
    Col<eT> y(y_out, mat->n_rows, false);
    y = (*mat) * x;
    }
  };