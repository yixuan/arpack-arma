// Copyright (C) 2013-2015 National ICT Australia (NICTA)
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.
// -------------------------------------------------------------------
//
// Written by Yixuan Qiu


namespace alt_eigs
{


//! Define matrix operations on existing matrix objects
template<typename eT>
class SparseGenMatProd
  {
  private:

  const SpMat<eT>* op_mat;


  public:

  //! Constructor to create the matrix operation object.
  inline SparseGenMatProd(const SpMat<eT>& mat_)
    : op_mat(&mat_)
  {}

  //! Return the number of rows of the underlying matrix.
  arma_inline uword rows() const { return op_mat->n_rows; }

  //! Return the number of columns of the underlying matrix.
  arma_inline uword cols() const { return op_mat->n_cols; }

  //! Perform the matrix-vector multiplication operation \f$y=Ax\f$.
  // y_out = A * x_in
  arma_inline void perform_op(eT* x_in, eT* y_out) const
    {
    Col<eT> x(x_in , op_mat->n_cols, false);
    Col<eT> y(y_out, op_mat->n_rows, false);
    y = (*op_mat) * x;
    }
  };


}  // namespace alt_eigs
