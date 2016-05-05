// Copyright (C) 2016 National ICT Australia (NICTA)
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

  const SpMat<eT>& op_mat;


  public:

  //! Number of rows of the underlying matrix.
  const uword n_rows;

  //! Number of columns of the underlying matrix.
  const uword n_cols;

  //! Constructor to create the matrix operation object.
  inline SparseGenMatProd(const SpMat<eT>& mat_obj)
    : op_mat(mat_obj)
    , n_rows(mat_obj.n_rows)
    , n_cols(mat_obj.n_cols)
  {}

  //! Perform the matrix-vector multiplication operation \f$y=Ax\f$.
  // y_out = A * x_in
  arma_inline void perform_op(eT* x_in, eT* y_out) const
    {
    Col<eT> x(x_in , n_cols, false);
    Col<eT> y(y_out, n_rows, false);
    y = op_mat * x;
    }
  };


}  // namespace alt_eigs
