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


template<typename eT>
inline
TridiagEigen<eT>::TridiagEigen()
  : n(0)
  , computed(false)
  {}



template<typename eT>
inline
TridiagEigen<eT>::TridiagEigen(const Mat<eT>& mat_obj)
  : n(mat_obj.n_rows)
  , computed(false)
  {
  compute(mat_obj);
  }



template<typename eT>
inline
void
TridiagEigen<eT>::compute(const Mat<eT>& mat_obj)
  {
  arma_debug_check( (mat_obj.is_square() == false), "TridiagEigen::compute(): matrix must be square" );

  n = mat_obj.n_rows;
  main_diag = mat_obj.diag();
  sub_diag = mat_obj.diag(-1);
  evecs.set_size(n, n);

  char compz = 'I';
  blas_int lwork = -1;
  eT lwork_opt;

  blas_int liwork = -1;
  blas_int liwork_opt, info;

  // Query of lwork and liwork
  lapack::stedc(&compz, &n, main_diag.memptr(), sub_diag.memptr(),
                evecs.memptr(), &n, &lwork_opt, &lwork, &liwork_opt, &liwork, &info);

  if(info == 0)
    {
    lwork = (blas_int) lwork_opt;
    liwork = liwork_opt;
    }
  else
    {
    lwork = 1 + 4 * n + n * n;
    liwork = 3 + 5 * n;
  }

  podarray<eT> work(lwork);
  podarray<blas_int> iwork(liwork);

  lapack::stedc(&compz, &n, main_diag.memptr(), sub_diag.memptr(),
                evecs.memptr(), &n, work.memptr(), &lwork, iwork.memptr(), &liwork, &info);

  if(info < 0) { arma_stop("lapack::stedc(): illegal value"); }
  if(info > 0) { arma_stop("lapack::stedc(): failed to compute all the eigenvalues"); }

  computed = true;
  }



template<typename eT>
inline
Col<eT>
TridiagEigen<eT>::eigenvalues()
  {
  arma_debug_check( (computed == false), "TridiagEigen::eigenvalues(): need to call compute() first" );

  // After calling compute(), main_diag will contain the eigenvalues.
  return main_diag;
  }



template<typename eT>
inline
Mat<eT>
TridiagEigen<eT>::eigenvectors()
  {
  arma_debug_check( (computed == false), "TridiagEigen::eigenvectors(): need to call compute() first" );

  return evecs;
  }


}  // namespace alt_eigs
