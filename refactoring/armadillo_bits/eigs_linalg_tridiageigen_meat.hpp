// Copyright (C) 2015 Yixuan Qiu
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.



template<typename eT>
inline
TridiagEigen<eT>::TridiagEigen()
  : n(0)
  , computed(false)
  {}



template<typename eT>
inline
TridiagEigen<eT>::TridiagEigen(const Mat<eT>& mat)
  : n(mat.n_rows)
  , computed(false)
  {
  compute(mat);
  }



template<typename eT>
inline
void
TridiagEigen<eT>::compute(const Mat<eT>& mat)
  {
  if(!mat.is_square())
    throw std::invalid_argument("TridiagEigen: matrix must be square");

  n = mat.n_rows;
  main_diag = mat.diag();
  sub_diag = mat.diag(-1);
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

  eT *work = new eT[lwork];
  blas_int *iwork = new int[liwork];

  lapack::stedc(&compz, &n, main_diag.memptr(), sub_diag.memptr(),
                evecs.memptr(), &n, work, &lwork, iwork, &liwork, &info);

  delete [] work;
  delete [] iwork;

  if(info < 0)
    throw std::invalid_argument("Lapack stedc: illegal value");
  if(info > 0)
    throw std::logic_error("Lapack stedc: failed to compute all the eigenvalues");

  computed = true;
  }



template<typename eT>
inline
Col<eT>
TridiagEigen<eT>::eigenvalues()
  {
  if(!computed)
    throw std::logic_error("TridiagEigen: need to call compute() first");

  // After calling compute(), main_diag will contain the eigenvalues.
  return main_diag;
  }



template<typename eT>
inline
Mat<eT>
TridiagEigen<eT>::eigenvectors()
  {
  if(!computed)
    throw std::logic_error("TridiagEigen: need to call compute() first");

  return evecs;
  }
