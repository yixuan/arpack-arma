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
UpperHessenbergEigen<eT>::UpperHessenbergEigen()
  : n(0)
  , computed(false)
  {}



template<typename eT>
inline
UpperHessenbergEigen<eT>::UpperHessenbergEigen(const Mat<eT>& mat_obj)
  : n(mat_obj.n_rows)
  , computed(false)
  {
  compute(mat_obj);
  }



template<typename eT>
inline
void
UpperHessenbergEigen<eT>::compute(const Mat<eT>& mat_obj)
  {
  arma_debug_check( (mat_obj.is_square() == false), "alt_eigs::UpperHessenbergEigen::compute(): matrix must be square" );

  n = mat_obj.n_rows;
  mat_Z.set_size(n, n);
  mat_T.set_size(n, n);
  evals.set_size(n);

  mat_Z.eye();
  mat_T = mat_obj;

  blas_int want_T = 1, want_Z = 1;
  blas_int ilo = 1, ihi = n, iloz = 1, ihiz = n;
  podarray<eT> wr(n);
  podarray<eT> wi(n);
  blas_int info;
  lapack::lahqr(&want_T, &want_Z, &n, &ilo, &ihi,
                mat_T.memptr(), &n, wr.memptr(), wi.memptr(), &iloz, &ihiz,
                mat_Z.memptr(), &n, &info);

  for(blas_int i = 0; i < n; i++)
    {
    evals(i) = std::complex<eT>(wr[i], wi[i]);
    }

  if(info > 0) { arma_stop("lapack::lahqr(): failed to compute all the eigenvalues"); }

  char side = 'R', howmny = 'B';
  podarray<eT> work(3 * n);
  blas_int m;

  lapack::trevc(&side, &howmny, (blas_int*) NULL, &n, mat_T.memptr(), &n,
                (eT*) NULL, &n, mat_Z.memptr(), &n, &n, &m, work.memptr(), &info);

  if(info < 0) { arma_stop("lapack::trevc(): illegal value"); }

  computed = true;
  }



template<typename eT>
inline
Col< std::complex<eT> >
UpperHessenbergEigen<eT>::eigenvalues()
  {
  arma_debug_check( (computed == false), "alt_eigs::UpperHessenbergEigen::eigenvalues(): need to call compute() first" );

  return evals;
  }



template<typename eT>
inline
Mat< std::complex<eT> >
UpperHessenbergEigen<eT>::eigenvectors()
  {
  arma_debug_check( (computed == false), "alt_eigs::UpperHessenbergEigen::eigenvectors(): need to call compute() first" );

  eT prec = std::pow(std::numeric_limits<eT>::epsilon(), eT(2.0) / 3);
  Mat< std::complex<eT> > evecs(n, n);
  std::complex<eT>* col_ptr = evecs.memptr();
  for(blas_int i = 0; i < n; i++)
    {
    if(cx_attrib<eT>::is_real(evals(i), prec))
      {
      // For real eigenvector, normalize and copy
      eT z_norm = norm(mat_Z.col(i));
      for(blas_int j = 0; j < n; j++)
        {
        col_ptr[j] = std::complex<eT>(mat_Z(j, i) / z_norm, 0);
        }

      col_ptr += n;
      }
    else
      {
      // Complex eigenvectors are stored in consecutive columns
      eT r2 = dot(mat_Z.col(i), mat_Z.col(i));
      eT i2 = dot(mat_Z.col(i + 1), mat_Z.col(i + 1));
      eT z_norm = std::sqrt(r2 + i2);
      eT* z_ptr = mat_Z.colptr(i);
      for(blas_int j = 0; j < n; j++)
        {
        col_ptr[j] = std::complex<eT>(z_ptr[j] / z_norm, z_ptr[j + n] / z_norm);
        col_ptr[j + n] = std::conj(col_ptr[j]);
        }

      i++;
      col_ptr += 2 * n;
      }
    }

  return evecs;
  }


}  // namespace alt_eigs
