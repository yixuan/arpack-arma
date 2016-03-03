// Copyright (C) 2015 Yixuan Qiu
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.


template<typename eT, int SelectionRule, typename OpType>
inline
void
GenEigsSolver<eT, SelectionRule, OpType>::factorize_from(uword from_k, uword to_m, const Col<eT>& fk)
  {
  if(to_m <= from_k)
    return;

  fac_f = fk;

  Col<eT> w(dim_n);
  eT beta = std::sqrt(dot(fac_f, fac_f));
  // Keep the upperleft k x k submatrix of H and set other elements to 0
  fac_H.tail_cols(ncv - from_k).zeros();
  fac_H.submat(span(from_k, ncv - 1), span(0, from_k - 1)).zeros();
  for(uword i = from_k; i <= to_m - 1; i++)
    {
    // v <- f / ||f||
    fac_V.col(i) = fac_f / beta; // The (i+1)-th column
    fac_H(i, i - 1) = beta;

    // w <- A * v, v = fac_V.col(i)
    op->perform_op(fac_V.colptr(i), w.memptr());
    nmatop++;

    // First i+1 columns of V
    Mat<eT> Vs(fac_V.memptr(), dim_n, i + 1, false);
    // h = fac_H(0:i, i)
    Col<eT> h(fac_H.colptr(i), i + 1, false);
    // h <- V' * w
    h = Vs.t() * w;

    // f <- w - V * h
    fac_f = w - Vs * h;
    beta = std::sqrt(dot(fac_f, fac_f));

    if(beta > 0.717 * std::sqrt(dot(h, h)))
      continue;

    // f/||f|| is going to be the next column of V, so we need to test
    // whether V' * (f/||f||) ~= 0
    Col<eT> Vf = Vs.t() * fac_f;
    // If not, iteratively correct the residual
    uword count = 0;
    while(count < 5 && abs(Vf).max() > prec * beta)
      {
      // f <- f - V * Vf
      fac_f -= Vs * Vf;
      // h <- h + Vf
      h += Vf;
      // beta <- ||f||
      beta = std::sqrt(dot(fac_f, fac_f));

      Vf = Vs.t() * fac_f;
      count++;
      }
    }
  }



template<typename eT, int SelectionRule, typename OpType>
inline
void
GenEigsSolver<eT, SelectionRule, OpType>::restart(uword k)
  {
  if(k >= ncv)
    return;

  DoubleShiftQR<eT> decomp_ds(ncv);
  UpperHessenbergQR<eT> decomp;
  Mat<eT> Q(ncv, ncv, fill::eye);

  for(uword i = k; i < ncv; i++)
    {
    if(is_complex(ritz_val[i], prec) && is_conj(ritz_val[i], ritz_val[i + 1], prec))
      {
      // H - mu * I = Q1 * R1
      // H <- R1 * Q1 + mu * I = Q1' * H * Q1
      // H - conj(mu) * I = Q2 * R2
      // H <- R2 * Q2 + conj(mu) * I = Q2' * H * Q2
      //
      // (H - mu * I) * (H - conj(mu) * I) = Q1 * Q2 * R2 * R1 = Q * R
      eT s = 2 * ritz_val[i].real();
      eT t = std::norm(ritz_val[i]);
      decomp_ds.compute(fac_H, s, t);

      // Q -> Q * Qi
      decomp_ds.apply_YQ(Q);
      // H -> Q'HQ
      fac_H = decomp_ds.matrix_QtHQ();

      i++;
      }
    else
      {
      // QR decomposition of H - mu * I, mu is real
      fac_H.diag() -= ritz_val[i].real();
      decomp.compute(fac_H);

      // Q -> Q * Qi
      decomp.apply_YQ(Q);
      // H -> Q'HQ = RQ + mu * I
      fac_H = decomp.matrix_RQ();
      fac_H.diag() += ritz_val[i].real();
      }
    }
  // V -> VQ
  // Q has some elements being zero
  // The first (ncv - k + i) elements of the i-th column of Q are non-zero
  Mat<eT> Vs(dim_n, k + 1);
  uword nnz;
  for(uword i = 0; i < k; i++)
    {
    nnz = ncv - k + i + 1;
    Mat<eT> V(fac_V.memptr(), dim_n, nnz, false);
    Col<eT> q(Q.colptr(i), nnz, false);
    Col<eT> v(Vs.colptr(i), dim_n, false);
    v = V * q;
    }
  Vs.col(k) = fac_V * Q.col(k);
  fac_V.head_cols(k + 1) = Vs;

  Col<eT> fk = fac_f * Q(ncv - 1, k - 1) + fac_V.col(k) * fac_H(k, k - 1);
  factorize_from(k, ncv, fk);
  retrieve_ritzpair();
}



template<typename eT, int SelectionRule, typename OpType>
inline
uword
GenEigsSolver<eT, SelectionRule, OpType>::num_converged(eT tol)
  {
  const eT f_norm = arma::norm(fac_f);
  for(uword i = 0; i < ritz_conv.n_elem; i++)
    {
    eT thresh = tol * std::max(prec, std::abs(ritz_val[i]));
    eT resid = std::abs(ritz_vec(ncv - 1, i)) * f_norm;
    ritz_conv[i] = (resid < thresh);
    }

  return arma::sum(ritz_conv);
  }



template<typename eT, int SelectionRule, typename OpType>
inline
uword
GenEigsSolver<eT, SelectionRule, OpType>::nev_adjusted(uword nconv)
  {
  uword nev_new = nev;

  // Increase nev by one if ritz_val[nev - 1] and
  // ritz_val[nev] are conjugate pairs
  if(is_complex(ritz_val[nev - 1], prec) &&
     is_conj(ritz_val[nev - 1], ritz_val[nev], prec))
    {
    nev_new = nev + 1;
    }
  // Adjust nev_new again, according to dnaup2.f line 660~674 in ARPACK
  nev_new = nev_new + std::min(nconv, (ncv - nev_new) / 2);
  if(nev_new == 1 && ncv >= 6)
    nev_new = ncv / 2;
  else
  if(nev_new == 1 && ncv > 3)
    nev_new = 2;

  if(nev_new > ncv - 2)
    nev_new = ncv - 2;

  // Examine conjugate pairs again
  if(is_complex(ritz_val[nev_new - 1], prec) &&
     is_conj(ritz_val[nev_new - 1], ritz_val[nev_new], prec))
    {
    nev_new++;
    }

  return nev_new;
  }



template<typename eT, int SelectionRule, typename OpType>
inline
void
GenEigsSolver<eT, SelectionRule, OpType>::retrieve_ritzpair()
  {
  UpperHessenbergEigen<eT> decomp(fac_H);
  Col< std::complex<eT> > evals = decomp.eigenvalues();
  Mat< std::complex<eT> > evecs = decomp.eigenvectors();

  SortEigenvalue< std::complex<eT>, SelectionRule > sorting(evals.memptr(), evals.n_elem);
  std::vector<uword> ind = sorting.index();

  // Copy the ritz values and vectors to ritz_val and ritz_vec, respectively
  for(uword i = 0; i < ncv; i++)
    {
    ritz_val[i] = evals[ind[i]];
    }
  for(uword i = 0; i < nev; i++)
    {
    ritz_vec.col(i) = evecs.col(ind[i]);
    }
  }



template<typename eT, int SelectionRule, typename OpType>
inline
void
GenEigsSolver<eT, SelectionRule, OpType>::sort_ritzpair()
  {
  SortEigenvalue< std::complex<eT>, EigsSelect::LARGEST_MAGN > sorting(ritz_val.memptr(), nev);
  std::vector<uword> ind = sorting.index();

  Col< std::complex<eT> > new_ritz_val(ncv);
  Mat< std::complex<eT> > new_ritz_vec(ncv, nev);
  Col<uword>              new_ritz_conv(nev);

  for(uword i = 0; i < nev; i++)
    {
    new_ritz_val[i] = ritz_val[ind[i]];
    new_ritz_vec.col(i) = ritz_vec.col(ind[i]);
    new_ritz_conv[i] = ritz_conv[ind[i]];
    }

  ritz_val.swap(new_ritz_val);
  ritz_vec.swap(new_ritz_vec);
  ritz_conv.swap(new_ritz_conv);
  }



template<typename eT, int SelectionRule, typename OpType>
inline
GenEigsSolver<eT, SelectionRule, OpType>::GenEigsSolver(OpType* op_, uword nev_, uword ncv_)
  : op(op_)
  , nev(nev_)
  , dim_n(op->rows())
  , ncv(ncv_ > dim_n ? dim_n : ncv_)
  , nmatop(0)
  , niter(0)
  , prec(std::pow(std::numeric_limits<eT>::epsilon(), eT(2.0) / 3))
  {
  if(nev_ < 1 || nev_ > dim_n - 2)
    throw std::invalid_argument("nev must satisfy 1 <= nev <= n - 2, n is the size of matrix");

  if(ncv_ < nev_ + 2 || ncv_ > dim_n)
    throw std::invalid_argument("ncv must satisfy nev + 2 <= ncv <= n, n is the size of matrix");
  }



template<typename eT, int SelectionRule, typename OpType>
inline
void
GenEigsSolver<eT, SelectionRule, OpType>::init(eT* init_resid)
  {
  // Reset all matrices/vectors to zero
  fac_V.zeros(dim_n, ncv);
  fac_H.zeros(ncv, ncv);
  fac_f.zeros(dim_n);
  ritz_val.zeros(ncv);
  ritz_vec.zeros(ncv, nev);
  ritz_conv.zeros(nev);

  nmatop = 0;
  niter = 0;

  Col<eT> r(init_resid, dim_n, false);
  // The first column of fac_V
  Col<eT> v(fac_V.colptr(0), dim_n, false);
  eT rnorm = norm(r);
  if(rnorm < prec)
    throw std::invalid_argument("initial residual vector cannot be zero");
  v = r / rnorm;

  Col<eT> w(dim_n);
  op->perform_op(v.memptr(), w.memptr());
  nmatop++;

  fac_H(0, 0) = dot(v, w);
  fac_f = w - v * fac_H(0, 0);
  }



template<typename eT, int SelectionRule, typename OpType>
inline
void
GenEigsSolver<eT, SelectionRule, OpType>::init()
  {
  Col<eT> init_resid(dim_n, fill::randu);
  init_resid -= 0.5;
  init(init_resid.memptr());
  }



template<typename eT, int SelectionRule, typename OpType>
inline
uword
GenEigsSolver<eT, SelectionRule, OpType>::compute(uword maxit, eT tol)
  {
  // The m-step Arnoldi factorization
  factorize_from(1, ncv, fac_f);
  retrieve_ritzpair();
  // Restarting
  uword i, nconv = 0, nev_adj;
  for(i = 0; i < maxit; i++)
    {
    nconv = num_converged(tol);
    if(nconv >= nev)
      break;

    nev_adj = nev_adjusted(nconv);
    restart(nev_adj);
    }
  // Sorting results
  sort_ritzpair();

  niter = i + 1;

  return std::min(nev, nconv);
  }



template<typename eT, int SelectionRule, typename OpType>
inline
Col< std::complex<eT> >
GenEigsSolver<eT, SelectionRule, OpType>::eigenvalues()
  {
  uword nconv = sum(ritz_conv);
  Col< std::complex<eT> > res(nconv);

  if(!nconv)
    return res;

  uword j = 0;
  for(uword i = 0; i < nev; i++)
    {
    if(ritz_conv[i])
      {
      res[j] = ritz_val[i];
      j++;
      }
    }

    return res;
  }



template<typename eT, int SelectionRule, typename OpType>
inline
Mat< std::complex<eT> >
GenEigsSolver<eT, SelectionRule, OpType>::eigenvectors(uword nvec)
  {
  uword nconv = sum(ritz_conv);
  nvec = std::min(nvec, nconv);
  Mat<std::complex<eT>> res(dim_n, nvec);

  if(!nvec)
    return res;

  Mat< std::complex<eT> > ritz_vec_conv(ncv, nvec);
  uword j = 0;
  for(uword i = 0; i < nev && j < nvec; i++)
    {
    if(ritz_conv[i])
      {
      ritz_vec_conv.col(j) = ritz_vec.col(i);
      j++;
      }
    }

  res = fac_V * ritz_vec_conv;

  return res;
  }
