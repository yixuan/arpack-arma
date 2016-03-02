// Copyright (C) 2015 Yixuan Qiu
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.


template<typename eT, int SelectionRule, typename OpType>
inline
void
SymEigsSolver<eT, SelectionRule, OpType>::factorize_from(uword from_k, uword to_m, const Col<eT>& fk)
  {
  if(to_m <= from_k)
    return;

  fac_f = fk;

  Col<eT> w(dim_n);
  eT beta = std::sqrt(dot(fac_f, fac_f)), Hii = 0.0;
  // Keep the upperleft k x k submatrix of H and set other elements to 0
  fac_H.tail_cols(ncv - from_k).zeros();
  fac_H.submat(span(from_k, ncv - 1), span(0, from_k - 1)).zeros();
  for(uword i = from_k; i <= to_m - 1; i++)
    {
    Col<eT> v(fac_V.colptr(i), dim_n, false);
    // v <- f / ||f||
    v = fac_f / beta; // The (i+1)-th column
    fac_H(i, i - 1) = beta;

    // w <- A * v, v = fac_V.col(i)
    op->perform_op(v.memptr(), w.memptr());
    nmatop++;

    Hii = dot(v, w);
    fac_H(i - 1, i) = beta;
    fac_H(i, i) = Hii;

    // f <- w - V * V' * w
    fac_f = w - beta * fac_V.col(i - 1) - Hii * v;
    beta = std::sqrt(dot(fac_f, fac_f));

    // f/||f|| is going to be the next column of V, so we need to test
    // whether V' * (f/||f||) ~= 0
    Mat<eT> Vs(fac_V.memptr(), dim_n, i + 1, false); // First i+1 columns
    Col<eT> Vf = Vs.t() * fac_f;
    // If not, iteratively correct the residual
    uword count = 0;
    while(count < 5 && abs(Vf).max() > prec * beta)
      {
      // f <- f - V * Vf
      fac_f -= Vs * Vf;
      // h <- h + Vf
      fac_H(i - 1, i) += Vf[i - 1];
      fac_H(i, i - 1) = fac_H(i - 1, i);
      fac_H(i, i) += Vf[i];
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
SymEigsSolver<eT, SelectionRule, OpType>::restart(uword k)
  {
  if(k >= ncv)
    return;

  TridiagQR<eT> decomp;
  Mat<eT> Q = arma::eye< Mat<eT> >(ncv, ncv);

  for(uword i = k; i < ncv; i++)
    {
    // QR decomposition of H-mu*I, mu is the shift
    fac_H.diag() -= ritz_val[i];
    decomp.compute(fac_H);

    // Q -> Q * Qi
    decomp.apply_YQ(Q);

    // H -> Q'HQ
    // Since QR = H - mu * I, we have H = QR + mu * I
    // and therefore Q'HQ = RQ + mu * I
    fac_H = decomp.matrix_RQ();
    fac_H.diag() += ritz_val[i];
    }

  // V -> VQ, only need to update the first k+1 columns
  // Q has some elements being zero
  // The first (ncv - k + i) elements of the i-th column of Q are non-zero
  Mat<eT> Vs(dim_n, k + 1);
  uword nnz;
  for(uword i = 0; i < k; i++)
    {
    nnz = ncv - k + i + 1;
    Mat<eT> V(fac_V.memptr(), dim_n, nnz, false);
    Col<eT> q(Q.colptr(i), nnz, false);
    Vs.col(i) = V * q;
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
SymEigsSolver<eT, SelectionRule, OpType>::num_converged(eT tol)
  {
  // thresh = tol * max(prec, abs(theta)), theta for ritz value
  Col<eT> rv = abs(ritz_val.head(nev));
  Col<eT> thresh = tol * clamp(rv, prec, std::max(prec, rv.max()));
  Col<eT> resid = abs(ritz_vec.tail_rows(1).t()) * norm(fac_f);
  // Converged "wanted" ritz values
  ritz_conv = (resid < thresh);

  return sum(ritz_conv);
  }



template<typename eT, int SelectionRule, typename OpType>
inline
uword
SymEigsSolver<eT, SelectionRule, OpType>::nev_adjusted(uword nconv)
  {
  uword nev_new = nev;

  // Adjust nev_new, according to dsaup2.f line 677~684 in ARPACK
  nev_new = nev + std::min(nconv, (ncv - nev) / 2);
  if(nev == 1 && ncv >= 6)
    nev_new = ncv / 2;
  else
  if(nev == 1 && ncv > 2)
    nev_new = 2;

  return nev_new;
  }



template<typename eT, int SelectionRule, typename OpType>
inline
void
SymEigsSolver<eT, SelectionRule, OpType>::retrieve_ritzpair()
  {
  TridiagEigen<eT> decomp(fac_H);
  Col<eT> evals = decomp.eigenvalues();
  Mat<eT> evecs = decomp.eigenvectors();

  SortEigenvalue<eT, SelectionRule> sorting(evals.memptr(), evals.n_elem);
  std::vector<uword> ind = sorting.index();

  // For BOTH_ENDS, the eigenvalues are sorted according
  // to the LARGEST_ALGE rule, so we need to move those smallest
  // values to the left
  // The order would be
  // Largest => Smallest => 2nd largest => 2nd smallest => ...
  // We keep this order since the first k values will always be
  // the wanted collection, no matter k is nev_updated (used in restart())
  // or is nev (used in sort_ritzpair())
  if(SelectionRule == EigsSelect::BOTH_ENDS)
    {
    std::vector<uword> ind_copy(ind);
    for(uword i = 0; i < ncv; i++)
      {
      // If i is even, pick values from the left (large values)
      // If i is odd, pick values from the right (small values)
      if(i % 2 == 0)
        ind[i] = ind_copy[i / 2];
      else
        ind[i] = ind_copy[ncv - 1 - i / 2];
      }
    }

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
SymEigsSolver<eT, SelectionRule, OpType>::sort_ritzpair()
  {
  SortEigenvalue<eT, EigsSelect::LARGEST_MAGN> sorting(ritz_val.memptr(), nev);
  std::vector<uword> ind = sorting.index();

  Col<eT> new_ritz_val(ncv);
  Mat<eT> new_ritz_vec(ncv, nev);
  Col<uword> new_ritz_conv(nev);

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
SymEigsSolver<eT, SelectionRule, OpType>::SymEigsSolver(OpType* op_, uword nev_, uword ncv_)
  : op(op_)
  , nev(nev_)
  , dim_n(op->rows())
  , ncv(ncv_ > dim_n ? dim_n : ncv_)
  , nmatop(0)
  , niter(0)
  , prec(std::pow(std::numeric_limits<eT>::epsilon(), eT(2.0) / 3))
  {
  if(nev_ < 1 || nev_ > dim_n - 1)
    throw std::invalid_argument("nev must satisfy 1 <= nev <= n - 1, n is the size of matrix");

  if(ncv_ <= nev_ || ncv_ > dim_n)
    throw std::invalid_argument("ncv must satisfy nev < ncv <= n, n is the size of matrix");
  }



template<typename eT, int SelectionRule, typename OpType>
inline
void
SymEigsSolver<eT, SelectionRule, OpType>::init(eT* init_resid)
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
SymEigsSolver<eT, SelectionRule, OpType>::init()
  {
  Col<eT> init_resid(dim_n, fill::randu);
  init_resid -= 0.5;
  init(init_resid.memptr());
  }



template<typename eT, int SelectionRule, typename OpType>
inline
uword
SymEigsSolver<eT, SelectionRule, OpType>::compute(uword maxit, eT tol)
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
Col<eT>
SymEigsSolver<eT, SelectionRule, OpType>::eigenvalues()
  {
  uword nconv = sum(ritz_conv);
  Col<eT> res(nconv);

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
Mat<eT>
SymEigsSolver<eT, SelectionRule, OpType>::eigenvectors(uword nvec)
  {
  uword nconv = sum(ritz_conv);
  nvec = std::min(nvec, nconv);
  Mat<eT> res(dim_n, nvec);

  if(!nvec)
    return res;

  Mat<eT> ritz_vec_conv(ncv, nvec);
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