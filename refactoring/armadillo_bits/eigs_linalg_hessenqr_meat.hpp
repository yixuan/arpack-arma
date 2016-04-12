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


template<typename eT>
inline
UpperHessenbergQR<eT>::UpperHessenbergQR()
  : n(0)
  , computed(false)
  {}



template<typename eT>
inline
UpperHessenbergQR<eT>::UpperHessenbergQR(const Mat<eT>& mat)
  : n(mat.n_rows)
  , mat_T(n, n)
  , rot_cos(n - 1)
  , rot_sin(n - 1)
  , computed(false)
  {
  compute(mat);
  }



template<typename eT>
void
UpperHessenbergQR<eT>::compute(const Mat<eT>& mat)
  {
  n = mat.n_rows;
  mat_T.set_size(n, n);
  rot_cos.set_size(n - 1);
  rot_sin.set_size(n - 1);

  // Make a copy of mat
  std::copy(mat.memptr(), mat.memptr() + mat.n_elem, mat_T.memptr());

  eT xi, xj, r, c, s, eps = std::numeric_limits<eT>::epsilon();
  eT *Tii, *ptr;
  for(uword i = 0; i < n - 1; i++)
    {
    Tii = mat_T.colptr(i) + i;

    // Make sure mat_T is upper Hessenberg
    // Zero the elements below mat_T(i + 1, i)
    std::fill(Tii + 2, Tii + n - i, eT(0));

    xi = Tii[0];  // mat_T(i, i)
    xj = Tii[1];  // mat_T(i + 1, i)
    r = std::sqrt(xi * xi + xj * xj);
    if(r <= eps)
      {
      r = 0;
      rot_cos[i] = c = 1;
      rot_sin[i] = s = 0;
      }
    else
      {
      rot_cos[i] = c = xi / r;
      rot_sin[i] = s = -xj / r;
      }

    // For a complete QR decomposition,
    // we first obtain the rotation matrix
    // G = [ cos  sin]
    //     [-sin  cos]
    // and then do T[i:(i + 1), i:(n - 1)] = G' * T[i:(i + 1), i:(n - 1)]

    // mat_T.submat(i, i, i + 1, n - 1) = Gt * mat_T.submat(i, i, i + 1, n - 1);
    Tii[0] = r;    // mat_T(i, i)     => r
    Tii[1] = 0;    // mat_T(i + 1, i) => 0
    ptr = Tii + n; // mat_T(i, k), k = i+1, i+2, ..., n-1
    for(uword j = i + 1; j < n; j++, ptr += n)
      {
      eT tmp = ptr[0];
      ptr[0] = c * tmp - s * ptr[1];
      ptr[1] = s * tmp + c * ptr[1];
      }
    }

    computed = true;
  }



template<typename eT>
Mat<eT>
UpperHessenbergQR<eT>::matrix_RQ()
  {
  arma_debug_check( (computed == false),
    "UpperHessenbergQR::matrix_RQ(): need to call compute() first" );

  // Make a copy of the R matrix
  Mat<eT> RQ = trimatu(mat_T);

  eT *c = rot_cos.memptr(),
     *s = rot_sin.memptr();
  for(uword i = 0; i < n - 1; i++)
    {
    // RQ[, i:(i + 1)] = RQ[, i:(i + 1)] * Gi
    // Gi = [ cos[i]  sin[i]]
    //      [-sin[i]  cos[i]]

    eT *Yi, *Yi1;
    Yi = RQ.colptr(i);
    Yi1 = Yi + n;  // RQ(0, i + 1)
    for(uword j = 0; j < i + 2; j++)
      {
      eT tmp = Yi[j];
      Yi[j]  = (*c) * tmp - (*s) * Yi1[j];
      Yi1[j] = (*s) * tmp + (*c) * Yi1[j];
      }

    /* Yi = RQ(span(0, i + 1), i);
    RQ(span(0, i + 1), i)     = (*c) * Yi - (*s) * RQ(span(0, i + 1), i + 1);
    RQ(span(0, i + 1), i + 1) = (*s) * Yi + (*c) * RQ(span(0, i + 1), i + 1); */
    c++;
    s++;
    }

    return RQ;
  }



template<typename eT>
inline
void
UpperHessenbergQR<eT>::apply_YQ(Mat<eT>& Y)
  {
  arma_debug_check( (computed == false),
    "UpperHessenbergQR::apply_YQ(): need to call compute() first" );

  eT *c = rot_cos.memptr(),
     *s = rot_sin.memptr();

  eT *Y_col_i, *Y_col_i1;
  uword nrow = Y.n_rows;
  for(uword i = 0; i < n - 1; i++)
    {
    Y_col_i  = Y.colptr(i);
    Y_col_i1 = Y.colptr(i + 1);
    for(uword j = 0; j < nrow; j++)
      {
      eT tmp = Y_col_i[j];
      Y_col_i[j]  = (*c) * tmp - (*s) * Y_col_i1[j];
      Y_col_i1[j] = (*s) * tmp + (*c) * Y_col_i1[j];
      }
    c++;
    s++;
    }
  }



template<typename eT>
inline
TridiagQR<eT>::TridiagQR()
  : UpperHessenbergQR<eT>()
  {}



template<typename eT>
inline
TridiagQR<eT>::TridiagQR(const Mat<eT>& mat)
  : UpperHessenbergQR<eT>()
  {
  this->compute(mat);
  }



template<typename eT>
inline
void
TridiagQR<eT>::compute(const Mat<eT>& mat)
  {
  this->n = mat.n_rows;
  this->mat_T.set_size(this->n, this->n);
  this->rot_cos.set_size(this->n - 1);
  this->rot_sin.set_size(this->n - 1);

  this->mat_T.zeros();
  this->mat_T.diag() = mat.diag();
  this->mat_T.diag(1) = mat.diag(-1);
  this->mat_T.diag(-1) = mat.diag(-1);

  // A number of pointers to avoid repeated address calculation
  eT *Tii = this->mat_T.memptr(),  // pointer to T[i, i]
     *ptr,                         // some location relative to Tii
     *c = this->rot_cos.memptr(),  // pointer to the cosine vector
     *s = this->rot_sin.memptr(),  // pointer to the sine vector
     r, tmp,
     eps = std::numeric_limits<eT>::epsilon();
  for(uword i = 0; i < this->n - 2; i++)
    {
    // Tii[0] == T[i, i]
    // Tii[1] == T[i + 1, i]
    r = std::sqrt(Tii[0] * Tii[0] + Tii[1] * Tii[1]);
    if(r <= eps)
      {
      r = 0;
      *c = 1;
      *s = 0;
      }
    else
      {
      *c =  Tii[0] / r;
      *s = -Tii[1] / r;
      }

    // For a complete QR decomposition,
    // we first obtain the rotation matrix
    // G = [ cos  sin]
    //     [-sin  cos]
    // and then do T[i:(i + 1), i:(i + 2)] = G' * T[i:(i + 1), i:(i + 2)]

    // Update T[i, i] and T[i + 1, i]
    // The updated value of T[i, i] is known to be r
    // The updated value of T[i + 1, i] is known to be 0
    Tii[0] = r;
    Tii[1] = 0;
    // Update T[i, i + 1] and T[i + 1, i + 1]
    // ptr[0] == T[i, i + 1]
    // ptr[1] == T[i + 1, i + 1]
    ptr = Tii + this->n;
    tmp = *ptr;
    ptr[0] = (*c) * tmp - (*s) * ptr[1];
    ptr[1] = (*s) * tmp + (*c) * ptr[1];
    // Update T[i, i + 2] and T[i + 1, i + 2]
    // ptr[0] == T[i, i + 2] == 0
    // ptr[1] == T[i + 1, i + 2]
    ptr += this->n;
    ptr[0] = -(*s) * ptr[1];
    ptr[1] *= (*c);

    // Move from T[i, i] to T[i + 1, i + 1]
    Tii += this->n + 1;
    // Increase c and s by 1
    c++;
    s++;
    }

  // For i = n - 2
  r = std::sqrt(Tii[0] * Tii[0] + Tii[1] * Tii[1]);
  if(r <= eps)
    {
    r = 0;
    *c = 1;
    *s = 0;
    }
  else
    {
    *c =  Tii[0] / r;
    *s = -Tii[1] / r;
    }
  Tii[0] = r;
  Tii[1] = 0;
  ptr = Tii + this->n;  // points to T[i - 2, i - 1]
  tmp = *ptr;
  ptr[0] = (*c) * tmp - (*s) * ptr[1];
  ptr[1] = (*s) * tmp + (*c) * ptr[1];

  this->computed = true;
  }



template<typename eT>
Mat<eT>
TridiagQR<eT>::matrix_RQ()
  {
  arma_debug_check( (this->computed == false),
    "TridiagQR::matrix_RQ(): need to call compute() first" );

  // Make a copy of the R matrix
  Mat<eT> RQ(this->n, this->n, fill::zeros);
  RQ.diag() = this->mat_T.diag();
  RQ.diag(1) = this->mat_T.diag(1);

  // [m11  m12] will point to RQ[i:(i+1), i:(i+1)]
  // [m21  m22]
  eT *m11 = RQ.memptr(), *m12, *m21, *m22,
     *c = this->rot_cos.memptr(),
     *s = this->rot_sin.memptr(),
     tmp;
  for(uword i = 0; i < this->n - 1; i++)
    {
    m21 = m11 + 1;
    m12 = m11 + this->n;
    m22 = m12 + 1;
    tmp = *m21;

    // Update diagonal and the below-subdiagonal
    *m11 = (*c) * (*m11) - (*s) * (*m12);
    *m21 = (*c) * tmp - (*s) * (*m22);
    *m22 = (*s) * tmp + (*c) * (*m22);

    // Move m11 to RQ[i+1, i+1]
    m11  = m22;
    c++;
    s++;
    }

  // Copy the below-subdiagonal to above-subdiagonal
  RQ.diag(1) = RQ.diag(-1);

  return RQ;
  }


}  // namespace alt_eigs
