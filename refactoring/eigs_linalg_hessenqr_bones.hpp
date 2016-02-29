// Copyright (C) 2015 Yixuan Qiu
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.


// Perform the QR decomposition of an upper Hessenberg matrix.
template<typename eT>
class UpperHessenbergQR
  {
  public:

  inline UpperHessenbergQR();

  inline UpperHessenbergQR(const Mat<eT>& mat);

  virtual void compute(const Mat<eT>& mat);

  virtual Mat<eT> matrix_RQ();

  inline void apply_YQ(Mat<eT>& Y);


  protected:

  uword   n;
  Mat<eT> mat_T;
  // Gi = [ cos[i]  sin[i]]
  //      [-sin[i]  cos[i]]
  // Q = G1 * G2 * ... * G_{n-1}
  Col<eT> rot_cos;
  Col<eT> rot_sin;
  bool    computed;
  };



// Perform the QR decomposition of a tridiagonal matrix, a special
// case of upper Hessenberg matrices.
template<typename eT>
class TridiagQR : public UpperHessenbergQR<eT>
  {
  public:

  inline TridiagQR();

  inline TridiagQR(const Mat<eT>& mat);

  inline void compute(const Mat<eT>& mat);

  inline Mat<eT> matrix_RQ();
  };
