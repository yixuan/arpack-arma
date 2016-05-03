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


//! Tiny functions to check attributes of complex numbers
template<typename eT>
class cx_attrib
  {
  public:

  arma_inline static bool is_real(std::complex<eT> v, eT eps)
    {
    return std::abs(v.imag()) <= eps;
    }

  arma_inline static bool is_complex(std::complex<eT> v, eT eps)
    {
    return std::abs(v.imag()) > eps;
    }

  arma_inline static bool is_conj(std::complex<eT> v1, std::complex<eT> v2, eT eps)
    {
    return std::abs(v1 - std::conj(v2)) <= eps;
    }

  };


}  // namespace alt_eigs
