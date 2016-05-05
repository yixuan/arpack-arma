// Copyright (C) 2016 National ICT Australia (NICTA)
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.
// -------------------------------------------------------------------
//
// Written by Yixuan Qiu


#include <utility>    // std::pair


namespace alt_eigs
{


//! The enumeration of selection rules of desired eigenvalues.
struct EigsSelect
  {
  enum SELECT_EIGENVALUE
    {
    LARGEST_MAGN = 0,  //!< Select eigenvalues with largest magnitude. Magnitude
                       //!< means the absolute value for real numbers and norm for
                       //!< complex numbers. Applies to both symmetric and general
                       //!< eigen solvers.

    LARGEST_REAL,      //!< Select eigenvalues with largest real part. Only for general eigen solvers.

    LARGEST_IMAG,      //!< Select eigenvalues with largest imaginary part (in magnitude). Only for general eigen solvers.

    LARGEST_ALGE,      //!< Select eigenvalues with largest algebraic value, considering
                       //!< any negative sign. Only for symmetric eigen solvers.

    SMALLEST_MAGN,     //!< Select eigenvalues with smallest magnitude. Applies to both symmetric and general
                       //!< eigen solvers.

    SMALLEST_REAL,     //!< Select eigenvalues with smallest real part. Only for general eigen solvers.

    SMALLEST_IMAG,     //!< Select eigenvalues with smallest imaginary part (in magnitude). Only for general eigen solvers.

    SMALLEST_ALGE,     //!< Select eigenvalues with smallest algebraic value. Only for symmetric eigen solvers.

    BOTH_ENDS          //!< Select eigenvalues half from each end of the spectrum. When
                       //!< `nev` is odd, compute more from the high end. Only for symmetric eigen solvers.
    };
  };


// When comparing eigenvalues, we first calculate the "target"
// to sort. For example, if we want to choose the eigenvalues with
// largest magnitude, the target will be -std::abs(x).
// The minus sign is due to the fact that std::sort() sorts in ascending order.

// Default target: throw an exceptoin
template<typename eT, int SelectionRule>
struct SortingTarget
  {
  arma_inline static typename get_pod_type<eT>::result get(const eT& val)
    {
    arma_stop("alt_eigs::SortingTarget: incompatible selection rule");
    return -std::abs(val);
    }
  };

// Specialization for LARGEST_MAGN
// This covers [float, double, complex] x [LARGEST_MAGN]
template<typename eT>
struct SortingTarget<eT, EigsSelect::LARGEST_MAGN>
  {
  arma_inline static typename get_pod_type<eT>::result get(const eT& val)
    {
    return -std::abs(val);
    }
  };

// Specialization for LARGEST_REAL
// This covers [complex] x [LARGEST_REAL]
template<typename eT>
struct SortingTarget<std::complex<eT>, EigsSelect::LARGEST_REAL>
  {
  arma_inline static eT get(const std::complex<eT>& val)
    {
    return -val.real();
    }
  };

// Specialization for LARGEST_IMAG
// This covers [complex] x [LARGEST_IMAG]
template<typename eT>
struct SortingTarget<std::complex<eT>, EigsSelect::LARGEST_IMAG>
  {
  arma_inline static eT get(const std::complex<eT>& val)
    {
    return -std::abs(val.imag());
    }
  };

// Specialization for LARGEST_ALGE
// This covers [float, double] x [LARGEST_ALGE]
template<typename eT>
struct SortingTarget<eT, EigsSelect::LARGEST_ALGE>
  {
  arma_inline static eT get(const eT& val)
    {
    return -val;
    }
  };

// Here BOTH_ENDS is the same as LARGEST_ALGE, but
// we need some additional steps, which are done in
// SymEigsSolver.h => retrieve_ritzpair().
// There we move the smallest values to the proper locations.
template<typename eT>
struct SortingTarget<eT, EigsSelect::BOTH_ENDS>
  {
  arma_inline static eT get(const eT& val)
    {
    return -val;
    }
  };

// Specialization for SMALLEST_MAGN
// This covers [float, double, complex] x [SMALLEST_MAGN]
template<typename eT>
struct SortingTarget<eT, EigsSelect::SMALLEST_MAGN>
  {
  arma_inline static typename get_pod_type<eT>::result get(const eT& val)
    {
    return std::abs(val);
    }
  };

// Specialization for SMALLEST_REAL
// This covers [complex] x [SMALLEST_REAL]
template<typename eT>
struct SortingTarget<std::complex<eT>, EigsSelect::SMALLEST_REAL>
  {
  arma_inline static eT get(const std::complex<eT>& val)
    {
    return val.real();
    }
  };

// Specialization for SMALLEST_IMAG
// This covers [complex] x [SMALLEST_IMAG]
template<typename eT>
struct SortingTarget<std::complex<eT>, EigsSelect::SMALLEST_IMAG>
  {
  arma_inline static eT get(const std::complex<eT>& val)
    {
    return std::abs(val.imag());
    }
  };

// Specialization for SMALLEST_ALGE
// This covers [float, double] x [SMALLEST_ALGE]
template<typename eT>
struct SortingTarget<eT, EigsSelect::SMALLEST_ALGE>
  {
  arma_inline static eT get(const eT& val)
    {
    return val;
    }
  };

// Sort eigenvalues and return the order index
template<typename PairType>
struct PairComparator
  {
  arma_inline bool operator() (const PairType& v1, const PairType& v2)
    {
    return v1.first < v2.first;
    }
  };

template<typename eT, int SelectionRule>
class SortEigenvalue
  {
  private:

  typedef typename get_pod_type<eT>::result TargetType;  // Type of the sorting target, will be
                                                         // a floating number type, e.g. "double"
  typedef std::pair<TargetType, uword> PairType;         // Type of the sorting pair, including
                                                         // the sorting target and the index
  std::vector<PairType> pair_sort;


  public:

  inline SortEigenvalue(const eT* start, uword size)
    : pair_sort(size)
    {
    for(uword i = 0; i < size; i++)
      {
      pair_sort[i].first = SortingTarget<eT, SelectionRule>::get(start[i]);
      pair_sort[i].second = i;
      }
    PairComparator<PairType> comp;
    std::sort(pair_sort.begin(), pair_sort.end(), comp);
    }

  inline std::vector<uword> index()
    {
    const uword len = pair_sort.size();
    std::vector<uword> ind(len);
    for(uword i = 0; i < len; i++) { ind[i] = pair_sort[i].second; }

    return ind;
    }
  };


}  // namespace alt_eigs
