// Copyright (C) 2015 Yixuan Qiu
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef UPPER_HESSENBERG_EIGEN_H
#define UPPER_HESSENBERG_EIGEN_H

#include <armadillo>
#include <cmath>
#include <complex>
#include <stdexcept>
#include "LapackWrapperExtra.h"

///
/// \ingroup LinearAlgebra
///
/// Calculate the eigenvalues and eigenvectors of an upper Hessenberg matrix.
///
/// \tparam Scalar The element type of the matrix.
/// Currently supported types are `float` and `double`.
///
/// This class is a wrapper of the Lapack functions `_lahqr` and `_trevc`.
///
template <typename Scalar = double>
class UpperHessenbergEigen
{
private:
    typedef arma::Mat<Scalar> Matrix;
    typedef arma::Col<Scalar> Vector;
    typedef arma::Row<Scalar> RowVector;

    typedef std::complex<Scalar> Complex;
    typedef arma::Mat<Complex> ComplexMatrix;
    typedef arma::Col<Complex> ComplexVector;

    int n;
    Matrix mat_Z;         // In the first stage, H = ZTZ', Z is an orthogonal matrix
                          // In the second stage, Z will be overwritten by the eigenvectors of H
    Matrix mat_T;         // H = ZTZ', T is a Schur form matrix
    ComplexVector evals;  // eigenvalues of H

    bool computed;

    static bool is_real(Complex v, Scalar eps)
    {
        return std::abs(v.imag()) <= eps;
    }

public:
    ///
    /// Default constructor. Computation can
    /// be performed later by calling the compute() method.
    ///
    UpperHessenbergEigen() :
        n(0), computed(false)
    {}

    ///
    /// Constructor to create an object that calculates the eigenvalues
    /// and eigenvectors of an upper Hessenberg matrix `mat`.
    ///
    /// \param mat Matrix type can be `arma::mat` or `arma::fmat`, depending on
    /// the template parameter `Scalar` defined.
    /// Only the upper triangular and the lower subdiagonal parts of
    /// the matrix are used.
    ///
    UpperHessenbergEigen(const Matrix &mat) :
        n(mat.n_rows), computed(false)
    {
        compute(mat);
    }

    ///
    /// Compute the eigenvalue decomposition of an upper Hessenberg matrix.
    ///
    /// \param mat Matrix type can be `arma::mat` or `arma::fmat`, depending on
    /// the template parameter `Scalar` defined.
    /// Only the upper triangular and the lower subdiagonal parts of
    /// the matrix are used.
    ///
    void compute(const Matrix &mat)
    {
        if(!mat.is_square())
            throw std::invalid_argument("UpperHessenbergEigen: matrix must be square");

        n = mat.n_rows;
        mat_Z.set_size(n, n);
        mat_T.set_size(n, n);
        evals.set_size(n);

        mat_Z.eye();
        mat_T = mat;

        int want_T = 1, want_Z = 1;
        int ilo = 1, ihi = n, iloz = 1, ihiz = n;
        Scalar *wr = new Scalar[n];
        Scalar *wi = new Scalar[n];
        int info;
        arma::lapack::lahqr(&want_T, &want_Z, &n, &ilo, &ihi,
                            mat_T.memptr(), &n, wr, wi, &iloz, &ihiz,
                            mat_Z.memptr(), &n, &info);

        for(int i = 0; i < n; i++)
        {
            evals[i] = Complex(wr[i], wi[i]);
        }
        delete [] wr;
        delete [] wi;

        if(info < 0)
            throw std::logic_error("Lapack lahqr: failed to compute all the eigenvalues");

        char side = 'R', howmny = 'B';
        Scalar *work = new Scalar[3 * n];
        int m;

        arma::lapack::trevc(&side, &howmny, (int*) NULL, &n, mat_T.memptr(), &n,
                            (Scalar*) NULL, &n, mat_Z.memptr(), &n, &n, &m, work, &info);
        delete [] work;

        if(info < 0)
            throw std::invalid_argument("Lapack trevc: illegal value");

        computed = true;
    }

    ///
    /// Retrieve the eigenvalues.
    ///
    /// \return Returned vector type will be `arma::cx_vec` or `arma::cx_fvec`, depending on
    /// the template parameter `Scalar` defined.
    ///
    ComplexVector eigenvalues()
    {
        if(!computed)
            throw std::logic_error("UpperHessenbergEigen: need to call compute() first");

        return evals;
    }

    ///
    /// Retrieve the eigenvectors.
    ///
    /// \return Returned matrix type will be `arma::cx_mat` or `arma::cx_fmat`, depending on
    /// the template parameter `Scalar` defined.
    ///
    ComplexMatrix eigenvectors()
    {
        if(!computed)
            throw std::logic_error("UpperHessenbergEigen: need to call compute() first");

        Scalar prec = std::pow(std::numeric_limits<Scalar>::epsilon(), Scalar(2.0) / 3);
        ComplexMatrix evecs(n, n);
        Complex *col_ptr = evecs.memptr();
        for(int i = 0; i < n; i++)
        {
            if(is_real(evals[i], prec))
            {
                // For real eigenvector, normalize and copy
                Scalar z_norm = arma::norm(mat_Z.col(i));
                for(int j = 0; j < n; j++)
                {
                    col_ptr[j] = Complex(mat_Z(j, i) / z_norm, 0);
                }

                col_ptr += n;
            } else {
                // Complex eigenvectors are stored in consecutive columns
                Scalar r2 = arma::dot(mat_Z.col(i), mat_Z.col(i));
                Scalar i2 = arma::dot(mat_Z.col(i + 1), mat_Z.col(i + 1));
                Scalar z_norm = std::sqrt(r2 + i2);
                Scalar *z_ptr = mat_Z.colptr(i);
                for(int j = 0; j < n; j++)
                {
                    col_ptr[j] = Complex(z_ptr[j] / z_norm, z_ptr[j + n] / z_norm);
                    col_ptr[j + n] = std::conj(col_ptr[j]);
                }

                i++;
                col_ptr += 2 * n;
            }
        }

        return evecs;
    }
};



#endif // UPPER_HESSENBERG_EIGEN_H
