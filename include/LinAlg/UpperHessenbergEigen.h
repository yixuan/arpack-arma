#ifndef UPPER_HESSENBERG_EIGEN_H
#define UPPER_HESSENBERG_EIGEN_H

#include <armadillo>
#include <stdexcept>
#include "LapackWrapperExtra.h"

// Eigenvalues and eigenvectors of an upper Hessenberg matrix
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
    UpperHessenbergEigen() :
        n(0), computed(false)
    {}

    UpperHessenbergEigen(const Matrix &mat) :
        n(mat.n_rows), computed(false)
    {
        compute(mat);
    }

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

    ComplexVector eigenvalues()
    {
        if(!computed)
            throw std::logic_error("UpperHessenbergEigen: need to call compute() first");

        return evals;
    }

    ComplexMatrix eigenvectors()
    {
        if(!computed)
            throw std::logic_error("UpperHessenbergEigen: need to call compute() first");

        Scalar prec = std::pow(std::numeric_limits<Scalar>::epsilon(), Scalar(2.0 / 3));
        ComplexMatrix evecs(n, n);
        ComplexVector tmp(n);
        for(int i = 0; i < n; i++)
        {
            if(is_real(evals[i], prec))
            {
                tmp.zeros();
                tmp.set_real(mat_Z.col(i));
                evecs.col(i) = arma::normalise(tmp);
            } else {
                tmp.set_real(mat_Z.col(i));
                tmp.set_imag(mat_Z.col(i + 1));
                evecs.col(i)     = arma::normalise(tmp);
                evecs.col(i + 1) = arma::conj(evecs.col(i));

                i++;
            }
        }

        return evecs;
    }
};



#endif // UPPER_HESSENBERG_EIGEN_H
