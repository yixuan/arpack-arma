#ifndef GENEIGSSOLVER_H
#define GENEIGSSOLVER_H

#include <armadillo>
#include <vector>
#include <algorithm>
#include <cmath>
#include <complex>
#include <utility>
#include <stdexcept>

#include "MatOp.h"
#include "SelectionRule.h"
#include "UpperHessenbergQR.h"


template <typename Scalar = double, int SelectionRule = LARGEST_MAGN>
class GenEigsSolver
{
private:
    typedef arma::Mat<Scalar> Matrix;
    typedef arma::Col<Scalar> Vector;
    typedef arma::uvec BoolVector;

    typedef std::complex<Scalar> Complex;
    typedef arma::Mat<Complex> ComplexMatrix;
    typedef arma::Col<Complex> ComplexVector;

    typedef std::pair<Complex, int> SortPair;

    MatOp<Scalar> *op;      // object to conduct matrix operation,
                            // e.g. matrix-vector product
    const int dim_n;        // dimension of matrix A

protected:
    const int nev;          // number of eigenvalues requested

private:
    int nev_updated;        // increase nev in factorization if needed
    const int ncv;          // number of ritz values
    int nmatop;             // number of matrix operations called
    int niter;              // number of restarting iterations

    Matrix fac_V;           // V matrix in the Arnoldi factorization
    Matrix fac_H;           // H matrix in the Arnoldi factorization
    Vector fac_f;           // residual in the Arnoldi factorization

protected:
    ComplexVector ritz_val; // ritz values

private:
    ComplexMatrix ritz_vec; // ritz vectors
    BoolVector ritz_conv;   // indicator of the convergence of ritz values

    const Scalar prec;      // precision parameter used to test convergence
                            // prec = epsilon^(2/3)
                            // epsilon is the machine precision,
                            // e.g. ~= 1e-16 for the "double" type

    // Matrix product in this case, and shift solve for SymEigsShiftSolver
    virtual void matrix_operation(Scalar *x_in, Scalar *y_out)
    {
        op->prod(x_in, y_out);
    }

    // Arnoldi factorization starting from step-k
    void factorize_from(int from_k, int to_m, const Vector &fk)
    {
        if(to_m <= from_k) return;

        fac_f = fk;

        Vector v(dim_n), w(dim_n);
        Scalar beta = 0.0;
        // Keep the upperleft k x k submatrix of H and set other elements to 0
        fac_H.tail_cols(ncv - from_k).zeros();
        fac_H.submat(arma::span(from_k, ncv - 1), arma::span(0, from_k - 1)).zeros();
        for(int i = from_k; i <= to_m - 1; i++)
        {
            beta = arma::norm(fac_f);
            v = fac_f / beta;
            fac_V.col(i) = v; // The (i+1)-th column
            fac_H(i, i - 1) = beta;

            matrix_operation(v.memptr(), w.memptr());
            nmatop++;

            Vector h = fac_V.head_cols(i + 1).t() * w;
            fac_H(arma::span(0, i), i) = h;

            fac_f = w - fac_V.head_cols(i + 1) * h;
            // Correct f if it is not orthogonal to V
            // Typically the largest absolute value occurs in
            // the first element, i.e., <v1, f>, so we use this
            // to test the orthogonality
            Scalar v1f = arma::dot(fac_f, fac_V.col(0));
            if(v1f > prec || v1f < -prec)
            {
                Vector Vf(i + 1);
                Vf.tail(i) = fac_V.cols(1, i).t() * fac_f;
                Vf[0] = v1f;
                fac_f -= fac_V.head_cols(i + 1) * Vf;
            }
        }
    }

    static bool is_complex(Complex v, Scalar eps)
    {
        return std::abs(v.imag()) > eps;
    }

    static bool is_conj(Complex v1, Complex v2, Scalar eps)
    {
        return std::abs(v1 - std::conj(v2)) < eps;
    }

    // Implicitly restarted Arnoldi factorization
    void restart(int k)
    {
        if(k >= ncv)
            return;

        UpperHessenbergQR<Scalar> decomp;
        Matrix Q, R;
        Vector em(ncv, arma::fill::zeros);
        em[ncv - 1] = 1;

        for(int i = k; i < ncv; i++)
        {
            if(is_complex(ritz_val[i], prec) && is_conj(ritz_val[i], ritz_val[i + 1], prec))
            {
                // H - mu * I = Q1 * R1
                // H <- R1 * Q1 + mu * I = Q1' * H * Q1
                // H - conj(mu) * I = Q2 * R2
                // H <- R2 * Q2 + conj(mu) * I = Q2' * H * Q2
                //
                // (H - mu * I) * (H - conj(mu) * I) = Q1 * Q2 * R2 * R1 = Q * R
                Scalar re = ritz_val[i].real();
                Scalar s = std::norm(ritz_val[i]);
                Matrix HH = fac_H * fac_H - 2 * re * fac_H;
                HH.diag() += s;
                // NOTE: HH is no longer upper Hessenburg
                arma::qr(Q, R, HH);

                // V -> VQ
                fac_V = fac_V * Q;
                // H -> Q'HQ
                fac_H = Q.t() * fac_H * Q;
                // em -> Q'em
                em = Q.t() * em;

                i++;
            } else {
                // QR decomposition of H - mu * I, mu is real
                fac_H.diag() -= ritz_val[i].real();
                decomp.compute(fac_H);
                fac_H.diag() += ritz_val[i].real();

                // V -> VQ
                decomp.apply_YQ(fac_V);
                // H -> Q'HQ
                decomp.apply_QtY(fac_H);
                decomp.apply_YQ(fac_H);
                // em -> Q'em
                decomp.apply_QtY(em);
            }
        }
        Vector fk = fac_f * em[k - 1] + fac_V.col(k) * fac_H(k, k - 1);
        factorize_from(k, ncv, fk);
        retrieve_ritzpair();
    }

    // Test convergence
    bool converged(Scalar tol)
    {
        // thresh = tol * max(prec, abs(theta)), theta for ritz value
        Vector rv = arma::abs(ritz_val.head(nev));
        Vector thresh = tol * arma::clamp(rv, prec, rv.max());
        Vector resid = arma::abs(ritz_vec.tail_rows(1).t()) * arma::norm(fac_f);
        ritz_conv = (resid < thresh);

        // Converged "wanted" ritz values
        int nconv = arma::sum(ritz_conv);

        // Increase nev by one if ritz_val[nev - 1] and
        // ritz_val[nev] are conjugate pairs
        if(is_complex(ritz_val[nev - 1], prec) &&
           is_conj(ritz_val[nev - 1], ritz_val[nev], prec))
        {
            nev_updated = nev + 1;
        }
        // Adjust nev_updated again, according to dnaup2.f line 660~674 in ARPACK
        nev_updated = nev_updated + std::min(nconv, (ncv - nev_updated) / 2);
        if(nev_updated == 1 && ncv >= 6)
            nev_updated = ncv / 2;
        else if(nev_updated == 1 && ncv > 3)
            nev_updated = 2;

        if(nev_updated > ncv - 2)
            nev_updated = ncv - 2;

        if(is_complex(ritz_val[nev_updated - 1], prec) &&
           is_conj(ritz_val[nev_updated - 1], ritz_val[nev_updated], prec))
        {
            nev_updated++;
        }

        return nconv >= nev;
    }

    // Retrieve and sort ritz values and ritz vectors
    void retrieve_ritzpair()
    {
        ComplexVector evals(ncv);
        ComplexMatrix evecs(ncv, ncv);
        arma::eig_gen(evals, evecs, fac_H);

        std::vector<SortPair> pairs(ncv);
        EigenvalueComparator<Complex, SelectionRule> comp;
        for(int i = 0; i < ncv; i++)
        {
            pairs[i].first = evals[i];
            pairs[i].second = i;
        }
        std::sort(pairs.begin(), pairs.end(), comp);

        // Copy the ritz values and vectors to ritz_val and ritz_vec, respectively
        for(int i = 0; i < ncv; i++)
        {
            ritz_val[i] = pairs[i].first;
        }
        for(int i = 0; i < nev; i++)
        {
            ritz_vec.col(i) = evecs.col(pairs[i].second);
        }
    }

protected:
    // Sort the first nev Ritz pairs in decreasing magnitude order
    // This is used to return the final results
    virtual void sort_ritzpair()
    {
        std::vector<SortPair> pairs(nev);
        EigenvalueComparator<Complex, LARGEST_MAGN> comp;
        for(int i = 0; i < nev; i++)
        {
            pairs[i].first = ritz_val[i];
            pairs[i].second = i;
        }
        std::sort(pairs.begin(), pairs.end(), comp);

        ComplexMatrix new_ritz_vec(ncv, nev);
        BoolVector new_ritz_conv(nev);

        for(int i = 0; i < nev; i++)
        {
            ritz_val[i] = pairs[i].first;
            new_ritz_vec.col(i) = ritz_vec.col(pairs[i].second);
            new_ritz_conv[i] = ritz_conv[pairs[i].second];
        }

        ritz_vec.swap(new_ritz_vec);
        ritz_conv.swap(new_ritz_conv);
    }

public:
    GenEigsSolver(MatOp<Scalar> *op_, int nev_, int ncv_) :
        op(op_),
        dim_n(op->rows()),
        nev(nev_),
        nev_updated(nev_),
        ncv(ncv_ > dim_n ? dim_n : ncv_),
        nmatop(0),
        niter(0),
        prec(std::pow(std::numeric_limits<Scalar>::epsilon(), Scalar(2.0 / 3)))
    {
        if(nev_ < 1 || nev_ >= dim_n)
            throw std::invalid_argument("nev must be greater than zero and less than the size of the matrix");

        if(ncv_ <= nev_)
            throw std::invalid_argument("ncv must be greater than nev");
    }

    // Initialization and clean-up
    virtual void init(Scalar *init_resid)
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

        Vector v(init_resid, dim_n);
        Scalar vnorm = arma::norm(v);
        if(vnorm < prec)
            throw std::invalid_argument("initial residual vector cannot be zero");
        v /= vnorm;

        Vector w(dim_n);
        matrix_operation(v.memptr(), w.memptr());
        nmatop++;

        fac_H(0, 0) = arma::dot(v, w);
        fac_f = w - v * fac_H(0, 0);
        fac_V.col(0) = v;
    }
    // Initialization with random initial coefficients
    virtual void init()
    {
        Vector init_resid(dim_n, arma::fill::randu);
        init_resid -= 0.5;
        init(init_resid.memptr());
    }

    // Compute Ritz pairs and return the number of converged eigenvalues
    int compute(int maxit = 1000, Scalar tol = 1e-10)
    {
        // The m-step Arnoldi factorization
        factorize_from(1, ncv, fac_f);
        retrieve_ritzpair();
        // Restarting
        int i;
        for(i = 0; i < maxit; i++)
        {
            if(converged(tol))
                break;

            restart(nev_updated);
        }
        // Sorting results
        sort_ritzpair();

        niter += i + 1;
        int nconv = arma::sum(ritz_conv);

        return std::min(nev, nconv);
    }

    void info(int &iters, int &mat_ops)
    {
        iters = niter;
        mat_ops = nmatop;
    }

    // Return converged eigenvalues
    ComplexVector eigenvalues()
    {
        int nconv = arma::sum(ritz_conv);
        ComplexVector res(nconv);

        if(!nconv)
            return res;

        int j = 0;
        for(int i = 0; i < nev; i++)
        {
            if(ritz_conv[i])
            {
                res[j] = ritz_val[i];
                j++;
            }
        }

        return res;
    }

    // Return converged eigenvectors
    ComplexMatrix eigenvectors()
    {
        int nconv = arma::sum(ritz_conv);
        ComplexMatrix res(dim_n, nconv);

        if(!nconv)
            return res;

        ComplexMatrix ritz_vec_conv(ncv, nconv);
        int j = 0;
        for(int i = 0; i < nev; i++)
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
};




/*
template <typename Scalar = double, int SelectionRule = LARGEST_MAGN>
class SymEigsShiftSolver: public SymEigsSolver<Scalar, SelectionRule>
{
private:
    typedef arma::Col<Scalar> Vector;

    Scalar sigma;
    MatOpWithRealShiftSolve<Scalar> *op_shift;

    // Shift solve in this case
    void matrix_operation(Scalar *x_in, Scalar *y_out)
    {
        op_shift->shift_solve(x_in, y_out);
    }

    // First transform back the ritz values, and then sort
    void sort_ritzpair()
    {
        Vector ritz_val_org = Scalar(1.0) / this->ritz_val.head(this->nev) + sigma;
        this->ritz_val.head(this->nev) = ritz_val_org;
        SymEigsSolver<Scalar, SelectionRule>::sort_ritzpair();
    }
public:
    SymEigsShiftSolver(MatOpWithRealShiftSolve<Scalar> *op_,
                       int nev_, int ncv_, Scalar sigma_) :
        SymEigsSolver<Scalar, SelectionRule>(op_, nev_, ncv_),
        sigma(sigma_),
        op_shift(op_)
    {
        op_shift->set_shift(sigma);
    }
};
*/

#endif // GENEIGSSOLVER_H
