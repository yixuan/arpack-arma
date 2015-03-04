#ifndef SYMEIGSSOLVER_H
#define SYMEIGSSOLVER_H

#include <armadillo>
#include <vector>
#include <algorithm>
#include <cmath>
#include <utility>

#include "MatOp.h"
#include "SelectionRule.h"


template <typename Scalar = double, int SelectionRule = LARGEST_MAGN>
class SymEigsSolver
{
protected:
    typedef arma::Mat<Scalar> Matrix;
    typedef arma::Col<Scalar> Vector;
    typedef arma::uvec BoolVector;
    typedef std::pair<Scalar, int> SortPair;

    MatOp<Scalar> *op;    // object to conduct matrix operation,
                          // e.g. matrix-vector product
    int dim_n;            // dimension of matrix A
    int nev;              // number of eigenvalues requested
    int ncv;              // number of ritz values

    Matrix fac_V;         // V matrix in the Arnoldi factorization
    Matrix fac_H;         // H matrix in the Arnoldi factorization
    Vector fac_f;         // residual in the Arnoldi factorization

    Vector ritz_val;      // ritz values
    Matrix ritz_vec;      // ritz vectors
    BoolVector ritz_conv; // indicator of the convergence of ritz values

    const Scalar prec;    // precision parameter used to test convergence
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

        Vector v(dim_n);
        Scalar beta = 0.0;
        for(int i = from_k; i <= to_m - 1; i++)
        {
            beta = arma::norm(fac_f);
            v = fac_f / beta;
            fac_V.col(i) = v; // The (i+1)-th column
            fac_H(i, arma::span(0, i - 1)).zeros();
            fac_H(i, i - 1) = beta;

            Vector w(dim_n);
            matrix_operation(v.memptr(), w.memptr());
            Vector h = fac_V.head_cols(i + 1).t() * w;
            fac_f = w - fac_V.head_cols(i + 1) * h;
            fac_H(arma::span(0, i), i) = h;
        }
    }

    // Implicitly restarted Arnoldi factorization
    void restart(int k)
    {
        if(k >= ncv)
            return;

        Matrix Q(ncv, ncv), R(ncv, ncv);
        Vector em(ncv, arma::fill::zeros);
        em[ncv - 1] = 1;

        for(int i = k; i < ncv; i++)
        {
            Matrix H = fac_H;
            fac_H.diag() -= ritz_val[i];
            arma::qr(Q, R, fac_H);

            // V -> VQ
            Matrix V = fac_V;
            fac_V = V * Q;
            // H -> Q'HQ
            fac_H = Q.t() * H * Q;
            // em -> Q'em
            Vector e = em;
            em = Q.t() * e;
        }

        Vector fk = fac_f * em[k - 1];
        factorize_from(k, ncv, fk);
        retrieve_ritzpair();
    }

    // Test convergence
    bool converged(Scalar tol)
    {
        // bound = tol * max(prec, abs(theta)), theta for ritz value
        Vector rv = arma::abs(ritz_val.head(nev));
        Vector bound = tol * arma::clamp(rv, prec, rv.max());
        Vector resid = arma::abs(ritz_vec.tail_rows(1).t()) * arma::norm(fac_f);
        ritz_conv = (resid < bound);

        return arma::all(ritz_conv);
    }

    // Retrieve and sort ritz values and ritz vectors
    void retrieve_ritzpair()
    {
        Vector evals(ncv);
        Matrix evecs(ncv, ncv);
        eig_sym(evals, evecs, symmatl(fac_H));

        std::vector<SortPair> pairs(ncv);
        EigenvalueComparator<Scalar, SelectionRule> comp;
        for(int i = 0; i < ncv; i++)
        {
            pairs[i].first = evals[i];
            pairs[i].second = i;
        }
        std::sort(pairs.begin(), pairs.end(), comp);
        // For BOTH_ENDS, the eigenvalues are sorted according
        // to the LARGEST_ALGE rule, so we need to move those smallest
        // values to the left
        if(SelectionRule == BOTH_ENDS)
        {
            int offset = (nev + 1) / 2;
            for(int i = 0; i < nev - offset; i++)
            {
                std::swap(pairs[offset + i], pairs[ncv - 1 - i]);
            }
        }

        for(int i = 0; i < ncv; i++)
        {
            ritz_val[i] = pairs[i].first;
        }
        for(int i = 0; i < nev; i++)
        {
            ritz_vec.col(i) = evecs.col(pairs[i].second);
        }
    }

    // Sort the first nev Ritz pairs in decreasing magnitude order
    // This is used to return the final results
    virtual void sort_ritzpair()
    {
        std::vector<SortPair> pairs(nev);
        EigenvalueComparator<Scalar, LARGEST_MAGN> comp;
        for(int i = 0; i < nev; i++)
        {
            pairs[i].first = ritz_val[i];
            pairs[i].second = i;
        }
        std::sort(pairs.begin(), pairs.end(), comp);

        Matrix new_ritz_vec(ncv, nev);
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
    SymEigsSolver(MatOp<Scalar> *op_, int nev_, int ncv_) :
        op(op_),
        dim_n(op->rows()),
        nev(nev_),
        ncv(ncv_),
        fac_V(dim_n, ncv, arma::fill::zeros),
        fac_H(ncv, ncv, arma::fill::zeros),
        fac_f(dim_n, arma::fill::zeros),
        ritz_val(ncv, arma::fill::zeros),
        ritz_vec(ncv, nev, arma::fill::zeros),
        ritz_conv(nev, arma::fill::zeros),
        prec(std::pow(std::numeric_limits<Scalar>::epsilon(), Scalar(2.0 / 3)))
    {}

    // Initialization and clean-up
    virtual void init(Scalar *init_coef)
    {
        Vector v(dim_n);
        matrix_operation(init_coef, v.memptr());
        Vector w = arma::normalise(v);
        v.swap(w);

        matrix_operation(v.memptr(), w.memptr());

        fac_H(0, 0) = arma::dot(v, w);
        fac_f = w - v * fac_H(0, 0);
        fac_V.col(0) = v;
    }
    // Initialization with random initial coefficients
    virtual void init()
    {
        Vector init_coef(dim_n);
        init_coef.randu();
        init(init_coef.memptr());
    }

    // Compute Ritz pairs and return the number of iteration
    int compute(int maxit = 1000, Scalar tol = 1e-10)
    {
        // The m-step Arnoldi factorization
        factorize_from(1, ncv, fac_f);
        retrieve_ritzpair();
        // Restarting
        int i = 0;
        for(i = 0; i < maxit; i++)
        {
            if(converged(tol))
                break;

            restart(nev);
        }
        // Sorting results
        sort_ritzpair();

        return i + 1;
    }

    // Return converged eigenvalues
    Vector eigenvalues()
    {
        int nconv = arma::sum(ritz_conv);
        Vector res(nconv);

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
    Matrix eigenvectors()
    {
        int nconv = arma::sum(ritz_conv);
        Matrix res(dim_n, nconv);

        if(!nconv)
            return res;

        Matrix ritz_vec_conv(ncv, nconv);
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


#endif // SYMEIGSSOLVER_H
