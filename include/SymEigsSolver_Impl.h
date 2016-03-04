// Copyright (C) 2015 Yixuan Qiu
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

// Arnoldi factorization starting from step-k
template < typename Scalar,
           int SelectionRule,
           typename OpType >
inline void SymEigsSolver<Scalar, SelectionRule, OpType>::factorize_from(int from_k, int to_m, const Vector &fk)
{
    if(to_m <= from_k) return;

    fac_f = fk;

    Vector w(dim_n);
    Scalar beta = std::sqrt(arma::dot(fac_f, fac_f)), Hii = 0.0;
    // Keep the upperleft k x k submatrix of H and set other elements to 0
    fac_H.tail_cols(ncv - from_k).zeros();
    fac_H.submat(arma::span(from_k, ncv - 1), arma::span(0, from_k - 1)).zeros();
    for(int i = from_k; i <= to_m - 1; i++)
    {
        bool restart = false;
        // If beta = 0, then the next V is not full rank
        // We need to generate a new residual vector that is orthogonal
        // to the current V, which we call a restart
        if(beta < prec)
        {
            fac_f.randu();
            // f <- f - V * V' * f, so that f is orthogonal to V
            Matrix Vs(fac_V.memptr(), dim_n, i, false); // First i columns
            Vector Vf = Vs.t() * fac_f;
            fac_f -= Vs * Vf;
            // beta <- ||f||
            beta = std::sqrt(arma::dot(fac_f, fac_f));

            restart = true;
        }

        // v <- f / ||f||
        Vector v(fac_V.colptr(i), dim_n, false); // The (i+1)-th column
        v = fac_f / beta;

        // Note that H[i+1, i] equals to the unrestarted beta
        if(restart)
            fac_H(i, i - 1) = 0.0;
        else
            fac_H(i, i - 1) = beta;

        // w <- A * v, v = fac_V.col(i)
        op->perform_op(v.memptr(), w.memptr());
        nmatop++;

        Hii = arma::dot(v, w);
        fac_H(i - 1, i) = fac_H(i, i - 1); // Due to symmetry
        fac_H(i, i) = Hii;

        // f <- w - V * V' * w = w - H[i+1, i] * V{i} - H[i+1, i+1] * V{i+1}
        // If restarting, we know that H[i+1, i] = 0
        if(restart)
            fac_f = w - Hii * v;
        else
            fac_f = w - fac_H(i, i - 1) * fac_V.col(i - 1) - Hii * v;

        beta = std::sqrt(arma::dot(fac_f, fac_f));

        // f/||f|| is going to be the next column of V, so we need to test
        // whether V' * (f/||f||) ~= 0
        Matrix Vs(fac_V.memptr(), dim_n, i + 1, false); // First i+1 columns
        Vector Vf = Vs.t() * fac_f;
        // If not, iteratively correct the residual
        int count = 0;
        while(count < 5 && arma::abs(Vf).max() > prec * beta)
        {
            // f <- f - V * Vf
            fac_f -= Vs * Vf;
            // h <- h + Vf
            fac_H(i - 1, i) += Vf[i - 1];
            fac_H(i, i - 1) = fac_H(i - 1, i);
            fac_H(i, i) += Vf[i];
            // beta <- ||f||
            beta = std::sqrt(arma::dot(fac_f, fac_f));

            Vf = Vs.t() * fac_f;
            count++;
        }
    }
}

// Implicitly restarted Arnoldi factorization
template < typename Scalar,
           int SelectionRule,
           typename OpType >
inline void SymEigsSolver<Scalar, SelectionRule, OpType>::restart(int k)
{
    if(k >= ncv)
        return;

    TridiagQR<Scalar> decomp;
    Matrix Q = arma::eye<Matrix>(ncv, ncv);

    for(int i = k; i < ncv; i++)
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
    Matrix Vs(dim_n, k + 1);
    int nnz;
    for(int i = 0; i < k; i++)
    {
        nnz = ncv - k + i + 1;
        Matrix V(fac_V.memptr(), dim_n, nnz, false);
        Vector q(Q.colptr(i), nnz, false);
        Vs.col(i) = V * q;
    }
    Vs.col(k) = fac_V * Q.col(k);
    fac_V.head_cols(k + 1) = Vs;

    Vector fk = fac_f * Q(ncv - 1, k - 1) + fac_V.col(k) * fac_H(k, k - 1);
    factorize_from(k, ncv, fk);
    retrieve_ritzpair();
}

// Calculate the number of converged Ritz values
template < typename Scalar,
           int SelectionRule,
           typename OpType >
inline int SymEigsSolver<Scalar, SelectionRule, OpType>::num_converged(Scalar tol)
{
    // thresh = tol * max(prec, abs(theta)), theta for ritz value
    const Scalar f_norm = arma::norm(fac_f);
    for(int i = 0; i < nev; i++)
    {
        Scalar thresh = tol * std::max(prec, std::abs(ritz_val[i]));
        Scalar resid = std::abs(ritz_vec(ncv - 1, i)) * f_norm;
        ritz_conv[i] = (resid < thresh);
    }

    return std::count(ritz_conv.begin(), ritz_conv.end(), true);
}

// Return the adjusted nev for restarting
template < typename Scalar,
           int SelectionRule,
           typename OpType >
inline int SymEigsSolver<Scalar, SelectionRule, OpType>::nev_adjusted(int nconv)
{
    int nev_new = nev;

    // Adjust nev_new, according to dsaup2.f line 677~684 in ARPACK
    nev_new = nev + std::min(nconv, (ncv - nev) / 2);
    if(nev == 1 && ncv >= 6)
        nev_new = ncv / 2;
    else if(nev == 1 && ncv > 2)
        nev_new = 2;

    return nev_new;
}

// Retrieve and sort ritz values and ritz vectors
template < typename Scalar,
           int SelectionRule,
           typename OpType >
inline void SymEigsSolver<Scalar, SelectionRule, OpType>::retrieve_ritzpair()
{
    /*Vector evals(ncv);
    Matrix evecs(ncv, ncv);
    arma::eig_sym(evals, evecs, arma::symmatl(fac_H));*/
    TridiagEigen<Scalar> decomp(fac_H);
    Vector evals = decomp.eigenvalues();
    Matrix evecs = decomp.eigenvectors();

    SortEigenvalue<Scalar, SelectionRule> sorting(evals.memptr(), evals.n_elem);
    std::vector<int> ind = sorting.index();

    // For BOTH_ENDS, the eigenvalues are sorted according
    // to the LARGEST_ALGE rule, so we need to move those smallest
    // values to the left
    // The order would be
    // Largest => Smallest => 2nd largest => 2nd smallest => ...
    // We keep this order since the first k values will always be
    // the wanted collection, no matter k is nev_updated (used in restart())
    // or is nev (used in sort_ritzpair())
    if(SelectionRule == BOTH_ENDS)
    {
        std::vector<int> ind_copy(ind);
        for(int i = 0; i < ncv; i++)
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
    for(int i = 0; i < ncv; i++)
    {
        ritz_val[i] = evals[ind[i]];
    }
    for(int i = 0; i < nev; i++)
    {
        ritz_vec.col(i) = evecs.col(ind[i]);
    }
}



// Sort the first nev Ritz pairs in decreasing magnitude order
// This is used to return the final results
template < typename Scalar,
           int SelectionRule,
           typename OpType >
inline void SymEigsSolver<Scalar, SelectionRule, OpType>::sort_ritzpair()
{
    SortEigenvalue<Scalar, LARGEST_MAGN> sorting(ritz_val.memptr(), nev);
    std::vector<int> ind = sorting.index();

    Vector new_ritz_val(ncv);
    Matrix new_ritz_vec(ncv, nev);
    BoolVector new_ritz_conv(nev);

    for(int i = 0; i < nev; i++)
    {
        new_ritz_val[i] = ritz_val[ind[i]];
        new_ritz_vec.col(i) = ritz_vec.col(ind[i]);
        new_ritz_conv[i] = ritz_conv[ind[i]];
    }

    ritz_val.swap(new_ritz_val);
    ritz_vec.swap(new_ritz_vec);
    ritz_conv.swap(new_ritz_conv);
}



// Initialization and clean-up
template < typename Scalar,
           int SelectionRule,
           typename OpType >
inline void SymEigsSolver<Scalar, SelectionRule, OpType>::init(Scalar *init_resid)
{
    // Reset all matrices/vectors to zero
    fac_V.zeros(dim_n, ncv);
    fac_H.zeros(ncv, ncv);
    fac_f.zeros(dim_n);
    ritz_val.zeros(ncv);
    ritz_vec.zeros(ncv, nev);
    ritz_conv.assign(nev, false);

    nmatop = 0;
    niter = 0;

    Vector r(init_resid, dim_n, false);
    // The first column of fac_V
    Vector v(fac_V.colptr(0), dim_n, false);
    Scalar rnorm = arma::norm(r);
    if(rnorm < prec)
        throw std::invalid_argument("initial residual vector cannot be zero");
    v = r / rnorm;

    Vector w(dim_n);
    op->perform_op(v.memptr(), w.memptr());
    nmatop++;

    fac_H(0, 0) = arma::dot(v, w);
    fac_f = w - v * fac_H(0, 0);
}

// Initialization with random initial coefficients
template < typename Scalar,
           int SelectionRule,
           typename OpType >
inline void SymEigsSolver<Scalar, SelectionRule, OpType>::init()
{
    Vector init_resid(dim_n, arma::fill::randu);
    init_resid -= 0.5;
    init(init_resid.memptr());
}

// Compute Ritz pairs and return the number of converged eigenvalues
template < typename Scalar,
           int SelectionRule,
           typename OpType >
inline int SymEigsSolver<Scalar, SelectionRule, OpType>::compute(int maxit, Scalar tol)
{
    // The m-step Arnoldi factorization
    factorize_from(1, ncv, fac_f);
    retrieve_ritzpair();
    // Restarting
    int i, nconv = 0, nev_adj;
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

// Return converged eigenvalues
template < typename Scalar,
           int SelectionRule,
           typename OpType >
inline typename SymEigsSolver<Scalar, SelectionRule, OpType>::Vector SymEigsSolver<Scalar, SelectionRule, OpType>::eigenvalues()
{
    int nconv = std::count(ritz_conv.begin(), ritz_conv.end(), true);
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
template < typename Scalar,
           int SelectionRule,
           typename OpType >
inline typename SymEigsSolver<Scalar, SelectionRule, OpType>::Matrix SymEigsSolver<Scalar, SelectionRule, OpType>::eigenvectors(int nvec)
{
    int nconv = std::count(ritz_conv.begin(), ritz_conv.end(), true);
    nvec = std::min(nvec, nconv);
    Matrix res(dim_n, nvec);

    if(!nvec)
        return res;

    Matrix ritz_vec_conv(ncv, nvec);
    int j = 0;
    for(int i = 0; i < nev && j < nvec; i++)
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
