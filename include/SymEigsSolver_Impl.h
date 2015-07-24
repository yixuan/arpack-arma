// Arnoldi factorization starting from step-k
template < typename Scalar,
           int SelectionRule,
           typename OpType >
inline void SymEigsSolver<Scalar, SelectionRule, OpType>::factorize_from(int from_k, int to_m, const Vector &fk)
{
    if(to_m <= from_k) return;

    fac_f = fk;

    Vector v(dim_n), w(dim_n);
    Scalar beta = 0.0, Hii = 0.0;
    // Keep the upperleft k x k submatrix of H and set other elements to 0
    fac_H.tail_cols(ncv - from_k).zeros();
    fac_H.submat(arma::span(from_k, ncv - 1), arma::span(0, from_k - 1)).zeros();
    for(int i = from_k; i <= to_m - 1; i++)
    {
        beta = arma::norm(fac_f);
        v = fac_f / beta;
        fac_V.col(i) = v; // The (i+1)-th column
        fac_H(i, i - 1) = beta;

#ifdef USE_PROFILER
PROFILER_START(mat_vec_prod);
#endif
        op->perform_op(v.memptr(), w.memptr());
        nmatop++;
#ifdef USE_PROFILER
PROFILER_END();
#endif

        Hii = arma::dot(v, w);
        fac_H(i - 1, i) = beta;
        fac_H(i, i) = Hii;

        fac_f = w - beta * fac_V.col(i - 1) - Hii * v;
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

// Implicitly restarted Arnoldi factorization
template < typename Scalar,
           int SelectionRule,
           typename OpType >
inline void SymEigsSolver<Scalar, SelectionRule, OpType>::restart(int k)
{
    if(k >= ncv)
        return;

    TridiagQR<Scalar> decomp;
    Vector em(ncv, arma::fill::zeros);
    em[ncv - 1] = 1;

    for(int i = k; i < ncv; i++)
    {
        // QR decomposition of H-mu*I, mu is the shift
        fac_H.diag() -= ritz_val[i];
        decomp.compute(fac_H);

        // V -> VQ
        decomp.apply_YQ(fac_V);
        // H -> Q'HQ
        // Since QR = H - mu * I, we have H = QR + mu * I
        // and therefore Q'HQ = RQ + mu * I
        fac_H = decomp.matrix_RQ();
        fac_H.diag() += ritz_val[i];
        // em -> Q'em
        decomp.apply_QtY(em);
    }
    Vector fk = fac_f * em[k - 1] + fac_V.col(k) * fac_H(k, k - 1);
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
    Vector rv = arma::abs(ritz_val.head(nev));
    Vector thresh = tol * arma::clamp(rv, prec, std::max(prec, rv.max()));
    Vector resid = arma::abs(ritz_vec.tail_rows(1).t()) * arma::norm(fac_f);
    // Converged "wanted" ritz values
    ritz_conv = (resid < thresh);

    return arma::sum(ritz_conv);
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
    TridiagEigen<double> decomp(fac_H);
    Vector evals = decomp.eigenvalues();
    Matrix evecs = decomp.eigenvectors();

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
    // The order would be
    // Largest => Smallest => 2nd largest => 2nd smallest => ...
    // We keep this order since the first k values will always be
    // the wanted collection, no matter k is nev_updated (used in restart())
    // or is nev (used in sort_ritzpair())
    if(SelectionRule == BOTH_ENDS)
    {
        std::vector<SortPair> pairs_copy(pairs);
        for(int i = 0; i < ncv; i++)
        {
            // If i is even, pick values from the left (large values)
            // If i is odd, pick values from the right (small values)
            if(i % 2 == 0)
                pairs[i] = pairs_copy[i / 2];
            else
                pairs[i] = pairs_copy[ncv - 1 - i / 2];
        }
    }

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



// Sort the first nev Ritz pairs in decreasing magnitude order
// This is used to return the final results
template < typename Scalar,
           int SelectionRule,
           typename OpType >
inline void SymEigsSolver<Scalar, SelectionRule, OpType>::sort_ritzpair()
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
    ritz_conv.zeros(nev);

    nmatop = 0;
    niter = 0;

    Vector v(init_resid, dim_n);
    Scalar vnorm = arma::norm(v);
    if(vnorm < prec)
        throw std::invalid_argument("initial residual vector cannot be zero");
    v /= vnorm;

    Vector w(dim_n);
#ifdef USE_PROFILER
PROFILER_START(mat_vec_prod);
#endif
    op->perform_op(v.memptr(), w.memptr());
    nmatop++;
#ifdef USE_PROFILER
PROFILER_END();
#endif

    fac_H(0, 0) = arma::dot(v, w);
    fac_f = w - v * fac_H(0, 0);
    fac_V.col(0) = v;
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
    int i, nconv, nev_adj;
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
template < typename Scalar,
           int SelectionRule,
           typename OpType >
inline typename SymEigsSolver<Scalar, SelectionRule, OpType>::Matrix SymEigsSolver<Scalar, SelectionRule, OpType>::eigenvectors()
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
