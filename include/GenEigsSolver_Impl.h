// Arnoldi factorization starting from step-k
template < typename Scalar,
           int SelectionRule,
           typename OpType >
inline void GenEigsSolver<Scalar, SelectionRule, OpType>::factorize_from(int from_k, int to_m, const Vector &fk)
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

#ifdef USE_PROFILER
PROFILER_START(mat_vec_prod);
#endif
        op->perform_op(v.memptr(), w.memptr());
        nmatop++;
#ifdef USE_PROFILER
PROFILER_END();
#endif

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

// Implicitly restarted Arnoldi factorization
template < typename Scalar,
           int SelectionRule,
           typename OpType >
inline void GenEigsSolver<Scalar, SelectionRule, OpType>::restart(int k)
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
            Matrix HH = fac_H;
            HH.diag() -= 2 * re;
            HH = fac_H * HH;
            HH.diag() += s;
            // NOTE: HH is no longer upper Hessenberg
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

            // V -> VQ
            decomp.apply_YQ(fac_V);
            // H -> Q'HQ = RQ + mu * I
            fac_H = decomp.matrix_RQ();
            fac_H.diag() += ritz_val[i].real();
            // em -> Q'em
            decomp.apply_QtY(em);
        }
    }
    Vector fk = fac_f * em[k - 1] + fac_V.col(k) * fac_H(k, k - 1);
    factorize_from(k, ncv, fk);
    retrieve_ritzpair();
}

// Calculate the number of converged Ritz values
template < typename Scalar,
           int SelectionRule,
           typename OpType >
inline int GenEigsSolver<Scalar, SelectionRule, OpType>::num_converged(Scalar tol)
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
inline int GenEigsSolver<Scalar, SelectionRule, OpType>::nev_adjusted(int nconv)
{
    int nev_new = nev;

    // Increase nev by one if ritz_val[nev - 1] and
    // ritz_val[nev] are conjugate pairs
    if(is_complex(ritz_val[nev - 1], prec) &&
       is_conj(ritz_val[nev - 1], ritz_val[nev], prec))
    {
        nev_new = nev + 1;
    }
    // Adjust nev_new again, according to dnaup2.f line 660~674 in ARPACK
    nev_new = nev_new + std::min(nconv, (ncv - nev_new) / 2);
    if(nev_new == 1 && ncv >= 6)
        nev_new = ncv / 2;
    else if(nev_new == 1 && ncv > 3)
        nev_new = 2;

    if(nev_new > ncv - 2)
        nev_new = ncv - 2;

    // Examine conjugate pairs again
    if(is_complex(ritz_val[nev_new - 1], prec) &&
       is_conj(ritz_val[nev_new - 1], ritz_val[nev_new], prec))
    {
        nev_new++;
    }

    return nev_new;
}

// Retrieve and sort ritz values and ritz vectors
template < typename Scalar,
           int SelectionRule,
           typename OpType >
inline void GenEigsSolver<Scalar, SelectionRule, OpType>::retrieve_ritzpair()
{
    /*ComplexVector evals(ncv);
    ComplexMatrix evecs(ncv, ncv);
    arma::eig_gen(evals, evecs, fac_H);*/
    UpperHessenbergEigen<double> decomp(fac_H);
    ComplexVector evals = decomp.eigenvalues();
    ComplexMatrix evecs = decomp.eigenvectors();

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



// Sort the first nev Ritz pairs in decreasing magnitude order
// This is used to return the final results
template < typename Scalar,
           int SelectionRule,
           typename OpType >
inline void GenEigsSolver<Scalar, SelectionRule, OpType>::sort_ritzpair()
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



// Initialization and clean-up
template < typename Scalar,
           int SelectionRule,
           typename OpType >
inline void GenEigsSolver<Scalar, SelectionRule, OpType>::init(Scalar *init_resid)
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
inline void GenEigsSolver<Scalar, SelectionRule, OpType>::init()
{
    Vector init_resid(dim_n, arma::fill::randu);
    init_resid -= 0.5;
    init(init_resid.memptr());
}

// Compute Ritz pairs and return the number of converged eigenvalues
template < typename Scalar,
           int SelectionRule,
           typename OpType >
inline int GenEigsSolver<Scalar, SelectionRule, OpType>::compute(int maxit, Scalar tol)
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

    niter += i + 1;

    return std::min(nev, nconv);
}

// Return converged eigenvalues
template < typename Scalar,
           int SelectionRule,
           typename OpType >
inline typename GenEigsSolver<Scalar, SelectionRule, OpType>::ComplexVector GenEigsSolver<Scalar, SelectionRule, OpType>::eigenvalues()
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
template < typename Scalar,
           int SelectionRule,
           typename OpType >
inline typename GenEigsSolver<Scalar, SelectionRule, OpType>::ComplexMatrix GenEigsSolver<Scalar, SelectionRule, OpType>::eigenvectors()
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
