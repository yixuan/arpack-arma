// Test ../include/LinAlg/UpperHessenbergEigen.h
#include <LinAlg/UpperHessenbergEigen.h>

#define CATCH_CONFIG_MAIN
#include "catch.hpp"

using arma::mat;
using arma::vec;
using arma::cx_vec;
using arma::cx_mat;

TEST_CASE("Eigen decomposition of upper Hessenberg matrix", "[Eigen]")
{
    arma::arma_rng::set_seed(123);
    int n = 100;
    mat m(n, n, arma::fill::randn);
    mat H = arma::trimatu(m);
    H.diag(-1) = m.diag(-1);

    UpperHessenbergEigen<double> decomp(H);
    cx_vec evals = decomp.eigenvalues();
    cx_mat evecs = decomp.eigenvectors();

    cx_mat err = H * evecs - evecs * arma::diagmat(evals);

    INFO( "||HU - UD||_inf = " << arma::abs(err).max() );
    REQUIRE( arma::abs(err).max() == Approx(0.0) );

    clock_t t1, t2;
    t1 = clock();
    for(int i = 0; i < 100; i++)
    {
        UpperHessenbergEigen<double> decomp(H);
        cx_vec evals = decomp.eigenvalues();
        cx_mat evecs = decomp.eigenvectors();
    }
    t2 = clock();
    std::cout << "elapsed time for UpperHessenbergEigen: "
              << double(t2 - t1) / CLOCKS_PER_SEC << " secs\n";

    t1 = clock();
    for(int i = 0; i < 100; i++)
    {
        cx_vec evals(n);
        cx_mat evecs(n, n);
        arma::eig_gen(evals, evecs, H);
    }
    t2 = clock();
    std::cout << "elapsed time for arma::eig_gen: "
              << double(t2 - t1) / CLOCKS_PER_SEC << " secs\n";
}
