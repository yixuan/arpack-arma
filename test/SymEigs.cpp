#include <armadillo>
#include <iostream>

#include <SymEigsSolver.h>

#define CATCH_CONFIG_MAIN
#include "catch.hpp"

typedef arma::mat Matrix;
typedef arma::vec Vector;

template <int SelectionRule>
void run_test(const Matrix &A, int k, int m)
{
    Matrix mat;
    if(SelectionRule == BOTH_ENDS)
    {
        mat = A.t() + A;
    } else {
        mat = A.t() * A;
    }

    // Vector all_eval = arma::eig_sym(mat);
    // all_eval.t().print("all eigenvalues =");

    DenseMatProd<double> op(mat);
    SymEigsSolver<double, SelectionRule, DenseMatProd<double>> eigs(&op, k, m);
    eigs.init();
    int nconv = eigs.compute();
    int niter, nops;
    eigs.info(niter, nops);

    REQUIRE( nconv > 0 );

    Vector evals = eigs.eigenvalues();
    Matrix evecs = eigs.eigenvectors();

    // evals.print("computed eigenvalues D =");
    // evecs.print("computed eigenvectors U =");
    Matrix err = mat * evecs - evecs * arma::diagmat(evals);

    INFO( "nconv = " << nconv );
    INFO( "niter = " << niter );
    INFO( "nops = " << nops );
    INFO( "||AU - UD||_inf = " << arma::abs(err).max() );
    REQUIRE( arma::abs(err).max() == Approx(0.0) );
}

TEST_CASE("Eigensolver of symmetric real matrix", "[eigs_sym]")
{
    arma::arma_rng::set_seed(123);
    Matrix A = arma::randu(10, 10);

    int k = 3;
    int m = 6;

    SECTION( "Largest Magnitude" )
    {
        run_test<LARGEST_MAGN>(A, k, m);
    }
    SECTION( "Largest Value" )
    {
        run_test<LARGEST_ALGE>(A, k, m);
    }
    SECTION( "Smallest Magnitude" )
    {
        run_test<SMALLEST_MAGN>(A, k, m);
    }
    SECTION( "Smallest Value" )
    {
        run_test<SMALLEST_ALGE>(A, k, m);
    }
    SECTION( "Both Ends" )
    {
        run_test<BOTH_ENDS>(A, k, m);
    }
}
