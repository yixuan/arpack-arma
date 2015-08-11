#include <armadillo>
#include <iostream>

#include <SymEigsSolver.h>

#define CATCH_CONFIG_MAIN
#include "catch.hpp"

typedef arma::mat Matrix;
typedef arma::vec Vector;

template <int SelectionRule>
void run_test(Matrix &mat, int k, int m, double sigma)
{
    // Vector all_eval = arma::eig_sym(mat);
    // all_eval.t().print("all eigenvalues =");

    DenseSymShiftSolve<double> op(mat);
    SymEigsShiftSolver<double, SelectionRule, DenseSymShiftSolve<double>> eigs(&op, k, m, sigma);
    eigs.init();
    int nconv = eigs.compute();
    int niter = eigs.num_iterations();
    int nops = eigs.num_operations();

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

void run_test_sets(Matrix &mat, int k, int m, double sigma)
{
    SECTION( "Largest Magnitude" )
    {
        run_test<LARGEST_MAGN>(mat, k, m, sigma);
    }
    SECTION( "Largest Value" )
    {
        run_test<LARGEST_ALGE>(mat, k, m, sigma);
    }
    SECTION( "Smallest Magnitude" )
    {
        run_test<SMALLEST_MAGN>(mat, k, m, sigma);
    }
    SECTION( "Smallest Value" )
    {
        run_test<SMALLEST_ALGE>(mat, k, m, sigma);
    }
    SECTION( "Both Ends" )
    {
        run_test<BOTH_ENDS>(mat, k, m, sigma);
    }
}

TEST_CASE("Eigensolver of symmetric real matrix [10x10]", "[eigs_sym]")
{
    arma::arma_rng::set_seed(123);

    Matrix A = arma::randu(10, 10);
    Matrix mat = A + A.t();
    int k = 3;
    int m = 6;
    double sigma = 1.0;

    run_test_sets(mat, k, m, sigma);
}

TEST_CASE("Eigensolver of symmetric real matrix [100x100]", "[eigs_sym]")
{
    arma::arma_rng::set_seed(123);

    Matrix A = arma::randu(100, 100);
    Matrix mat = A + A.t();
    int k = 10;
    int m = 30;
    double sigma = 1.0;

    run_test_sets(mat, k, m, sigma);
}

TEST_CASE("Eigensolver of symmetric real matrix [1000x1000]", "[eigs_sym]")
{
    arma::arma_rng::set_seed(123);

    Matrix A = arma::randu(1000, 1000);
    Matrix mat = A + A.t();
    int k = 20;
    int m = 50;
    double sigma = 1.0;

    run_test_sets(mat, k, m, sigma);
}
