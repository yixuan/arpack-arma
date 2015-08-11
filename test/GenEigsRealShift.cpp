#include <armadillo>
#include <iostream>

#include <GenEigsSolver.h>

#define CATCH_CONFIG_MAIN
#include "catch.hpp"

typedef arma::mat Matrix;
typedef arma::vec Vector;
typedef arma::cx_mat ComplexMatrix;
typedef arma::cx_vec ComplexVector;

template <int SelectionRule>
void run_test(Matrix &mat, int k, int m, double sigma)
{
    // ComplexVector all_eval = arma::eig_gen(mat);
    // all_eval.t().print("all eigenvalues =");

    DenseGenRealShiftSolve<double> op(mat);
    GenEigsRealShiftSolver<double, SelectionRule, DenseGenRealShiftSolve<double>> eigs(&op, k, m, sigma);
    eigs.init();
    int nconv = eigs.compute();
    int niter = eigs.num_iterations();
    int nops = eigs.num_operations();

    REQUIRE( nconv > 0 );

    ComplexVector evals = eigs.eigenvalues();
    ComplexMatrix evecs = eigs.eigenvectors();

    // evals.print("computed eigenvalues D =");
    // evecs.print("computed eigenvectors U =");
    ComplexMatrix err = mat * evecs - evecs * arma::diagmat(evals);

    INFO( "nconv = " << nconv );
    INFO( "niter = " << niter );
    INFO( "nops = " << nops );
    INFO( "||AU - UD||_inf = " << arma::abs(err).max() );
    REQUIRE( arma::abs(err).max() == Approx(0.0) );
}


void run_test_sets(Matrix &A, int k, int m, double sigma)
{
    SECTION( "Largest Magnitude" )
    {
        run_test<LARGEST_MAGN>(A, k, m, sigma);
    }
    SECTION( "Largest Real Part" )
    {
        run_test<LARGEST_REAL>(A, k, m, sigma);
    }
    SECTION( "Largest Imaginary Part" )
    {
        run_test<LARGEST_IMAG>(A, k, m, sigma);
    }
    SECTION( "Smallest Magnitude" )
    {
        run_test<SMALLEST_MAGN>(A, k, m, sigma);
    }
    SECTION( "Smallest Real Part" )
    {
        run_test<SMALLEST_REAL>(A, k, m, sigma);
    }
    SECTION( "Smallest Imaginary Part" )
    {
        run_test<SMALLEST_IMAG>(A, k, m, sigma);
    }
}

TEST_CASE("Eigensolver of general real matrix [10x10]", "[eigs_gen]")
{
    arma::arma_rng::set_seed(123);

    Matrix A = arma::randu(10, 10);
    int k = 3;
    int m = 6;
    double sigma = 0.0;

    run_test_sets(A, k, m, sigma);
}

TEST_CASE("Eigensolver of general real matrix [100x100]", "[eigs_gen]")
{
    arma::arma_rng::set_seed(123);

    Matrix A = arma::randu(100, 100);
    int k = 10;
    int m = 20;
    double sigma = 0.0;

    run_test_sets(A, k, m, sigma);
}

TEST_CASE("Eigensolver of general real matrix [1000x1000]", "[eigs_gen]")
{
    arma::arma_rng::set_seed(123);

    Matrix A = arma::randu(1000, 1000);
    int k = 20;
    int m = 50;
    double sigma = 0.0;

    run_test_sets(A, k, m, sigma);
}
