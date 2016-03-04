#include <armadillo>
#include <iostream>

#define CATCH_CONFIG_MAIN
#include "../../test/catch.hpp"

typedef arma::uword uword;

void run_test(arma::sp_mat& mat, uword k, const char* form)
{
    arma::cx_vec eigval;
    arma::cx_mat eigvec;

    bool status = arma::eigs_gen(eigval, eigvec, mat, k, form);

    REQUIRE( status == true );

    arma::mat dmat(mat);
    arma::cx_mat err = dmat * eigvec - eigvec * arma::diagmat(eigval);

    INFO( "||AU - UD||_inf = " << arma::abs(err).max() );
    REQUIRE( arma::abs(err).max() == Approx(0.0) );
}

void run_test_sets(arma::sp_mat& mat, uword k)
{
    SECTION( "Largest Magnitude" )
    {
        run_test(mat, k, "lm");
    }
    SECTION( "Smallest Magnitude" )
    {
        run_test(mat, k, "sm");
    }
    SECTION( "Largest Real" )
    {
        run_test(mat, k, "lr");
    }
    SECTION( "Smallest Real" )
    {
        run_test(mat, k, "sr");
    }
    SECTION( "Largest Imaginary" )
    {
        run_test(mat, k, "li");
    }
    SECTION( "Smallest Imaginary" )
    {
        run_test(mat, k, "si");
    }
}

TEST_CASE("Eigensolver of sparse general real matrix [10x10]", "[eigs_gen]")
{
  arma::arma_rng::set_seed(123);

  arma::sp_mat A = arma::sprandu(10, 10, 0.3);
  uword k = 3;

  run_test_sets(A, k);
}

TEST_CASE("Eigensolver of sparse general real matrix [100x100]", "[eigs_gen]")
{
    arma::arma_rng::set_seed(123);

    arma::sp_mat A = arma::sprandu(100, 100, 0.3);
    uword k = 10;

    run_test_sets(A, k);
}

TEST_CASE("Eigensolver of sparse general real matrix [1000x1000]", "[eigs_gen]")
{
    arma::arma_rng::set_seed(123);

    arma::sp_mat A = arma::sprandu(1000, 1000, 0.3);
    uword k = 20;

    run_test_sets(A, k);
}
