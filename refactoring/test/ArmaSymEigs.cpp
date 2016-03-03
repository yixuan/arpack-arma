#include <armadillo>
#include <iostream>

#define CATCH_CONFIG_MAIN
#include "../../test/catch.hpp"

typedef arma::uword uword;

void run_test(arma::sp_mat& mat, uword k, const char* form)
{
    arma::vec eigval;
    arma::mat eigvec;

    bool status = arma::eigs_sym(eigval, eigvec, mat, k, form);

    REQUIRE( status == true );

    arma::mat err = mat * eigvec - eigvec * arma::diagmat(eigval);

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
    SECTION( "Largest Algebraic" )
    {
        run_test(mat, k, "la");
    }
    SECTION( "Smallest Algebraic" )
    {
        run_test(mat, k, "sa");
    }
}

TEST_CASE("Eigensolver of symmetric real matrix [10x10]", "[eigs_sym]")
{
  arma::arma_rng::set_seed(123);

  arma::sp_mat A = arma::sprandu(10, 10, 0.3);
  arma::sp_mat mat = A + A.t();
  uword k = 3;

  run_test_sets(mat, k);
}

TEST_CASE("Eigensolver of symmetric real matrix [100x100]", "[eigs_sym]")
{
    arma::arma_rng::set_seed(123);

    arma::sp_mat A = arma::sprandu(100, 100, 0.3);
    arma::sp_mat mat = A + A.t();
    uword k = 10;

    run_test_sets(mat, k);
}

TEST_CASE("Eigensolver of sparse symmetric real matrix [1000x1000]", "[eigs_sym]")
{
    arma::arma_rng::set_seed(123);

    arma::sp_mat A = arma::sprandu(1000, 1000, 0.3);
    arma::sp_mat mat = A + A.t();
    uword k = 20;

    run_test_sets(mat, k);
}
