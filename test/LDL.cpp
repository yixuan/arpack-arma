// Test ../include/LinAlg/SymmetricLDL.h
#include <LinAlg/SymmetricLDL.h>

#define CATCH_CONFIG_MAIN
#include "catch.hpp"

using arma::mat;
using arma::vec;
using arma::fmat;
using arma::fvec;

template <typename Scalar>
void run_test(arma::Mat<Scalar> &A, arma::Col<Scalar> &b)
{
    const int n = A.n_rows;

    SymmetricLDL<Scalar> solver;
    arma::Col<Scalar> x0;
    arma::Col<Scalar> x(n);
    const Scalar prec = std::pow(std::numeric_limits<Scalar>::epsilon(), Scalar(2.0 / 3));

    SECTION( "Using Lower Triangular Part" )
    {
        x0 = arma::solve(arma::symmatl(A), b);

        solver.compute(A, 'L');
        solver.solve(b, x);

        INFO( "max|x - x0| = " << arma::abs(x - x0).max() );
        REQUIRE( arma::abs(x - x0).max() == Approx(0.0).epsilon(prec) );
    }
    SECTION( "Using Upper Triangular Part" )
    {
        x0 = arma::solve(arma::symmatu(A), b);

        solver.compute(A, 'U');
        solver.solve(b, x);

        INFO( "max|x - x0| = " << arma::abs(x - x0).max() );
        REQUIRE( arma::abs(x - x0).max() == Approx(0.0).epsilon(prec) );
    }
}

TEST_CASE("LDL on 'double' type", "[LDL]")
{
    arma::arma_rng::set_seed(123);
    int n = 100;
    mat A(n, n, arma::fill::randn);
    vec b(n, arma::fill::randn);
    run_test<double>(A, b);
}

TEST_CASE("LDL on 'float' type", "[LDL]")
{
    arma::arma_rng::set_seed(123);
    int n = 100;
    fmat A(n, n, arma::fill::randn);
    fvec b(n, arma::fill::randn);
    run_test<float>(A, b);
}
