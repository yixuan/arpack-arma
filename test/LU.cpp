// Test ../include/LinAlg/GeneralLU.h
#include <LinAlg/GeneralLU.h>

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

    GeneralLU<Scalar> solver;
    arma::Col<Scalar> x0;
    arma::Col<Scalar> x(n);
    const Scalar prec = std::pow(std::numeric_limits<Scalar>::epsilon(), Scalar(2.0) / 3);

    x0 = arma::solve(A, b);

    solver.compute(A);
    solver.solve(b, x);

    INFO( "max|x - x0| = " << arma::abs(x - x0).max() );
    REQUIRE( arma::abs(x - x0).max() == Approx(0.0).epsilon(prec) );
}

TEST_CASE("LU on 'double' type", "[LU]")
{
    arma::arma_rng::set_seed(123);
    int n = 100;
    mat A(n, n, arma::fill::randn);
    vec b(n, arma::fill::randn);
    run_test<double>(A, b);
}

TEST_CASE("LU on 'float' type", "[LU]")
{
    arma::arma_rng::set_seed(123);
    int n = 100;
    fmat A(n, n, arma::fill::randn);
    fvec b(n, arma::fill::randn);
    run_test<float>(A, b);
}
