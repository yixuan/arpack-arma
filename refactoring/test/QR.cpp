#include <armadillo>

#define CATCH_CONFIG_MAIN
#include "../../test/catch.hpp"

using arma::mat;
using arma::vec;
using arma::UpperHessenbergQR;
using arma::TridiagQR;
using arma::DoubleShiftQR;

template <typename Solver>
void run_test(mat &H)
{
    Solver decomp(H);
    int n = H.n_rows;

    // Obtain Q matrix
    mat I(n, n, arma::fill::eye);
    mat Q = I;
    decomp.apply_YQ(Q);

    // Test orthogonality
    mat QtQ = Q.t() * Q;
    INFO( "||Q'Q - I||_inf = " << arma::abs(QtQ - I).max() );
    REQUIRE( arma::abs(QtQ - I).max() == Approx(0.0) );

    mat QQt = Q * Q.t();
    INFO( "||QQ' - I||_inf = " << arma::abs(QQt - I).max() );
    REQUIRE( arma::abs(QQt - I).max() == Approx(0.0) );

    // Test RQ
    mat rq = Q.t() * H * Q;
    INFO( "max error of RQ = " << arma::abs(decomp.matrix_RQ() - rq).max() );
    REQUIRE( arma::abs(decomp.matrix_RQ() - rq).max() == Approx(0.0) );

    // Test "apply" functions
    mat Y(n, n, arma::fill::randn);
    mat YQ = Y;
    decomp.apply_YQ(YQ);
    INFO( "max error of YQ = " << arma::abs(YQ - Y * Q).max() );
    REQUIRE( arma::abs(YQ - Y * Q).max() == Approx(0.0) );
}


TEST_CASE("QR of upper Hessenberg matrix", "[QR]")
{
    arma::arma_rng::set_seed(123);
    int n = 100;
    mat m(n, n, arma::fill::randn);
    mat H = arma::trimatu(m);
    H.diag(-1) = m.diag(-1);

    run_test< UpperHessenbergQR<double> >(H);

}

TEST_CASE("QR of Tridiagonal matrix", "[QR]")
{
    arma::arma_rng::set_seed(123);
    int n = 100;
    mat m(n, n, arma::fill::randn);
    mat H(n, n, arma::fill::zeros);
    H.diag(-1) = m.diag(-1);
    H.diag(0) = m.diag(0);
    H.diag(1) = m.diag(-1);

    run_test< TridiagQR<double> >(H);
}


TEST_CASE("QR decomposition with double shifts", "QR")
{
    arma::arma_rng::set_seed(123);
    int n = 100;
    mat m(n, n, arma::fill::randn);
    mat H = arma::trimatu(m);
    H.diag(-1) = m.diag(-1);
    H(1, 0) = H(3, 2) = H(6, 5) = 0;

    double s = 2, t = 3;

    mat M = H * H - s * H;
    M.diag() += t;

    mat Q0, R0;
    arma::qr(Q0, R0, M);

    DoubleShiftQR<double> decomp(H, s, t);
    mat Q(n, n, arma::fill::eye);
    decomp.apply_YQ(Q);

    // Equal up to signs
    INFO( "max error of Q = " << arma::abs(arma::abs(Q) - arma::abs(Q0)).max() );
    REQUIRE( arma::abs(arma::abs(Q) - arma::abs(Q0)).max() == Approx(0.0) );

    // Test Q'HQ
    INFO( "max error of Q'HQ = " << arma::abs(decomp.matrix_QtHQ() - Q.t() * H * Q).max() );
    REQUIRE( arma::abs(decomp.matrix_QtHQ() - Q.t() * H * Q).max() == Approx(0.0) );

    // Test apply functions
    vec y(n, arma::fill::randu);
    mat Y(n / 2, n, arma::fill::randu);

    vec Qty = y;
    decomp.apply_QtY(Qty);
    INFO( "max error of Q'y = " << arma::abs(Qty - Q.t() * y).max() );
    REQUIRE( arma::abs(Qty - Q.t() * y).max() == Approx(0.0) );

    mat YQ = Y;
    decomp.apply_YQ(YQ);
    INFO( "max error of YQ = " << arma::abs(YQ - Y * Q).max() );
    REQUIRE( arma::abs(YQ - Y * Q).max() == Approx(0.0) );
}
