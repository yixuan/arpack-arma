// Test ../include/LinAlg/UpperHessenbergQR.h
#include <LinAlg/UpperHessenbergQR.h>

#define CATCH_CONFIG_MAIN
#include "catch.hpp"

using arma::mat;
using arma::vec;

TEST_CASE("QR of upper Hessenberg matrix", "[QR]")
{
    arma::arma_rng::set_seed(123);
    int n = 100;
    mat m(n, n, arma::fill::randn);
    mat H = arma::trimatu(m);
    H.diag(-1) = m.diag(-1);

    UpperHessenbergQR<double> decomp(H);

    // Obtain Q matrix
    mat I(n, n, arma::fill::eye);
    mat Q = I;
    decomp.apply_QY(Q);

    // Test orthogonality
    mat QtQ = Q.t() * Q;
    INFO( "||Q'Q - I||_inf = " << arma::abs(QtQ - I).max() );
    REQUIRE( arma::abs(QtQ - I).max() == Approx(0.0) );

    mat QQt = Q * Q.t();
    INFO( "||QQ' - I||_inf = " << arma::abs(QQt - I).max() );
    REQUIRE( arma::abs(QQt - I).max() == Approx(0.0) );

    // Calculate R = Q'H
    mat R = decomp.matrix_R();
    mat Rlower = arma::trimatl(R);
    Rlower.diag().zeros();
    INFO( "Whether R is upper triangular, error = " << arma::abs(Rlower).max() );
    REQUIRE( arma::abs(Rlower).max() == Approx(0.0) );

    // Compare H and QR
    INFO( "||H - QR||_inf = " << arma::abs(H - Q * R).max() );
    REQUIRE( arma::abs(H - Q * R).max() == Approx(0.0) );

    // Test "apply" functions
    mat Y(n, n, arma::fill::randn);

    mat QY = Y;
    decomp.apply_QY(QY);
    INFO( "max error of QY = " << arma::abs(QY - Q * Y).max() );
    REQUIRE( arma::abs(QY - Q * Y).max() == Approx(0.0) );

    mat YQ = Y;
    decomp.apply_YQ(YQ);
    INFO( "max error of YQ = " << arma::abs(YQ - Y * Q).max() );
    REQUIRE( arma::abs(YQ - Y * Q).max() == Approx(0.0) );

    mat QtY = Y;
    decomp.apply_QtY(QtY);
    INFO( "max error of Q'Y = " << arma::abs(QtY - Q.t() * Y).max() );
    REQUIRE( arma::abs(QtY - Q.t() * Y).max() == Approx(0.0) );

    mat YQt = Y;
    decomp.apply_YQt(YQt);
    INFO( "max error of YQ' = " << arma::abs(YQt - Y * Q.t()).max() );
    REQUIRE( arma::abs(YQt - Y * Q.t()).max() == Approx(0.0) );

    // Test "apply" functions for vectors
    vec y(n, arma::fill::randn);

    vec Qy = y;
    decomp.apply_QY(Qy);
    INFO( "max error of Qy = " << arma::abs(Qy - Q * y).max() );
    REQUIRE( arma::abs(Qy - Q * y).max() == Approx(0.0) );

    vec Qty = y;
    decomp.apply_QtY(Qty);
    INFO( "max error of Q'y = " << arma::abs(Qty - Q.t() * y).max() );
    REQUIRE( arma::abs(Qty - Q.t() * y).max() == Approx(0.0) );
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

    TridiagQR<double> decomp(H);

    // Obtain Q matrix
    mat I(n, n, arma::fill::eye);
    mat Q = I;
    decomp.apply_QY(Q);

    // Test orthogonality
    mat QtQ = Q.t() * Q;
    INFO( "||Q'Q - I||_inf = " << arma::abs(QtQ - I).max() );
    REQUIRE( arma::abs(QtQ - I).max() == Approx(0.0) );

    mat QQt = Q * Q.t();
    INFO( "||QQ' - I||_inf = " << arma::abs(QQt - I).max() );
    REQUIRE( arma::abs(QQt - I).max() == Approx(0.0) );

    // Calculate R = Q'H
    mat R = decomp.matrix_R();
    mat Rlower = arma::trimatl(R);
    Rlower.diag().zeros();
    INFO( "Whether R is upper triangular, error = " << arma::abs(Rlower).max() );
    REQUIRE( arma::abs(Rlower).max() == Approx(0.0) );

    // Compare H and QR
    INFO( "||H - QR||_inf = " << arma::abs(H - Q * R).max() );
    REQUIRE( arma::abs(H - Q * R).max() == Approx(0.0) );

    // Test RQ
    mat rq = R;
    decomp.apply_YQ(rq);
    INFO( "max error of RQ = " << arma::abs(decomp.matrix_RQ() - rq).max() );
    REQUIRE( arma::abs(decomp.matrix_RQ() - rq).max() == Approx(0.0) );

    // Test "apply" functions
    mat Y(n, n, arma::fill::randn);

    mat QY = Y;
    decomp.apply_QY(QY);
    INFO( "max error of QY = " << arma::abs(QY - Q * Y).max() );
    REQUIRE( arma::abs(QY - Q * Y).max() == Approx(0.0) );

    mat YQ = Y;
    decomp.apply_YQ(YQ);
    INFO( "max error of YQ = " << arma::abs(YQ - Y * Q).max() );
    REQUIRE( arma::abs(YQ - Y * Q).max() == Approx(0.0) );

    mat QtY = Y;
    decomp.apply_QtY(QtY);
    INFO( "max error of Q'Y = " << arma::abs(QtY - Q.t() * Y).max() );
    REQUIRE( arma::abs(QtY - Q.t() * Y).max() == Approx(0.0) );

    mat YQt = Y;
    decomp.apply_YQt(YQt);
    INFO( "max error of YQ' = " << arma::abs(YQt - Y * Q.t()).max() );
    REQUIRE( arma::abs(YQt - Y * Q.t()).max() == Approx(0.0) );

    // Test "apply" functions for vectors
    vec y(n, arma::fill::randn);

    vec Qy = y;
    decomp.apply_QY(Qy);
    INFO( "max error of Qy = " << arma::abs(Qy - Q * y).max() );
    REQUIRE( arma::abs(Qy - Q * y).max() == Approx(0.0) );

    vec Qty = y;
    decomp.apply_QtY(Qty);
    INFO( "max error of Q'y = " << arma::abs(Qty - Q.t() * y).max() );
    REQUIRE( arma::abs(Qty - Q.t() * y).max() == Approx(0.0) );
}
