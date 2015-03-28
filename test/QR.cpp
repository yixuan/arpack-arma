// Test ../include/UpperHessenbergQR.h
#include <UpperHessenbergQR.h>

using arma::mat;

void QR_UpperHessenberg()
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
    decomp.applyQY(Q);

    // Test orthogonality
    mat QtQ = Q.t() * Q;
    QtQ.diag() -= 1.0;
    std::cout << "||Q'Q - I||_inf = " << arma::abs(QtQ).max() << std::endl;

    mat QQt = Q * Q.t();
    QQt.diag() -= 1.0;
    std::cout << "||QQ' - I||_inf = " << arma::abs(QQt).max() << std::endl;

    // Calculate R = Q'H
    mat R = H;
    decomp.applyQtY(R);
    mat Rlower = arma::trimatl(R);
    Rlower.diag() *= 0.0;
    std::cout << "whether R is lower triangular, error = "
              << arma::abs(Rlower).max() << std::endl;

    // Compare H and QR
    std::cout << "||H - QR||_inf = " << arma::abs(H - Q * R).max() << std::endl;

    // Testing "apply" functions
    mat Y(n, n, arma::fill::randn);

    mat QY = Y;
    decomp.applyQY(QY);
    std::cout << "max error of QY = " << arma::abs(QY - Q * Y).max() << std::endl;

    mat YQ = Y;
    decomp.applyYQ(YQ);
    std::cout << "max error of YQ = " << arma::abs(YQ - Y * Q).max() << std::endl;

    mat QtY = Y;
    decomp.applyQtY(QtY);
    std::cout << "max error of Q'Y = " << arma::abs(QtY - Q.t() * Y).max() << std::endl;

    mat YQt = Y;
    decomp.applyYQt(YQt);
    std::cout << "max error of YQ' = " << arma::abs(YQt - Y * Q.t()).max() << std::endl;
}

void QR_Tridiagonal()
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
    decomp.applyQY(Q);

    // Test orthogonality
    mat QtQ = Q.t() * Q;
    QtQ.diag() -= 1.0;
    std::cout << "||Q'Q - I||_inf = " << arma::abs(QtQ).max() << std::endl;

    mat QQt = Q * Q.t();
    QQt.diag() -= 1.0;
    std::cout << "||QQ' - I||_inf = " << arma::abs(QQt).max() << std::endl;

    // Calculate R = Q'H
    mat R = H;
    decomp.applyQtY(R);
    mat Rlower = arma::trimatl(R);
    Rlower.diag() *= 0.0;
    std::cout << "whether R is lower triangular, error = "
              << arma::abs(Rlower).max() << std::endl;

    // Compare H and QR
    std::cout << "||H - QR||_inf = " << arma::abs(H - Q * R).max() << std::endl;

    // Testing "apply" functions
    mat Y(n, n, arma::fill::randn);

    mat QY = Y;
    decomp.applyQY(QY);
    std::cout << "max error of QY = " << arma::abs(QY - Q * Y).max() << std::endl;

    mat YQ = Y;
    decomp.applyYQ(YQ);
    std::cout << "max error of YQ = " << arma::abs(YQ - Y * Q).max() << std::endl;

    mat QtY = Y;
    decomp.applyQtY(QtY);
    std::cout << "max error of Q'Y = " << arma::abs(QtY - Q.t() * Y).max() << std::endl;

    mat YQt = Y;
    decomp.applyYQt(YQt);
    std::cout << "max error of YQ' = " << arma::abs(YQt - Y * Q.t()).max() << std::endl;
}

int main()
{
    std::cout << "========== Test of upper Hessenberg matrix ==========\n";
    QR_UpperHessenberg();

    std::cout << "\n========== Test of Tridiagonal matrix ==========\n";
    QR_Tridiagonal();

}
