// Test ../include/SymmetricLDL.h
#include <SymmetricLDL.h>

using arma::mat;
using arma::vec;
using arma::fmat;
using arma::fvec;

template <typename Scalar>
void run_test(arma::Mat<Scalar> &A, arma::Col<Scalar> &b)
{
    int n = A.n_rows;

    std::cout << "========== Using Lower Triangular Part ==========\n";
    arma::solve(arma::symmatl(A), b).t().print("solve(A, b) = ");

    SymmetricLDL<Scalar> solver(A, 'L');
    arma::Col<Scalar> x(n);
    solver.solve(b, x);
    x.t().print("x = ");

    std::cout << "========== Using Upper Triangular Part ==========\n";
    arma::solve(arma::symmatu(A), b).t().print("solve(A, b) = ");

    solver.compute(A, 'U');
    solver.solve(b, x);
    x.t().print("x = ");
}

int main()
{
    arma::arma_rng::set_seed(123);
    int n = 8;
    mat A(n, n, arma::fill::randn);
    vec b(n, arma::fill::randn);

    std::cout << ">>> Type = double\n\n";
    run_test<double>(A, b);

    fmat A2(n, n, arma::fill::randn);
    fvec b2(n, arma::fill::randn);

    std::cout << "\n\n>>> Type = float\n\n";
    run_test<float>(A2, b2);
}
