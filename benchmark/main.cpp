#include <armadillo>
#include <SymEigsSolver.h>
#include <MatOpDense.h>
#include <iostream>

int run_F77(arma::mat &M, arma::vec &init_resid, int k, int m);
int run_Cpp(arma::mat &M, arma::vec &init_resid, int k, int m);

int main()
{
    arma::arma_rng::set_seed(123);
    arma::mat A = arma::randu(1000, 1000);
    arma::mat M = A.t() * A;

    arma::vec init_resid(M.n_cols, arma::fill::randu);
    init_resid -= 0.5;
    init_resid = M * init_resid;

    int k = 5;
    int m = 10;

    clock_t t1, t2;
    t1 = clock();

    run_F77(M, init_resid, k, m);

    t2 = clock();
    std::cout << "elapsed time for F77 version: "
              << double(t2 - t1) / CLOCKS_PER_SEC << " secs\n";

    t1 = clock();

    run_Cpp(M, init_resid, k, m);

    t2 = clock();
    std::cout << "elapsed time for C++ version: "
              << double(t2 - t1) / CLOCKS_PER_SEC << " secs\n";

    return 0;
}
