#include <armadillo>
#include <SymEigsSolver.h>
#include <MatOpDense.h>
#include <iostream>

int run_Cpp(arma::mat &M, int k, int m)
{
    MatOpDense<double> op(M);
    SymEigsSolver<double, LARGEST_MAGN> eigs(&op, k, m);
    eigs.init();
    int niter = eigs.compute();

    arma::vec evals = eigs.eigenvalues();
    arma::mat evecs = eigs.eigenvectors();

    evals.print("computed eigenvalues D =");
    evecs.head_rows(5).print("first 5 rows of computed eigenvectors U =");
    std::cout << "niter = " << niter << std::endl;

    // arma::mat err = M * evecs - evecs * arma::diagmat(evals);
    // std::cout << "||AU - UD||_inf = " << arma::abs(err).max() << std::endl;

    return 0;
}
