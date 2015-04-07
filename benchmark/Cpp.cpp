#include <armadillo>
#include <SymEigsSolver.h>
#include <MatOpDense.h>
#include <iostream>

int run_Cpp(arma::mat &M, arma::vec &init_resid, int k, int m)
{
    MatOpDense<double> op(M);
    SymEigsSolver<double, LARGEST_MAGN> eigs(&op, k, m);
    eigs.init(init_resid.memptr());

    int nconv = eigs.compute();
    int niter, nops;
    eigs.info(niter, nops);

    arma::vec evals = eigs.eigenvalues();
    arma::mat evecs = eigs.eigenvectors();

    evals.print("computed eigenvalues D =");
    evecs.head_rows(5).print("first 5 rows of computed eigenvectors U =");
    std::cout << "nconv = " << nconv << std::endl;
    std::cout << "niter = " << niter << std::endl;
    std::cout << "nops = " << nops << std::endl;

    // arma::mat err = M * evecs - evecs * arma::diagmat(evals);
    // std::cout << "||AU - UD||_inf = " << arma::abs(err).max() << std::endl;

    return 0;
}
