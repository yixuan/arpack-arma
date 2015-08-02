#include <armadillo>
#include <iostream>

#include <SymEigsSolver.h>
#include <GenEigsSolver.h>

int eigs_sym_Cpp(arma::mat &M, arma::vec &init_resid, int k, int m)
{
    DenseGenMatProd<double> op(M);
    SymEigsSolver< double, LARGEST_MAGN, DenseGenMatProd<double> > eigs(&op, k, m);
    eigs.init(init_resid.memptr());

    int nconv = eigs.compute();
    int niter = eigs.num_iterations();
    int nops = eigs.num_operations();

    arma::vec evals = eigs.eigenvalues();
    arma::mat evecs = eigs.eigenvectors();

/*
    evals.print("computed eigenvalues D =");
    evecs.head_rows(5).print("first 5 rows of computed eigenvectors U =");
    std::cout << "nconv = " << nconv << std::endl;
    std::cout << "niter = " << niter << std::endl;
    std::cout << "nops = " << nops << std::endl;

    arma::mat err = M * evecs - evecs * arma::diagmat(evals);
    std::cout << "||AU - UD||_inf = " << arma::abs(err).max() << std::endl;
*/

    return 0;
}



int eigs_gen_Cpp(arma::mat &M, arma::vec &init_resid, int k, int m)
{
    DenseGenMatProd<double> op(M);
    GenEigsSolver< double, LARGEST_MAGN, DenseGenMatProd<double> > eigs(&op, k, m);
    eigs.init(init_resid.memptr());

    int nconv = eigs.compute();
    int niter = eigs.num_iterations();
    int nops = eigs.num_operations();

    arma::cx_vec evals = eigs.eigenvalues();
    arma::cx_mat evecs = eigs.eigenvectors();

/*
    evals.print("computed eigenvalues D =");
    evecs.head_rows(5).print("first 5 rows of computed eigenvectors U =");
    std::cout << "nconv = " << nconv << std::endl;
    std::cout << "niter = " << niter << std::endl;
    std::cout << "nops = " << nops << std::endl;

    arma::cx_mat err = M * evecs - evecs * arma::diagmat(evals);
    std::cout << "||AU - UD||_inf = " << arma::abs(err).max() << std::endl;
*/

    return 0;
}
