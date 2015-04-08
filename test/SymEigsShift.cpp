#include <armadillo>
#include <iostream>

#include <SymEigsSolver.h>
#include <MatOpDenseSym.h>

typedef arma::mat Matrix;
typedef arma::vec Vector;

template <int SelectionRule>
void test(const Matrix &A, int k, int m, double sigma)
{
    Matrix mat;
    if(SelectionRule == BOTH_ENDS)
    {
        mat = A.t() + A;
    } else {
        mat = A.t() * A;
    }

    Vector all_eval = arma::eig_sym(mat);
    all_eval.t().print("all eigenvalues =");

    MatOpDenseSym<double> op(mat);
    SymEigsShiftSolver<double, SelectionRule> eigs(&op, k, m, sigma);
    eigs.init();
    int nconv = 0;
    try {
        nconv = eigs.compute();
    } catch(std::exception &e) {
        std::cout << "ERROR: " << e.what() << std::endl;
    }

    int niter, nops;
    eigs.info(niter, nops);

    if(nconv > 0)
    {
        Vector evals = eigs.eigenvalues();
        Matrix evecs = eigs.eigenvectors();
        evals.print("computed eigenvalues D =");
        //evecs.print("computed eigenvectors U =");
        Matrix err = mat * evecs - evecs * arma::diagmat(evals);
        std::cout << "||AU - UD||_inf = " << arma::abs(err).max() << std::endl;
    }
    std::cout << "nconv = " << nconv << std::endl;
    std::cout << "niter = " << niter << std::endl;
    std::cout << "nops = " << nops << std::endl;
}

int main()
{
    arma::arma_rng::set_seed(123);
    Matrix A = arma::randu(10, 10);

    int k = 3;
    int m = 6;
    double sigma = 1.0;

    std::cout << "===== Largest Magnitude =====\n";
    test<LARGEST_MAGN>(A, k, m, sigma);

    std::cout << "\n===== Largest Value =====\n";
    test<LARGEST_ALGE>(A, k, m, sigma);

    std::cout << "\n===== Smallest Magnitude =====\n";
    test<SMALLEST_MAGN>(A, k, m, sigma);

    std::cout << "\n===== Smallest Value =====\n";
    test<SMALLEST_ALGE>(A, k, m, sigma);

    std::cout << "\n===== Both Ends =====\n";
    test<BOTH_ENDS>(A, k, m, sigma);

    return 0;
}
