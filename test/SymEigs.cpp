#include <armadillo>
#include <iostream>

#include <SymEigsSolver.h>
#include <MatOpDenseSym.h>

typedef arma::mat Matrix;
typedef arma::vec Vector;

template <int SelectionRule>
void test(const Matrix &A, int k, int m)
{
    Matrix mat;
    if(SelectionRule == BOTH_ENDS)
    {
        mat = A.t() + A;
    } else {
        mat = A.t() * A;
    }

    Vector all_eval = eig_sym(mat);
    all_eval.t().print("all eigenvalues =");

    MatOpDenseSym<double> op(mat);
    SymEigsSolver<double, SelectionRule> eigs(&op, k, m);
    eigs.init();
    int niter = eigs.compute();
    int nops;
    eigs.info(nops);

    Vector evals = eigs.eigenvalues();
    Matrix evecs = eigs.eigenvectors();

    evals.print("computed eigenvalues D =");
    //evecs.print("computed eigenvectors U =");
    Matrix err = mat * evecs - evecs * arma::diagmat(evals);
    std::cout << "||AU - UD||_inf = " << arma::abs(err).max() << std::endl;
    std::cout << "niter = " << niter << std::endl;
    std::cout << "nops = " << nops << std::endl;
}

int main()
{
    arma::arma_rng::set_seed(123);
    Matrix A = arma::randu(10, 10);

    int k = 3;
    int m = 6;

    std::cout << "===== Largest Magnitude =====\n";
    test<LARGEST_MAGN>(A, k, m);

    std::cout << "\n===== Largest Value =====\n";
    test<LARGEST_ALGE>(A, k, m);

    std::cout << "\n===== Smallest Magnitude =====\n";
    test<SMALLEST_MAGN>(A, k, m);

    std::cout << "\n===== Smallest Value =====\n";
    test<SMALLEST_ALGE>(A, k, m);

    std::cout << "\n===== Both Ends =====\n";
    test<BOTH_ENDS>(A, k, m);

    return 0;
}
