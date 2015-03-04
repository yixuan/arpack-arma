#include <armadillo>
#include <SymEigsSolver.h>
#include <MatOpDense.h>
#include <iostream>

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
    all_eval.print("all eigenvalues =");

    MatOpDense<double> op(mat);
    SymEigsSolver<double, SelectionRule> eigs(&op, k, m);
    eigs.init();
    int niter = eigs.compute();

    Vector evals = eigs.eigenvalues();
    Matrix evecs = eigs.eigenvectors();

    evals.print("computed eigenvalues D =");
    evecs.print("computed eigenvectors U =");
    (mat * evecs - evecs * arma::diagmat(evals)).print("AU - UD =");

    std::cout << "niter = " << niter << std::endl;
}

int main()
{
    Matrix A = arma::randu(10, 10);

    int k = 3;
    int m = 6;

    test<LARGEST_MAGN>(A, k, m);
    test<LARGEST_ALGE>(A, k, m);
    test<SMALLEST_MAGN>(A, k, m);
    test<SMALLEST_ALGE>(A, k, m);
    test<BOTH_ENDS>(A, k, m);

    return 0;
}
