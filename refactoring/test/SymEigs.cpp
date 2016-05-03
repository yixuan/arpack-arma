#include <armadillo>
#include <iostream>

#define CATCH_CONFIG_MAIN
#include "../../test/catch.hpp"

typedef arma::mat Matrix;
typedef arma::vec Vector;
typedef arma::sp_mat SpMatrix;
using arma::alt_eigs::SymEigsSolver;
using arma::alt_eigs::EigsSelect;
using arma::alt_eigs::DenseGenMatProd;
using arma::alt_eigs::SparseGenMatProd;

// Traits to obtain operation type from matrix type
template <typename MatType>
struct OpTypeTrait
{
    typedef DenseGenMatProd<double> OpType;
};

template <>
struct OpTypeTrait<SpMatrix>
{
    typedef SparseGenMatProd<double> OpType;
};


template <typename MatType, int SelectionRule>
void run_test(MatType &mat, int k, int m)
{
    typename OpTypeTrait<MatType>::OpType op(mat);
    SymEigsSolver<double, SelectionRule, typename OpTypeTrait<MatType>::OpType> eigs(op, k, m);
    eigs.init();
    int nconv = eigs.compute();
    int niter = eigs.num_iterations();
    int nops = eigs.num_operations();

    REQUIRE( nconv > 0 );

    Vector evals = eigs.eigenvalues();
    Matrix evecs = eigs.eigenvectors();

    // evals.print("computed eigenvalues D =");
    // evecs.print("computed eigenvectors U =");
    Matrix err = mat * evecs - evecs * arma::diagmat(evals);

    INFO( "nconv = " << nconv );
    INFO( "niter = " << niter );
    INFO( "nops = " << nops );
    INFO( "||AU - UD||_inf = " << arma::abs(err).max() );
    REQUIRE( arma::abs(err).max() == Approx(0.0) );
}

template <typename MatType>
void run_test_sets(MatType &mat, int k, int m)
{
    SECTION( "Largest Magnitude" )
    {
        run_test<MatType, EigsSelect::LARGEST_MAGN>(mat, k, m);
    }
    SECTION( "Largest Value" )
    {
        run_test<MatType, EigsSelect::LARGEST_ALGE>(mat, k, m);
    }
    SECTION( "Smallest Magnitude" )
    {
        run_test<MatType, EigsSelect::SMALLEST_MAGN>(mat, k, m);
    }
    SECTION( "Smallest Value" )
    {
        run_test<MatType, EigsSelect::SMALLEST_ALGE>(mat, k, m);
    }
    SECTION( "Both Ends" )
    {
        run_test<MatType, EigsSelect::BOTH_ENDS>(mat, k, m);
    }
}

TEST_CASE("Eigensolver of symmetric real matrix [10x10]", "[eigs_sym]")
{
    arma::arma_rng::set_seed(123);

    Matrix A = arma::randu(10, 10);
    Matrix mat = A + A.t();
    int k = 3;
    int m = 6;

    run_test_sets(mat, k, m);
}

TEST_CASE("Eigensolver of symmetric real matrix [100x100]", "[eigs_sym]")
{
    arma::arma_rng::set_seed(123);

    Matrix A = arma::randu(100, 100);
    Matrix mat = A + A.t();
    int k = 10;
    int m = 30;

    run_test_sets(mat, k, m);
}

TEST_CASE("Eigensolver of symmetric real matrix [1000x1000]", "[eigs_sym]")
{
    arma::arma_rng::set_seed(123);

    Matrix A = arma::randu(1000, 1000);
    Matrix mat = A + A.t();
    int k = 20;
    int m = 50;

    run_test_sets(mat, k, m);
}

TEST_CASE("Eigensolver of sparse symmetric real matrix [1000x1000]", "[eigs_sym]")
{
    arma::arma_rng::set_seed(123);

    SpMatrix A = arma::sprandu(1000, 1000, 0.1);
    SpMatrix mat = A + A.t();
    int k = 10;
    int m = 30;

    run_test_sets(mat, k, m);
}
