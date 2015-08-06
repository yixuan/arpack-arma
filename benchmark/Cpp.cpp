#include <armadillo>
#include <iostream>
#include "timer.h"

#include <SymEigsSolver.h>
#include <GenEigsSolver.h>
#include <MatOp/DenseGenMatProd.h>
#include <MatOp/SparseGenMatProd.h>

void eigs_sym_Cpp(arma::mat &M, arma::vec &init_resid, int k, int m,
                  double &time_used, double &prec_err)
{
    double start, end;
    start = get_wall_time();

    DenseGenMatProd<double> op(M);
    SymEigsSolver< double, LARGEST_MAGN, DenseGenMatProd<double> > eigs(&op, k, m);
    eigs.init(init_resid.memptr());

    int nconv = eigs.compute();
    int niter = eigs.num_iterations();
    int nops = eigs.num_operations();

    arma::vec evals = eigs.eigenvalues();
    arma::mat evecs = eigs.eigenvectors();

    end = get_wall_time();
    time_used = (end - start) * 1000;

    arma::mat err = M * evecs - evecs * arma::diagmat(evals);
    prec_err = arma::abs(err).max();

/*
    evals.print("computed eigenvalues D =");
    evecs.head_rows(5).print("first 5 rows of computed eigenvectors U =");
    std::cout << "nconv = " << nconv << std::endl;
    std::cout << "niter = " << niter << std::endl;
    std::cout << "nops = " << nops << std::endl;

    arma::mat err = M * evecs - evecs * arma::diagmat(evals);
    std::cout << "||AU - UD||_inf = " << arma::abs(err).max() << std::endl;
*/
}



void eigs_gen_Cpp(arma::mat &M, arma::vec &init_resid, int k, int m,
                  double &time_used, double &prec_err)
{
    double start, end;
    start = get_wall_time();

    DenseGenMatProd<double> op(M);
    GenEigsSolver< double, LARGEST_MAGN, DenseGenMatProd<double> > eigs(&op, k, m);
    eigs.init(init_resid.memptr());

    int nconv = eigs.compute();
    int niter = eigs.num_iterations();
    int nops = eigs.num_operations();
    // std::cout << "nops = " << nops << std::endl;

    arma::cx_vec evals = eigs.eigenvalues();
    arma::cx_mat evecs = eigs.eigenvectors();

    end = get_wall_time();
    time_used = (end - start) * 1000;

    arma::cx_mat err = M * evecs - evecs * arma::diagmat(evals);
    prec_err = arma::abs(err).max();

/*
    evals.print("computed eigenvalues D =");
    evecs.head_rows(5).print("first 5 rows of computed eigenvectors U =");
    std::cout << "nconv = " << nconv << std::endl;
    std::cout << "niter = " << niter << std::endl;
    std::cout << "nops = " << nops << std::endl;

    arma::cx_mat err = M * evecs - evecs * arma::diagmat(evals);
    std::cout << "||AU - UD||_inf = " << arma::abs(err).max() << std::endl;
*/
}



void sparse_eigs_sym_Cpp(arma::sp_mat &M, arma::vec &init_resid, int k, int m,
                         double &time_used, double &prec_err)
{
    double start, end;
    start = get_wall_time();

    SparseGenMatProd<double> op(M);
    SymEigsSolver< double, LARGEST_MAGN, SparseGenMatProd<double> > eigs(&op, k, m);
    eigs.init(init_resid.memptr());

    int nconv = eigs.compute();
    int niter = eigs.num_iterations();
    int nops = eigs.num_operations();

    arma::vec evals = eigs.eigenvalues();
    arma::mat evecs = eigs.eigenvectors();

    end = get_wall_time();
    time_used = (end - start) * 1000;

    arma::mat err = M * evecs - evecs * arma::diagmat(evals);
    prec_err = arma::abs(err).max();
}



void sparse_eigs_gen_Cpp(arma::sp_mat &M, arma::vec &init_resid, int k, int m,
                         double &time_used, double &prec_err)
{
    double start, end;
    start = get_wall_time();

    SparseGenMatProd<double> op(M);
    GenEigsSolver< double, LARGEST_MAGN, SparseGenMatProd<double> > eigs(&op, k, m);
    eigs.init(init_resid.memptr());

    int nconv = eigs.compute();
    int niter = eigs.num_iterations();
    int nops = eigs.num_operations();

    arma::cx_vec evals = eigs.eigenvalues();
    arma::cx_mat evecs = eigs.eigenvectors();

    end = get_wall_time();
    time_used = (end - start) * 1000;

    arma::mat M_dense(M);
    arma::cx_mat err = M_dense * evecs - evecs * arma::diagmat(evals);
    prec_err = arma::abs(err).max();
}
