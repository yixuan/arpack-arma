#include <armadillo>
#include <iostream>

int eigs_sym_F77(arma::mat &M, arma::vec &init_resid, int k, int m);
int eigs_gen_F77(arma::mat &M, arma::vec &init_resid, int k, int m);
int eigs_sym_Cpp(arma::mat &M, arma::vec &init_resid, int k, int m);
int eigs_gen_Cpp(arma::mat &M, arma::vec &init_resid, int k, int m);

#define USE_PROFILER 1
#define LIB_PROFILER_IMPLEMENTATION
#define LIB_PROFILER_PRINTF printf
#include "libProfiler.h"

int main()
{
    arma::arma_rng::set_seed(123);
    arma::mat A = arma::randu(1000, 1000);
    arma::mat M = A.t() * A;

    arma::vec init_resid(M.n_cols, arma::fill::randu);
    init_resid -= 0.5;
    init_resid = M * init_resid;

    int k = 10;
    int m = 20;

    const int replicates = 10;

    PROFILER_ENABLE;
    for(int i = 0; i < replicates; i++)
        eigs_sym_F77(M, init_resid, k, m);
    LogProfiler();
    PROFILER_DISABLE;


    PROFILER_ENABLE;
    for(int i = 0; i < replicates; i++)
        eigs_sym_Cpp(M, init_resid, k, m);
    LogProfiler();
    PROFILER_DISABLE;


    PROFILER_ENABLE;
    for(int i = 0; i < replicates; i++)
        eigs_gen_F77(A, init_resid, k, m);
    LogProfiler();
    PROFILER_DISABLE;


    PROFILER_ENABLE;
    for(int i = 0; i < replicates; i++)
        eigs_gen_Cpp(A, init_resid, k, m);
    LogProfiler();
    PROFILER_DISABLE;

    return 0;
}
