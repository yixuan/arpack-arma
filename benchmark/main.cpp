#include <armadillo>
#include <ctime>
#include <iostream>
#include <iomanip>
#include <string>

int eigs_sym_F77(arma::mat &M, arma::vec &init_resid, int k, int m);
int eigs_gen_F77(arma::mat &M, arma::vec &init_resid, int k, int m);
int eigs_sym_Cpp(arma::mat &M, arma::vec &init_resid, int k, int m);
int eigs_gen_Cpp(arma::mat &M, arma::vec &init_resid, int k, int m);


void print_header(std::string title)
{
    const int width = 60;
    const int col_width = width / 4;
    const char sep = ' ';

    std::cout << std::endl << std::string(width, '=') << std::endl;
    std::cout << std::string((width - title.length()) / 2, ' ') << title << std::endl;
    std::cout << std::string(width, '-') << std::endl;

    std::cout << std::left << std::setw(col_width) << std::setfill(sep) << "matrix_size";
    std::cout << std::left << std::setw(col_width) << std::setfill(sep) << "dataset";
    std::cout << std::left << std::setw(col_width) << std::setfill(sep) << "F77 (ms)";
    std::cout << std::left << std::setw(col_width) << std::setfill(sep) << "C++ (ms)";
    std::cout << std::endl;

    std::cout << std::string(width, '-') << std::endl;
}

void print_row(int dataset, int n, double time_f77, double time_cpp)
{
    const int width = 60;
    const int col_width = width / 4;
    const char sep = ' ';

    std::cout << std::left << std::setw(col_width) << std::setfill(sep) << n;
    std::cout << std::left << std::setw(col_width) << std::setfill(sep) << dataset;
    std::cout << std::left << std::setw(col_width) << std::setfill(sep) << time_f77;
    std::cout << std::left << std::setw(col_width) << std::setfill(sep) << time_cpp;
    std::cout << std::endl;
}

void print_footer()
{
    const int width = 60;
    std::cout << std::string(width, '=') << std::endl << std::endl;
}

void run_eigs_sym(int n_experiment, int n_replicate, int n, int k, int m)
{
    clock_t start, end;
    double time_f77, time_cpp;

    for(int i = 0; i < n_experiment; i++)
    {
        arma::mat A = arma::randu(n, n);
        arma::mat M = A.t() + A;

        arma::vec init_resid(M.n_cols, arma::fill::randu);
        init_resid -= 0.5;
        init_resid = M * init_resid;

        for(int j = 0; j < n_replicate; j++)
        {
            start = clock();
            eigs_sym_F77(M, init_resid, k, m);
            end = clock();
            time_f77 = (end - start) / double(CLOCKS_PER_SEC) * 1000;

            start = clock();
            eigs_sym_Cpp(M, init_resid, k, m);
            end = clock();
            time_cpp = (end - start) / double(CLOCKS_PER_SEC) * 1000;

            print_row(i + 1, n, time_f77, time_cpp);
        }
    }
}

void run_eigs_gen(int n_experiment, int n_replicate, int n, int k, int m)
{
    clock_t start, end;
    double time_f77, time_cpp;

    for(int i = 0; i < n_experiment; i++)
    {
        arma::mat A = arma::randu(n, n);

        arma::vec init_resid(A.n_cols, arma::fill::randu);
        init_resid -= 0.5;
        init_resid = A * init_resid;

        for(int j = 0; j < n_replicate; j++)
        {
            start = clock();
            eigs_gen_F77(A, init_resid, k, m);
            end = clock();
            time_f77 = (end - start) / double(CLOCKS_PER_SEC) * 1000;

            start = clock();
            eigs_gen_Cpp(A, init_resid, k, m);
            end = clock();
            time_cpp = (end - start) / double(CLOCKS_PER_SEC) * 1000;

            print_row(i + 1, n, time_f77, time_cpp);
        }
    }
}

int main()
{
    arma::arma_rng::set_seed(123);

    int n_experiment = 5;
    int n_replicate = 5;

    print_header("eigs_sym");
    run_eigs_sym(n_experiment, n_replicate, 100, 10, 20);
    run_eigs_sym(n_experiment, n_replicate, 1000, 10, 30);
    print_footer();

    print_header("eigs_gen");
    run_eigs_gen(n_experiment, n_replicate, 100, 10, 20);
    run_eigs_gen(n_experiment, n_replicate, 1000, 10, 30);
    print_footer();

    return 0;
}
