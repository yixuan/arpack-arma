#ifndef LAPACKWRAPPEREXTRA_H
#define LAPACKWRAPPEREXTRA_H

namespace arma
{
    #ifdef ARMA_USE_LAPACK

    #if !defined(ARMA_BLAS_CAPITALS)

        // Solving linear equations using LDL decomposition
        #define arma_ssytrs ssytrs
        #define arma_dsytrs dsytrs
        #define arma_csytrs csytrs
        #define arma_zsytrs zsytrs

        // Solving linear equations using LU decomposition
        #define arma_sgetrs sgetrs
        #define arma_dgetrs dgetrs
        #define arma_cgetrs cgetrs
        #define arma_zgetrs zgetrs

    #else

        #define arma_ssytrs SSYTRS
        #define arma_dsytrs DSYTRS
        #define arma_csytrs CSYTRS
        #define arma_zsytrs ZSYTRS

        #define arma_sgetrs SGETRS
        #define arma_dgetrs DGETRS
        #define arma_cgetrs CGETRS
        #define arma_zgetrs ZGETRS

    #endif



    extern "C"
    {
        void arma_fortran(arma_ssytrs)(char* uplo, blas_int* n, blas_int* nrhs, float*  a, blas_int* lda, blas_int* ipiv, float*  b, blas_int* ldb, blas_int* info);
        void arma_fortran(arma_dsytrs)(char* uplo, blas_int* n, blas_int* nrhs, double* a, blas_int* lda, blas_int* ipiv, double* b, blas_int* ldb, blas_int* info);
        void arma_fortran(arma_csytrs)(char* uplo, blas_int* n, blas_int* nrhs, void*   a, blas_int* lda, blas_int* ipiv, void*   b, blas_int* ldb, blas_int* info);
        void arma_fortran(arma_zsytrs)(char* uplo, blas_int* n, blas_int* nrhs, void*   a, blas_int* lda, blas_int* ipiv, void*   b, blas_int* ldb, blas_int* info);

        void arma_fortran(arma_sgetrs)(char* trans, blas_int* n, blas_int* nrhs, float*  a, blas_int* lda, blas_int* ipiv, float*  b, blas_int* ldb, blas_int* info);
        void arma_fortran(arma_dgetrs)(char* trans, blas_int* n, blas_int* nrhs, double* a, blas_int* lda, blas_int* ipiv, double* b, blas_int* ldb, blas_int* info);
        void arma_fortran(arma_cgetrs)(char* trans, blas_int* n, blas_int* nrhs, void*   a, blas_int* lda, blas_int* ipiv, void*   b, blas_int* ldb, blas_int* info);
        void arma_fortran(arma_zgetrs)(char* trans, blas_int* n, blas_int* nrhs, void*   a, blas_int* lda, blas_int* ipiv, void*   b, blas_int* ldb, blas_int* info);
    }



    namespace lapack
    {
        template<typename eT>
        inline
        void
        sytrs(char* uplo, blas_int* n, blas_int* nrhs, eT* a, blas_int* lda, blas_int* ipiv, eT* b, blas_int* ldb, blas_int* info)
        {
            arma_type_check(( is_supported_blas_type<eT>::value == false ));
            if(is_float<eT>::value == true)
            {
                typedef float T;
                arma_fortran(arma_ssytrs)(uplo, n, nrhs, (T*)a, lda, ipiv, (T*)b, ldb, info);
            }
            else
            if(is_double<eT>::value == true)
            {
                typedef double T;
                arma_fortran(arma_dsytrs)(uplo, n, nrhs, (T*)a, lda, ipiv, (T*)b, ldb, info);
            }
            else
            if(is_supported_complex_float<eT>::value == true)
            {
                typedef std::complex<float> T;
                arma_fortran(arma_csytrs)(uplo, n, nrhs, (T*)a, lda, ipiv, (T*)b, ldb, info);
            }
            else
            if(is_supported_complex_double<eT>::value == true)
            {
                typedef std::complex<double> T;
                arma_fortran(arma_zsytrs)(uplo, n, nrhs, (T*)a, lda, ipiv, (T*)b, ldb, info);
            }
        }

        template<typename eT>
        inline
        void
        getrs(char* trans, blas_int* n, blas_int* nrhs, eT* a, blas_int* lda, blas_int* ipiv, eT* b, blas_int* ldb, blas_int* info)
        {
            arma_type_check(( is_supported_blas_type<eT>::value == false ));
            if(is_float<eT>::value == true)
            {
                typedef float T;
                arma_fortran(arma_sgetrs)(trans, n, nrhs, (T*)a, lda, ipiv, (T*)b, ldb, info);
            }
            else
            if(is_double<eT>::value == true)
            {
                typedef double T;
                arma_fortran(arma_dgetrs)(trans, n, nrhs, (T*)a, lda, ipiv, (T*)b, ldb, info);
            }
            else
            if(is_supported_complex_float<eT>::value == true)
            {
                typedef std::complex<float> T;
                arma_fortran(arma_cgetrs)(trans, n, nrhs, (T*)a, lda, ipiv, (T*)b, ldb, info);
            }
            else
            if(is_supported_complex_double<eT>::value == true)
            {
                typedef std::complex<double> T;
                arma_fortran(arma_zgetrs)(trans, n, nrhs, (T*)a, lda, ipiv, (T*)b, ldb, info);
            }
        }
    }



    #endif  // ARMA_USE_LAPACK
}



#endif  // LAPACKWRAPPEREXTRA_H
