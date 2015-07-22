#ifndef LAPACK_WRAPPER_EXTRA_H
#define LAPACK_WRAPPER_EXTRA_H

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

        // Calculating eigenvalues of an upper Hessenberg matrix
        #define arma_slahqr slahqr
        #define arma_dlahqr dlahqr

        // Calculating eigenvalues of a symmetric tridiagonal matrix
        #define arma_ssteqr ssteqr
        #define arma_dsteqr dsteqr

        // Calculating eigenvectors of a Schur form matrix
        #define arma_strevc strevc
        #define arma_dtrevc dtrevc

    #else

        #define arma_ssytrs SSYTRS
        #define arma_dsytrs DSYTRS
        #define arma_csytrs CSYTRS
        #define arma_zsytrs ZSYTRS

        #define arma_sgetrs SGETRS
        #define arma_dgetrs DGETRS
        #define arma_cgetrs CGETRS
        #define arma_zgetrs ZGETRS

        #define arma_slahqr SLAHQR
        #define arma_dlahqr DLAHQR

        #define arma_ssteqr SSTEQR
        #define arma_dsteqr DSTEQR

        #define arma_strevc STREVC
        #define arma_dtrevc DTREVC

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

        void arma_fortran(arma_slahqr)(blas_int* wantt, blas_int* wantz, blas_int* n, blas_int* ilo, blas_int* ihi, float*  h, blas_int* ldh, float*  wr, float*  wi, blas_int* iloz, blas_int* ihiz, float*  z, blas_int* ldz, blas_int* info);
        void arma_fortran(arma_dlahqr)(blas_int* wantt, blas_int* wantz, blas_int* n, blas_int* ilo, blas_int* ihi, double* h, blas_int* ldh, double* wr, double* wi, blas_int* iloz, blas_int* ihiz, double* z, blas_int* ldz, blas_int* info);

        void arma_fortran(arma_ssteqr)(char* compz, blas_int* n, float*  d, float*  e, float*  z, blas_int* ldz, float*  work, blas_int* info);
        void arma_fortran(arma_dsteqr)(char* compz, blas_int* n, double* d, double* e, double* z, blas_int* ldz, double* work, blas_int* info);

        void arma_fortran(arma_strevc)(char* side, char* howmny, blas_int* select, blas_int* n, float*  t, blas_int* ldt, float*  vl, blas_int* ldvl, float*  vr, blas_int* ldvr, blas_int* mm, blas_int* m, float*  work, blas_int* info);
        void arma_fortran(arma_dtrevc)(char* side, char* howmny, blas_int* select, blas_int* n, double* t, blas_int* ldt, double* vl, blas_int* ldvl, double* vr, blas_int* ldvr, blas_int* mm, blas_int* m, double* work, blas_int* info);
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

        template<typename eT>
        inline
        void
        lahqr(blas_int* wantt, blas_int* wantz, blas_int* n, blas_int* ilo, blas_int* ihi, eT* h, blas_int* ldh, eT* wr, eT* wi, blas_int* iloz, blas_int* ihiz, eT* z, blas_int* ldz, blas_int* info)
        {
            arma_type_check(( is_supported_blas_type<eT>::value == false ));
            if(is_float<eT>::value == true)
            {
                typedef float T;
                arma_fortran(arma_slahqr)(wantt, wantz, n, ilo, ihi, (T*)h, ldh, (T*)wr, (T*)wi, iloz, ihiz, (T*)z, ldz, info);
            }
            else
            if(is_double<eT>::value == true)
            {
                typedef double T;
                arma_fortran(arma_dlahqr)(wantt, wantz, n, ilo, ihi, (T*)h, ldh, (T*)wr, (T*)wi, iloz, ihiz, (T*)z, ldz, info);
            }
        }

        template<typename eT>
        inline
        void
        steqr(char* compz, blas_int* n, eT* d, eT* e, eT* z, blas_int* ldz, eT* work, blas_int* info)
        {
            arma_type_check(( is_supported_blas_type<eT>::value == false ));
            if(is_float<eT>::value == true)
            {
                typedef float T;
                arma_fortran(arma_ssteqr)(compz, n, (T*)d, (T*)e, (T*)z, ldz, (T*)work, info);
            }
            else
            if(is_double<eT>::value == true)
            {
                typedef double T;
                arma_fortran(arma_dsteqr)(compz, n, (T*)d, (T*)e, (T*)z, ldz, (T*)work, info);
            }
        }

        template<typename eT>
        inline
        void
        trevc(char* side, char* howmny, blas_int* select, blas_int* n, eT* t, blas_int* ldt, eT* vl, blas_int* ldvl, eT* vr, blas_int* ldvr, blas_int* mm, blas_int* m, eT* work, blas_int* info)
        {
            arma_type_check(( is_supported_blas_type<eT>::value == false ));
            if(is_float<eT>::value == true)
            {
                typedef float T;
                arma_fortran(arma_strevc)(side, howmny, select, n, (T*)t, ldt, (T*)vl, ldvl, (T*)vr, ldvr, mm, m, (T*)work, info);
            }
            else
            if(is_double<eT>::value == true)
            {
                typedef double T;
                arma_fortran(arma_dtrevc)(side, howmny, select, n, (T*)t, ldt, (T*)vl, ldvl, (T*)vr, ldvr, mm, m, (T*)work, info);
            }
        }
    }



    #endif  // ARMA_USE_LAPACK
}



#endif  // LAPACK_WRAPPER_EXTRA_H
