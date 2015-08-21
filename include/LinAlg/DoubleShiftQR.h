// Copyright (C) 2015 Yixuan Qiu
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef DOUBLE_SHIFT_QR_H
#define DOUBLE_SHIFT_QR_H

#include <armadillo>
#include <vector>     // std::vector
#include <algorithm>  // std::min, std::fill
#include <cmath>      // std::abs, std::sqrt, std::pow
#include <limits>     // std::numeric_limits
#include <stdexcept>  // std::invalid_argument, std::logic_error

template <typename Scalar = double>
class DoubleShiftQR
{
private:
    typedef arma::Mat<Scalar> Matrix;
    typedef arma::Col<Scalar> Vector;

    int n;              // Dimension of the matrix
    Matrix mat_H;       // A copy of the matrix to be factorized
    Scalar shift_s;     // Shift constant
    Scalar shift_t;     // Shift constant
    Matrix ref_u;       // Householder reflectors
    const Scalar prec;  // Approximately zero
    const Scalar prec2;
    bool computed;      // Whether matrix has been factorized

    void compute_reflector(const Scalar &x1, const Scalar &x2, const Scalar &x3, int ind)
    {
        Scalar *u = ref_u.memptr() + 3 * ind;

        if(std::abs(x1) + std::abs(x2) + std::abs(x3) <= 3 * prec)
        {
            u[0] = u[1] = u[2] = 0;
            return;
        }
        // x1' = x1 - rho * ||x||
        // rho = -sign(x1)
        Scalar tmp = x2 * x2 + x3 * x3;
        Scalar x1_new = x1 - ((x1 < 0) - (x1 > 0)) * std::sqrt(x1 * x1 + tmp);
        Scalar x_norm = std::sqrt(x1_new * x1_new + tmp);
        u[0] = x1_new / x_norm;
        u[1] = x2 / x_norm;
        u[2] = x3 / x_norm;
    }

    void compute_reflector(const Scalar *x, int ind)
    {
        compute_reflector(x[0], x[1], x[2], ind);
    }

    void compute_reflectors_from_block(Matrix &X, int oi, int block_size, int start_ind)
    {
        // For the sub-block of X, we can assume all sub-diagonal elements are non-zero
        const int mat_size = X.n_rows;
        // For block size == 1, there is no need to apply reflectors
        if(block_size == 1)
        {
            compute_reflector(0, 0, 0, start_ind);
            return;
        }

        // For block size == 2, do a Givens rotation on M = X * X - s * X + t * I
        // x00 => X(oi, oi), x01 => X(oi, oi + 1)
        Scalar *x00 = X.colptr(oi) + oi, *x01 = x00 + mat_size;
        if(block_size == 2)
        {
            Scalar x = x00[0] * (x00[0] - shift_s) + x01[0] * x00[1] + shift_t;
            Scalar y = x00[1] * (x00[0] + x01[1] - shift_s);
            compute_reflector(x, y, 0, start_ind);
            apply_PX(X, oi, oi, 2, mat_size - oi, start_ind);
            apply_XP(X, 0, oi, oi + 2, 2, start_ind);
            compute_reflector(0, 0, 0, start_ind + 1);
            return;
        }

        // For block size >= 3, use the regular strategy
        // Scalar x = X(0, 0) * (X(0, 0) - shift_s) + X(0, 1) * X(1, 0) + shift_t;
        // Scalar y = X(1, 0) * (X(0, 0) + X(1, 1) - shift_s);
        // Scalar z = X(2, 1) * X(1, 0);
        Scalar x = x00[0] * (x00[0] - shift_s) + x01[0] * x00[1] + shift_t;
        Scalar y = x00[1] * (x00[0] + x01[1] - shift_s);
        Scalar z = x01[2] * x00[1];
        compute_reflector(x, y, z, start_ind);
        // Apply the first reflector
        apply_PX(X, oi, oi, 3, mat_size - oi, start_ind);
        apply_XP(X, 0, oi, oi + std::min(block_size, 4), 3, start_ind);

        // Calculate the remaining reflectors
        // If entering this loop, nrow is at least 4.
        for(int i = 1; i < block_size - 2; i++)
        {
            compute_reflector(&X(oi + i, oi + i - 1), start_ind + i);
            // Apply the reflector to X
            apply_PX(X, oi + i, oi + i - 1, 3, mat_size - oi - i + 1, start_ind + i);
            apply_XP(X, 0, oi + i, oi + std::min(block_size, i + 4), 3, start_ind + i);
        }

        // The last reflector
        compute_reflector(X(oi + block_size - 2, oi + block_size - 3), X(oi + block_size - 1, oi + block_size - 3), 0, start_ind + block_size - 2);
        // Apply the reflector to X
        apply_PX(X, oi + block_size - 2, oi + block_size - 3, 2, mat_size - oi - block_size + 3, start_ind + block_size - 2);
        apply_XP(X, 0, oi + block_size - 2, oi + block_size, 2, start_ind + block_size - 2);

        compute_reflector(0, 0, 0, start_ind + block_size - 1);
    }

    // P = I - 2 * u * u' = P'
    // PX = X - 2 * u * (u'X)
    void apply_PX(Matrix &X, int oi, int oj, int nrow, int ncol, int u_ind)
    {
        Scalar *u = ref_u.memptr() + 3 * u_ind;

        if(std::abs(u[0]) + std::abs(u[1]) + std::abs(u[2]) <= 3 * prec)
            return;

        const int stride = X.n_rows;
        const Scalar u0_2 = 2 * u[0];
        const Scalar u1_2 = 2 * u[1];

        Scalar *xptr = X.colptr(oj) + oi;
        if(nrow == 2)
        {
            for(int i = 0; i < ncol; i++, xptr += stride)
            {
                Scalar tmp = u0_2 * xptr[0] + u1_2 * xptr[1];
                xptr[0] -= tmp * u[0];
                xptr[1] -= tmp * u[1];
            }
        } else {
            const Scalar u2_2 = 2 * u[2];
            for(int i = 0; i < ncol; i++, xptr += stride)
            {
                Scalar tmp = u0_2 * xptr[0] + u1_2 * xptr[1] + u2_2 * xptr[2];
                xptr[0] -= tmp * u[0];
                xptr[1] -= tmp * u[1];
                xptr[2] -= tmp * u[2];
            }
        }
    }

    // x is a pointer to a vector
    // Px = x - 2 * dot(x, u) * u
    void apply_PX(Scalar *x, int u_ind)
    {
        Scalar u0 = ref_u(0, u_ind),
               u1 = ref_u(1, u_ind),
               u2 = ref_u(2, u_ind);

        if(std::abs(u0) + std::abs(u1) + std::abs(u2) <= 3 * prec)
            return;

        // When the reflector only contains two elements, u2 has been set to zero
        bool u2_is_zero = (std::abs(u2) <= prec);
        Scalar dot2 = x[0] * u0 + x[1] * u1 + (u2_is_zero ? 0 : (x[2] * u2));
        dot2 *= 2;
        x[0] -= dot2 * u0;
        x[1] -= dot2 * u1;
        if(!u2_is_zero)
            x[2] -= dot2 * u2;
    }

    // XP = X - 2 * (X * u) * u'
    void apply_XP(Matrix &X, int oi, int oj, int nrow, int ncol, int u_ind)
    {
        Scalar *u = ref_u.memptr() + 3 * u_ind;

        if(std::abs(u[0]) + std::abs(u[1]) + std::abs(u[2]) <= 3 * prec)
            return;

        int stride = X.n_rows;
        const Scalar u0_2 = 2 * u[0];
        const Scalar u1_2 = 2 * u[1];
        Scalar *X0 = X.colptr(oj) + oi, *X1 = X0 + stride;  // X0 => X(oi, oj), X1 => X(oi, oj + 1)

        if(ncol == 2)
        {
            // tmp = 2 * u0 * X0 + 2 * u1 * X1
            // X0 => X0 - u0 * tmp
            // X1 => X1 - u1 * tmp
            for(int i = 0; i < nrow; i++)
            {
                Scalar tmp = u0_2 * X0[i] + u1_2 * X1[i];
                X0[i] -= tmp * u[0];
                X1[i] -= tmp * u[1];
            }
        } else {
            Scalar *X2 = X1 + stride;  // X2 => X(oi, oj + 2)
            const Scalar u2_2 = 2 * u[2];
            for(int i = 0; i < nrow; i++)
            {
                Scalar tmp = u0_2 * X0[i] + u1_2 * X1[i] + u2_2 * X2[i];
                X0[i] -= tmp * u[0];
                X1[i] -= tmp * u[1];
                X2[i] -= tmp * u[2];
            }
        }
    }

public:
    DoubleShiftQR(int size) :
        n(size),
        prec(std::numeric_limits<Scalar>::epsilon()),
        prec2(std::min(std::pow(prec, Scalar(2.0) / 3), n * prec)),
        computed(false)
    {}

    DoubleShiftQR(const Matrix &mat, Scalar s, Scalar t) :
        n(mat.n_rows),
        mat_H(n, n),
        shift_s(s),
        shift_t(t),
        ref_u(3, n),
        prec(std::numeric_limits<Scalar>::epsilon()),
        prec2(std::min(std::pow(prec, Scalar(2.0) / 3), n * prec)),
        computed(false)
    {
        compute(mat, s, t);
    }

    void compute(const Matrix &mat, Scalar s, Scalar t)
    {
        if(!mat.is_square())
            throw std::invalid_argument("DoubleShiftQR: matrix must be square");

        n = mat.n_rows;
        mat_H.set_size(n, n);
        shift_s = s;
        shift_t = t;
        ref_u.set_size(3, n);

        // Make a copy of mat
        mat_H = mat;

        // Obtain the indices of zero elements in the subdiagonal,
        // so that H can be divided into several blocks
        std::vector<int> zero_ind;
        zero_ind.reserve(n / 2);
        zero_ind.push_back(0);
        Scalar *Hii = mat_H.memptr();
        for(int i = 0; i < n - 2; i++, Hii += (n + 1))
        {
            // Hii[1] => mat_H(i + 1, i)
            if(std::abs(Hii[1]) <= prec2)
            {
                Hii[1] = 0;
                zero_ind.push_back(i + 1);
            }
            // Make sure mat_H is upper Hessenberg
            // Zero the elements below mat_H(i + 1, i)
            std::fill(Hii + 2, Hii + n - i, Scalar(0));
        }
        zero_ind.push_back(n);

        for(std::vector<int>::size_type i = 0; i < zero_ind.size() - 1; i++)
        {
            int start = zero_ind[i];
            int end = zero_ind[i + 1] - 1;
            int block_size = end - start + 1;
            // Compute refelctors from each block X
            compute_reflectors_from_block(mat_H, start, block_size, start);
        }

        computed = true;
    }

    Matrix matrix_QtHQ()
    {
        if(!computed)
            throw std::logic_error("DoubleShiftQR: need to call compute() first");

        return mat_H;
    }

    // Q = P0 * P1 * ...
    // Q'y = P_{n-2} * ... * P1 * P0 * y
    void apply_QtY(Vector &y)
    {
        if(!computed)
            throw std::logic_error("DoubleShiftQR: need to call compute() first");

        Scalar *y_ptr = y.memptr();
        for(int i = 0; i < n - 1; i++, y_ptr++)
        {
            apply_PX(y_ptr, i);
        }
    }

    // Q = P0 * P1 * ...
    // YQ = Y * P0 * P1 * ...
    void apply_YQ(Matrix &Y)
    {
        if(!computed)
            throw std::logic_error("DoubleShiftQR: need to call compute() first");

        int nrow = Y.n_rows;
        for(int i = 0; i < n - 2; i++)
        {
            apply_XP(Y, 0, i, nrow, 3, i);
        }
        apply_XP(Y, 0, n - 2, nrow, 2, n - 2);
    }
};


#endif // DOUBLE_SHIFT_QR_H
