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
    typedef arma::Col<unsigned short> IntVector;
    typedef arma::sword Index;

    Index n;            // Dimension of the matrix
    Matrix mat_H;       // A copy of the matrix to be factorized
    Scalar shift_s;     // Shift constant
    Scalar shift_t;     // Shift constant
    Matrix ref_u;       // Householder reflectors
    IntVector ref_nr;   // How many rows does each reflector affects
                        // 3 - A general reflector
                        // 2 - A Givens rotation
                        // 1 - An identity transformation
    const Scalar prec;  // Approximately zero
    const Scalar eps_rel;
    const Scalar eps_abs;
    bool computed;      // Whether matrix has been factorized

    void compute_reflector(const Scalar &x1, const Scalar &x2, const Scalar &x3, Index ind)
    {
        Scalar *u = ref_u.memptr() + 3 * ind;
        unsigned short *nr = ref_nr.memptr();

        // In general case the reflector affects 3 rows
        nr[ind] = 3;
        // If x3 is zero, decrease nr by 1
        if(std::abs(x3) < prec)
        {
            // If x2 is also zero, nr will be 1, and we can exit this function
            if(std::abs(x2) < prec)
            {
                nr[ind] = 1;
                return;
            } else {
                nr[ind] = 2;
            }
        }

        // x1' = x1 - rho * ||x||
        // rho = -sign(x1), if x1 == 0, we choose rho = 1
        Scalar tmp = x2 * x2 + x3 * x3;
        Scalar x1_new = x1 - ((x1 <= 0) - (x1 > 0)) * std::sqrt(x1 * x1 + tmp);
        Scalar x_norm = std::sqrt(x1_new * x1_new + tmp);
        // Double check the norm of new x
        if(x_norm < prec)
        {
            nr[ind] = 1;
            return;
        }
        u[0] = x1_new / x_norm;
        u[1] = x2 / x_norm;
        u[2] = x3 / x_norm;
    }

    void compute_reflector(const Scalar *x, Index ind)
    {
        compute_reflector(x[0], x[1], x[2], ind);
    }

    // Update the block X = H(il:iu, il:iu)
    void update_block(Index il, Index iu)
    {
        // Block size
        Index bsize = iu - il + 1;

        // If block size == 1, there is no need to apply reflectors
        if(bsize == 1)
        {
            ref_nr[il] = 1;
            return;
        }

        // For block size == 2, do a Givens rotation on M = X * X - s * X + t * I
        if(bsize == 2)
        {
            // m00 = x00 * (x00 - s) + x01 * x10 + t
            Scalar m00 = mat_H(il, il) * (mat_H(il, il) - shift_s) +
                         mat_H(il, il + 1) * mat_H(il + 1, il) +
                         shift_t;
            // m10 = x10 * (x00 + x11 - s)
            Scalar m10 = mat_H(il + 1, il) * (mat_H(il, il) + mat_H(il + 1, il + 1) - shift_s);
            // This causes nr=2
            compute_reflector(m00, m10, 0, il);
            // Apply the reflector to X
            apply_PX(mat_H, il, il, 2, n - il, il);
            apply_XP(mat_H, 0, il, il + 2, 2, il);

            ref_nr[il + 1] = 1;
            return;
        }

        // For block size >=3, use the regular strategy
        Scalar m00 = mat_H(il, il) * (mat_H(il, il) - shift_s) +
                     mat_H(il, il + 1) * mat_H(il + 1, il) +
                     shift_t;
        Scalar m10 = mat_H(il + 1, il) * (mat_H(il, il) + mat_H(il + 1, il + 1) - shift_s);
        // m20 = x21 * x10
        Scalar m20 = mat_H(il + 2, il + 1) * mat_H(il + 1, il);
        compute_reflector(m00, m10, m20, il);

        // Apply the first reflector
        apply_PX(mat_H, il, il, 3, n - il, il);
        apply_XP(mat_H, 0, il, il + std::min(bsize, Index(4)), 3, il);

        // Calculate the following reflectors
        // If entering this loop, block size is at least 4.
        for(Index i = 1; i < bsize - 2; i++)
        {
            compute_reflector(mat_H.colptr(il + i - 1) + il + i, il + i);
            // Apply the reflector to X
            apply_PX(mat_H, il + i, il + i - 1, 3, n - il - i + 1, il + i);
            apply_XP(mat_H, 0, il + i, il + std::min(bsize, Index(i + 4)), 3, il + i);
        }

        // The last reflector
        // This causes nr=2
        compute_reflector(mat_H(iu - 1, iu - 2), mat_H(iu, iu - 2), 0, iu - 1);
        // Apply the reflector to X
        apply_PX(mat_H, iu - 1, iu - 2, 2, n - iu + 2, iu - 1);
        apply_XP(mat_H, 0, iu - 1, il + bsize, 2, iu - 1);

        ref_nr[iu] = 1;
    }

    // P = I - 2 * u * u' = P'
    // PX = X - 2 * u * (u'X)
    void apply_PX(Matrix &X, Index oi, Index oj, Index nrow, Index ncol, Index u_ind)
    {
        if(ref_nr[u_ind] == 1)
            return;

        Scalar *u = ref_u.memptr() + 3 * u_ind;

        const Index stride = X.n_rows;
        const Scalar u0_2 = 2 * u[0];
        const Scalar u1_2 = 2 * u[1];

        Scalar *xptr = X.colptr(oj) + oi;
        if(ref_nr[u_ind] == 2 || nrow == 2)
        {
            for(Index i = 0; i < ncol; i++, xptr += stride)
            {
                Scalar tmp = u0_2 * xptr[0] + u1_2 * xptr[1];
                xptr[0] -= tmp * u[0];
                xptr[1] -= tmp * u[1];
            }
        } else {
            const Scalar u2_2 = 2 * u[2];
            for(Index i = 0; i < ncol; i++, xptr += stride)
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
    void apply_PX(Scalar *x, Index u_ind)
    {
        if(ref_nr[u_ind] == 1)
            return;

        Scalar u0 = ref_u(0, u_ind),
               u1 = ref_u(1, u_ind),
               u2 = ref_u(2, u_ind);

        // When the reflector only contains two elements, u2 has been set to zero
        bool nr_is_2 = (ref_nr[u_ind] == 2);
        Scalar dot2 = x[0] * u0 + x[1] * u1 + (nr_is_2 ? 0 : (x[2] * u2));
        dot2 *= 2;
        x[0] -= dot2 * u0;
        x[1] -= dot2 * u1;
        if(!nr_is_2)
            x[2] -= dot2 * u2;
    }

    // XP = X - 2 * (X * u) * u'
    void apply_XP(Matrix &X, Index oi, Index oj, Index nrow, Index ncol, Index u_ind)
    {
        if(ref_nr[u_ind] == 1)
            return;

        Scalar *u = ref_u.memptr() + 3 * u_ind;
        Index stride = X.n_rows;
        const Scalar u0_2 = 2 * u[0];
        const Scalar u1_2 = 2 * u[1];
        Scalar *X0 = X.colptr(oj) + oi, *X1 = X0 + stride;  // X0 => X(oi, oj), X1 => X(oi, oj + 1)

        if(ref_nr[u_ind] == 2 || ncol == 2)
        {
            // tmp = 2 * u0 * X0 + 2 * u1 * X1
            // X0 => X0 - u0 * tmp
            // X1 => X1 - u1 * tmp
            for(Index i = 0; i < nrow; i++)
            {
                Scalar tmp = u0_2 * X0[i] + u1_2 * X1[i];
                X0[i] -= tmp * u[0];
                X1[i] -= tmp * u[1];
            }
        } else {
            Scalar *X2 = X1 + stride;  // X2 => X(oi, oj + 2)
            const Scalar u2_2 = 2 * u[2];
            for(Index i = 0; i < nrow; i++)
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
        eps_rel(std::pow(prec, Scalar(0.75))),
        eps_abs(std::min(std::pow(prec, Scalar(0.75)), n * prec)),
        computed(false)
    {}

    DoubleShiftQR(const Matrix &mat, Scalar s, Scalar t) :
        n(mat.n_rows),
        mat_H(n, n),
        shift_s(s),
        shift_t(t),
        ref_u(3, n),
        ref_nr(n),
        prec(std::numeric_limits<Scalar>::epsilon()),
        eps_rel(std::pow(prec, Scalar(0.75))),
        eps_abs(std::min(std::pow(prec, Scalar(0.75)), n * prec)),
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
        ref_nr.set_size(n);

        // Make a copy of mat
        std::copy(mat.memptr(), mat.memptr() + mat.n_elem, mat_H.memptr());

        // Obtain the indices of zero elements in the subdiagonal,
        // so that H can be divided into several blocks
        std::vector<int> zero_ind;
        zero_ind.reserve(n - 1);
        zero_ind.push_back(0);
        Scalar *Hii = mat_H.memptr();
        for(Index i = 0; i < n - 2; i++, Hii += (n + 1))
        {
            // Hii[1] => mat_H(i + 1, i)
            const Scalar h = std::abs(Hii[1]);
            if(h <= eps_abs || h <= eps_rel * (std::abs(Hii[0]) + std::abs(Hii[n + 1])))
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
            Index start = zero_ind[i];
            Index end = zero_ind[i + 1] - 1;
            // Compute refelctors from each block X
            update_block(start, end);
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
