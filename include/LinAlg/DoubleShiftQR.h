#ifndef DOUBLE_SHIFT_QR_H
#define DOUBLE_SHIFT_QR_H

#include <armadillo>
#include <vector>
#include <stdexcept>

template <typename Scalar = double>
class DoubleShiftQR
{
private:
    typedef arma::Mat<Scalar> Matrix;
    typedef arma::Col<Scalar> Vector;

    int n;
    Matrix mat_H;
    Scalar shift_s;
    Scalar shift_t;
    // Householder reflectors
    Matrix ref_u;
    // Approximately zero
    const Scalar prec;
    bool computed;

    void compute_reflector(const Scalar &x1, const Scalar &x2, const Scalar &x3, int ind)
    {
        Scalar tmp = x2 * x2 + x3 * x3;
        // x1' = x1 - rho * ||x||
        // rho = -sign(x1)
        Scalar x1_new = x1 - ((x1 < 0) - (x1 > 0)) * std::sqrt(x1 * x1 + tmp);
        Scalar x_norm = std::sqrt(x1_new * x1_new + tmp);
        if(x_norm <= prec)
        {
            ref_u(0, ind) = 0;
            ref_u(1, ind) = 0;
            ref_u(2, ind) = 0;
        } else {
            ref_u(0, ind) = x1_new / x_norm;
            ref_u(1, ind) = x2 / x_norm;
            ref_u(2, ind) = x3 / x_norm;
        }
    }

    void compute_reflector(const Scalar *x, int ind)
    {
        compute_reflector(x[0], x[1], x[2], ind);
    }

    void compute_reflectors_from_block(Matrix &X, int start_ind)
    {
        // For the block X, we can assume that ncol == nrow,
        // and all sub-diagonal elements are non-zero
        const int nrow = X.n_rows;
        // For block size <= 2, there is no need to apply reflectors
        if(nrow == 1)
        {
            compute_reflector(0, 0, 0, start_ind);
            return;
        } else if(nrow == 2) {
            compute_reflector(0, 0, 0, start_ind);
            compute_reflector(0, 0, 0, start_ind + 1);
            return;
        }
        // For block size >=3, use the regular strategy
        Scalar x = X(0, 0) * (X(0, 0) - shift_s) + X(0, 1) * X(1, 0) + shift_t;
        Scalar y = X(1, 0) * (X(0, 0) + X(1, 1) - shift_s);
        Scalar z = X(2, 1) * X(1, 0);
        compute_reflector(x, y, z, start_ind);
        // Apply the first reflector
        apply_PX(X, 0, 0, 3, nrow, start_ind);
        apply_XP(X, 0, 0, std::min(nrow, 4), 3, start_ind);

        // Calculate the following reflectors
        for(int i = 1; i < nrow - 2; i++)
        {
            // If entering this loop, nrow is at least 4.

            compute_reflector(&X(i, i - 1), start_ind + i);
            // Apply the reflector to X
            apply_PX(X, i, i - 1, 3, nrow - i + 1, start_ind + i);
            apply_XP(X, 0, i, std::min(nrow, i + 4), 3, start_ind + i);
        }

        // The last reflector
        compute_reflector(X(nrow - 2, nrow - 3), X(nrow - 1, nrow - 3), 0, start_ind + nrow - 2);
        compute_reflector(0, 0, 0, start_ind + nrow - 1);
        // Apply the reflector to X
        apply_PX(X, nrow - 2, nrow - 3, 2, 3, start_ind + nrow - 2);
        apply_XP(X, 0, nrow - 2, nrow, 2, start_ind + nrow - 2);
    }

    // P = I - 2 * u * u' = P'
    // PX = X - 2 * u * (u'X)
    void apply_PX(Matrix &X, int oi, int oj, int nrow, int ncol, int u_ind)
    {
        const Scalar sqrt_2 = std::sqrt(Scalar(2));

        Scalar u0 = sqrt_2 * ref_u(0, u_ind),
               u1 = sqrt_2 * ref_u(1, u_ind),
               u2 = sqrt_2 * ref_u(2, u_ind);

        if(u0 * u0 + u1 * u1 + u2 * u2 <= prec)
            return;

        if(nrow == 2)
        {
            for(int i = 0; i < ncol; i++)
            {
                Scalar tmp = u0 * X(oi, oj + i) + u1 * X(oi + 1, oj + i);
                X(oi,     oj + i) -= tmp * u0;
                X(oi + 1, oj + i) -= tmp * u1;
            }
        } else {
            for(int i = 0; i < ncol; i++)
            {
                Scalar tmp = u0 * X(oi, oj + i) + u1 * X(oi + 1, oj + i) + u2 * X(oi + 2, oj + i);
                X(oi,     oj + i) -= tmp * u0;
                X(oi + 1, oj + i) -= tmp * u1;
                X(oi + 2, oj + i) -= tmp * u2;
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

        if(u0 * u0 + u1 * u1 + u2 * u2 <= prec)
            return;

        Scalar dot2 = x[0] * u0 + x[1] * u1 + (std::abs(u2) <= prec ? 0 : (x[2] * u2));
        dot2 *= 2;
        x[0] -= dot2 * u0;
        x[1] -= dot2 * u1;
        if(std::abs(u2) > prec)
            x[2] -= dot2 * u2;
    }

    // XP = X - 2 * (X * u) * u'
    void apply_XP(Matrix &X, int oi, int oj, int nrow, int ncol, int u_ind)
    {
        const Scalar sqrt_2 = std::sqrt(Scalar(2));

        Scalar u0 = sqrt_2 * ref_u(0, u_ind),
               u1 = sqrt_2 * ref_u(1, u_ind),
               u2 = sqrt_2 * ref_u(2, u_ind);
        Scalar *X0 = &X(oi, oj), *X1 = &X(oi, oj + 1);

        if(u0 * u0 + u1 * u1 + u2 * u2 <= prec)
            return;

        if(ncol == 2)
        {
            for(int i = 0; i < nrow; i++)
            {
                Scalar tmp = u0 * X0[i] + u1 * X1[i];
                X0[i] -= tmp * u0;
                X1[i] -= tmp * u1;
            }
        } else {
            Scalar *X2 = &X(oi, oj + 2);
            for(int i = 0; i < nrow; i++)
            {
                Scalar tmp = u0 * X0[i] + u1 * X1[i] + u2 * X2[i];
                X0[i] -= tmp * u0;
                X1[i] -= tmp * u1;
                X2[i] -= tmp * u2;
            }
        }
    }

public:
    DoubleShiftQR() :
        n(0),
        prec(std::pow(std::numeric_limits<Scalar>::epsilon(), Scalar(0.9))),
        computed(false)
    {}

    DoubleShiftQR(const Matrix &mat, Scalar s, Scalar t) :
        n(mat.n_rows),
        mat_H(n, n),
        shift_s(s),
        shift_t(t),
        ref_u(3, n),
        prec(std::pow(std::numeric_limits<Scalar>::epsilon(), Scalar(0.9))),
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

        mat_H = arma::trimatu(mat);
        mat_H.diag(-1) = mat.diag(-1);

        std::vector<int> zero_ind;
        zero_ind.reserve(n - 1);
        zero_ind.push_back(0);
        for(int i = 1; i < n - 1; i++)
        {
            if(std::abs(mat_H(i, i - 1)) <= prec)
            {
                mat_H(i, i - 1) = 0;
                zero_ind.push_back(i);
            }
        }
        zero_ind.push_back(n);

        for(std::vector<int>::size_type i = 0; i < zero_ind.size() - 1; i++)
        {
            int start = zero_ind[i];
            int end = zero_ind[i + 1] - 1;
            // Call this block X
            Matrix tmp = mat_H.submat(start, start, arma::size(end - start + 1, end - start + 1));
            compute_reflectors_from_block(tmp, start);
            mat_H.submat(start, start, arma::size(end - start + 1, end - start + 1)) = tmp;
            // Apply reflectors to the block right to X
            if(end < n - 1 && end - start >= 2)
            {
                for(int j = start; j < end; j++)
                {
                    apply_PX(mat_H, j, end + 1, std::min(3, end - j + 1), n - 1 - end, j);
                }
            }
            // Apply reflectors to the block above X
            if(start > 0 && end - start >= 2)
            {
                for(int j = start; j < end; j++)
                {
                    apply_XP(mat_H, 0, j, start, std::min(3, end - j + 1), j);
                }
            }
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
