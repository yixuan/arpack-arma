#ifndef UpperHessenbergQR_H
#define UpperHessenbergQR_H

#include <armadillo>

// QR decomposition of an upper Hessenberg matrix
template <typename Scalar = double>
class UpperHessenbergQR
{
private:
    typedef arma::Mat<Scalar> Matrix;
    typedef arma::Col<Scalar> Vector;

protected:
    int n;
    Matrix mat_T;
    // Gi = [ cos[i]  sin[i]]
    //      [-sin[i]  cos[i]]
    // Q = G1 * G2 * ... * G_{n-1}
    Vector rot_cos;
    Vector rot_sin;
public:
    UpperHessenbergQR() :
        n(0)
    {}

    UpperHessenbergQR(int n_) :
        n(n_),
        mat_T(n, n),
        rot_cos(n - 1),
        rot_sin(n - 1)
    {}

    UpperHessenbergQR(const Matrix &mat) :
        n(mat.n_rows),
        mat_T(n, n),
        rot_cos(n - 1),
        rot_sin(n - 1)
    {
        compute(mat);
    }

    virtual void compute(const Matrix &mat)
    {
        n = mat.n_rows;
        mat_T.set_size(n, n);
        rot_cos.set_size(n - 1);
        rot_sin.set_size(n - 1);

        mat_T = mat;

        Scalar xi, xj, r, c, s;
        for(int i = 0; i < n - 2; i++)
        {
            xi = mat_T(i, i);
            xj = mat_T(i + 1, i);
            r = std::sqrt(xi * xi + xj * xj);
            rot_cos[i] = c = xi / r;
            rot_sin[i] = s = -xj / r;
            // For a complete QR decomposition,
            // we first obtain the rotation matrix
            // G = [ cos  sin]
            //     [-sin  cos]
            // and then do T[i:(i + 1), i:n] = G' * T[i:(i + 1), i:n]
            // Since here we only want to obtain the cos and sin sequence,
            // we only update T[i + 1, (i + 1):n]
            mat_T(i + 1, arma::span(i + 1, n - 1)) *= c;
            mat_T(i + 1, arma::span(i + 1, n - 1)) += s * mat_T(i, arma::span(i + 1, n - 1));
            // Matrix Gt;
            // Gt << c << -s << arma::endr << s << c << arma::endr;
            // mat_T.rows(i, i + 1) = Gt * mat_T.rows(i, i + 1);
        }
        // For i = n - 2
        xi = mat_T(n - 2, n - 2);
        xj = mat_T(n - 1, n - 2);
        r = std::sqrt(xi * xi + xj * xj);
        rot_cos[n - 2] = xi / r;
        rot_sin[n - 2] = -xj / r;
    }

    // Y -> QY = G1 * G2 * ... * Y
    virtual void applyQY(Vector &Y)
    {
        Scalar c, s, Yi, Yi1;
        for(int i = n - 2; i >= 0; i--)
        {
            // Y[i:(i + 1)] = Gi * Y[i:(i + 1)]
            // Gi = [ cos[i]  sin[i]]
            //      [-sin[i]  cos[i]]
            c = rot_cos[i];
            s = rot_sin[i];
            Yi = Y[i];
            Yi1 = Y[i + 1];
            Y[i] = c * Yi + s * Yi1;
            Y[i + 1] = -s * Yi + c * Yi1;
        }
    }

    // Y -> Q'Y = G_{n-1}' * ... * G2' * G1' * Y
    virtual void applyQtY(Vector &Y)
    {
        Scalar c, s, Yi, Yi1;
        for(int i = 0; i < n - 1; i++)
        {
            // Y[i:(i + 1)] = Gi' * Y[i:(i + 1)]
            // Gi = [ cos[i]  sin[i]]
            //      [-sin[i]  cos[i]]
            c = rot_cos[i];
            s = rot_sin[i];
            Yi = Y[i];
            Yi1 = Y[i + 1];
            Y[i] = c * Yi - s * Yi1;
            Y[i + 1] = s * Yi + c * Yi1;
        }
    }

    // Y -> QY = G1 * G2 * ... * Y
    virtual void applyQY(Matrix &Y)
    {
        Matrix Gi(2, 2);
        for(int i = n - 2; i >= 0; i--)
        {
            // Y[i:(i + 1), ] = Gi * Y[i:(i + 1), ]
            // Gi = [ cos[i]  sin[i]]
            //      [-sin[i]  cos[i]]
            Gi(1, 1) = Gi(0, 0) = rot_cos[i];
            Gi(0, 1) = rot_sin[i];
            Gi(1, 0) = -rot_sin[i];
            Y.rows(i, i + 1) = Gi * Y.rows(i, i + 1);
        }
    }

    // Y -> Q'Y = G_{n-1}' * ... * G2' * G1' * Y
    virtual void applyQtY(Matrix &Y)
    {
        Matrix Git(2, 2);
        for(int i = 0; i < n - 1; i++)
        {
            // Y[i:(i + 1), ] = Gi' * Y[i:(i + 1), ]
            // Gi = [ cos[i]  sin[i]]
            //      [-sin[i]  cos[i]]
            Git(1, 1) = Git(0, 0) = rot_cos[i];
            Git(0, 1) = -rot_sin[i];
            Git(1, 0) = rot_sin[i];
            Y.rows(i, i + 1) = Git * Y.rows(i, i + 1);
        }
    }

    // Y -> YQ = Y * G1 * G2 * ...
    virtual void applyYQ(Matrix &Y)
    {
        Matrix Gi(2, 2);
        for(int i = 0; i < n - 1; i++)
        {
            // Y[, i:(i + 1)] = Y[, i:(i + 1)] * Gi
            // Gi = [ cos[i]  sin[i]]
            //      [-sin[i]  cos[i]]
            Gi(1, 1) = Gi(0, 0) = rot_cos[i];
            Gi(0, 1) = rot_sin[i];
            Gi(1, 0) = -rot_sin[i];
            Y.cols(i, i + 1) = Y.cols(i, i + 1) * Gi;
        }
    }

    // Y -> YQ' = Y * G_{n-1}' * ... * G2' * G1'
    virtual void applyYQt(Matrix &Y)
    {
        Matrix Git(2, 2);
        for(int i = n - 2; i >= 0; i--)
        {
            // Y[, i:(i + 1)] = Y[, i:(i + 1)] * Gi'
            // Gi = [ cos[i]  sin[i]]
            //      [-sin[i]  cos[i]]
            Git(1, 1) = Git(0, 0) = rot_cos[i];
            Git(0, 1) = -rot_sin[i];
            Git(1, 0) = rot_sin[i];
            Y.cols(i, i + 1) = Y.cols(i, i + 1) * Git;
        }
    }
};



// QR decomposition of a tridiagonal matrix as a special case of
// upper Hessenberg matrix
template <typename Scalar = double>
class TridiagQR: public UpperHessenbergQR<Scalar>
{
private:
    typedef arma::Mat<Scalar> Matrix;
    typedef arma::Col<Scalar> Vector;

public:
    TridiagQR() :
        UpperHessenbergQR<Scalar>()
    {}

    TridiagQR(int n_) :
        UpperHessenbergQR<Scalar>(n_)
    {}

    TridiagQR(const Matrix &mat) :
        UpperHessenbergQR<Scalar>(mat.n_rows)
    {
        this->compute(mat);
    }

    virtual void compute(const Matrix &mat)
    {
        this->n = mat.n_rows;
        this->mat_T.set_size(this->n, this->n);
        this->rot_cos.set_size(this->n - 1);
        this->rot_sin.set_size(this->n - 1);

        this->mat_T.zeros();
        this->mat_T.diag() = mat.diag();
        this->mat_T.diag(1) = mat.diag(-1);
        this->mat_T.diag(-1) = mat.diag(-1);

        Scalar xi, xj, r, c, s;
        for(int i = 0; i < this->n - 2; i++)
        {
            xi = this->mat_T(i, i);
            xj = this->mat_T(i + 1, i);
            r = std::sqrt(xi * xi + xj * xj);
            this->rot_cos[i] = c = xi / r;
            this->rot_sin[i] = s = -xj / r;
            // For a complete QR decomposition,
            // we first obtain the rotation matrix
            // G = [ cos  sin]
            //     [-sin  cos]
            // and then do T[i:(i + 1), i:(i + 2)] = G' * T[i:(i + 1), i:(i + 2)]
            // Since here we only want to obtain the cos and sin sequence,
            // we only update T[i + 1, (i + 1):(i + 2)]
            this->mat_T(i + 1, i + 1) = s * this->mat_T(i, i + 1) + c * this->mat_T(i + 1, i + 1);
            this->mat_T(i + 1, i + 2) *= c;

        }
        // For i = n - 2
        xi = this->mat_T(this->n - 2, this->n - 2);
        xj = this->mat_T(this->n - 1, this->n - 2);
        r = std::sqrt(xi * xi + xj * xj);
        this->rot_cos[this->n - 2] = xi / r;
        this->rot_sin[this->n - 2] = -xj / r;
    }
};



#endif // UpperHessenbergQR_H
