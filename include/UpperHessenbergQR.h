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
    typedef arma::Row<Scalar> RowVector;

protected:
    int n;
    Matrix mat_T;
    // Gi = [ cos[i]  sin[i]]
    //      [-sin[i]  cos[i]]
    // Q = G1 * G2 * ... * G_{n-1}
    Vector rot_cos;
    Vector rot_sin;
    bool computed;
public:
    UpperHessenbergQR() :
        n(0), computed(false)
    {}

    UpperHessenbergQR(int n_) :
        n(n_),
        mat_T(n, n),
        rot_cos(n - 1),
        rot_sin(n - 1),
        computed(false)
    {}

    UpperHessenbergQR(const Matrix &mat) :
        n(mat.n_rows),
        mat_T(n, n),
        rot_cos(n - 1),
        rot_sin(n - 1),
        computed(false)
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
        Matrix Gt(2, 2);
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
            // and then do T[i:(i + 1), i:(n - 1)] = G' * T[i:(i + 1), i:(n - 1)]

            Gt << c << -s << arma::endr << s << c << arma::endr;
            mat_T.submat(i, i, i + 1, n - 1) = Gt * mat_T.submat(i, i, i + 1, n - 1);

            // If we do not need to calculate the R matrix, then
            // only the cos and sin sequences are required.
            // In such case we only update T[i + 1, (i + 1):(n - 1)]
            // mat_T(i + 1, arma::span(i + 1, n - 1)) *= c;
            // mat_T(i + 1, arma::span(i + 1, n - 1)) += s * mat_T(i, arma::span(i + 1, n - 1));
        }
        // For i = n - 2
        xi = mat_T(n - 2, n - 2);
        xj = mat_T(n - 1, n - 2);
        r = std::sqrt(xi * xi + xj * xj);
        rot_cos[n - 2] = c = xi / r;
        rot_sin[n - 2] = s = -xj / r;
        Gt << c << -s << arma::endr << s << c << arma::endr;
        mat_T.submat(n - 2, n - 2, n - 1, n - 1) = Gt * mat_T.submat(n - 2, n - 2, n - 1, n - 1);

        computed = true;
    }

    Matrix matrix_R()
    {
        if(!computed)
            return Matrix();

        return mat_T;
    }

    // Y -> QY = G1 * G2 * ... * Y
    void apply_QY(Vector &Y)
    {
        if(!computed)
            return;

        Scalar *c = rot_cos.memptr() + n - 2,
               *s = rot_sin.memptr() + n - 2,
               *Yi = Y.memptr() + n - 2,
               tmp;
        for(int i = n - 2; i >= 0; i--)
        {
            // Y[i:(i + 1)] = Gi * Y[i:(i + 1)]
            // Gi = [ cos[i]  sin[i]]
            //      [-sin[i]  cos[i]]
            tmp = *Yi;
            // Yi[0] == Y[i], Yi[1] == Y[i + 1]
            Yi[0] =  (*c) * tmp + (*s) * Yi[1];
            Yi[1] = -(*s) * tmp + (*c) * Yi[1];

            Yi--;
            c--;
            s--;
        }
    }

    // Y -> Q'Y = G_{n-1}' * ... * G2' * G1' * Y
    void apply_QtY(Vector &Y)
    {
        if(!computed)
            return;

        Scalar *c = rot_cos.memptr(),
               *s = rot_sin.memptr(),
               *Yi = Y.memptr(),
               tmp;
        for(int i = 0; i < n - 1; i++)
        {
            // Y[i:(i + 1)] = Gi' * Y[i:(i + 1)]
            // Gi = [ cos[i]  sin[i]]
            //      [-sin[i]  cos[i]]
            tmp = *Yi;
            // Yi[0] == Y[i], Yi[1] == Y[i + 1]
            Yi[0] = (*c) * tmp - (*s) * Yi[1];
            Yi[1] = (*s) * tmp + (*c) * Yi[1];

            Yi++;
            c++;
            s++;
        }
    }

    // Y -> QY = G1 * G2 * ... * Y
    void apply_QY(Matrix &Y)
    {
        if(!computed)
            return;

        Scalar *c = rot_cos.memptr() + n - 2,
               *s = rot_sin.memptr() + n - 2;
        RowVector Yi(Y.n_cols);
        for(int i = n - 2; i >= 0; i--)
        {
            // Y[i:(i + 1), ] = Gi * Y[i:(i + 1), ]
            // Gi = [ cos[i]  sin[i]]
            //      [-sin[i]  cos[i]]
            Yi = Y.row(i);
            Y.row(i)     =  (*c) * Yi + (*s) * Y.row(i + 1);
            Y.row(i + 1) = -(*s) * Yi + (*c) * Y.row(i + 1);
            c--;
            s--;
        }
    }

    // Y -> Q'Y = G_{n-1}' * ... * G2' * G1' * Y
    void apply_QtY(Matrix &Y)
    {
        if(!computed)
            return;

        Scalar *c = rot_cos.memptr(),
               *s = rot_sin.memptr();
        RowVector Yi(Y.n_cols);
        for(int i = 0; i < n - 1; i++)
        {
            // Y[i:(i + 1), ] = Gi' * Y[i:(i + 1), ]
            // Gi = [ cos[i]  sin[i]]
            //      [-sin[i]  cos[i]]
            Yi = Y.row(i);
            Y.row(i)     = (*c) * Yi - (*s) * Y.row(i + 1);
            Y.row(i + 1) = (*s) * Yi + (*c) * Y.row(i + 1);
            c++;
            s++;
        }
    }

    // Y -> YQ = Y * G1 * G2 * ...
    void apply_YQ(Matrix &Y)
    {
        if(!computed)
            return;

        Scalar *c = rot_cos.memptr(),
               *s = rot_sin.memptr();
        Vector Yi(Y.n_rows);
        for(int i = 0; i < n - 1; i++)
        {
            // Y[, i:(i + 1)] = Y[, i:(i + 1)] * Gi
            // Gi = [ cos[i]  sin[i]]
            //      [-sin[i]  cos[i]]
            Yi = Y.col(i);
            Y.col(i)     = (*c) * Yi - (*s) * Y.col(i + 1);
            Y.col(i + 1) = (*s) * Yi + (*c) * Y.col(i + 1);
            c++;
            s++;
        }
    }

    // Y -> YQ' = Y * G_{n-1}' * ... * G2' * G1'
    void apply_YQt(Matrix &Y)
    {
        if(!computed)
            return;

        Scalar *c = rot_cos.memptr() + n - 2,
               *s = rot_sin.memptr() + n - 2;
        Vector Yi(Y.n_rows);
        for(int i = n - 2; i >= 0; i--)
        {
            // Y[, i:(i + 1)] = Y[, i:(i + 1)] * Gi'
            // Gi = [ cos[i]  sin[i]]
            //      [-sin[i]  cos[i]]
            Yi = Y.col(i);
            Y.col(i)     =  (*c) * Yi + (*s) * Y.col(i + 1);
            Y.col(i + 1) = -(*s) * Yi + (*c) * Y.col(i + 1);
            c--;
            s--;
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

        // A number of pointers to avoid repeated address calculation
        Scalar *Tii = this->mat_T.memptr(),  // pointer to T[i, i]
               *ptr,                         // some location relative to Tii
               *c = this->rot_cos.memptr(),  // pointer to the cosine vector
               *s = this->rot_sin.memptr(),  // pointer to the sine vector
               r, tmp;
        for(int i = 0; i < this->n - 2; i++)
        {
            // Tii[0] == T[i, i]
            // Tii[1] == T[i + 1, i]
            r = std::sqrt(Tii[0] * Tii[0] + Tii[1] * Tii[1]);
            *c =  Tii[0] / r;
            *s = -Tii[1] / r;

            // For a complete QR decomposition,
            // we first obtain the rotation matrix
            // G = [ cos  sin]
            //     [-sin  cos]
            // and then do T[i:(i + 1), i:(i + 2)] = G' * T[i:(i + 1), i:(i + 2)]

            // Update T[i, i] and T[i + 1, i]
            // The updated value of T[i, i] is known to be r
            // The updated value of T[i + 1, i] is known to be 0
            Tii[0] = r;
            Tii[1] = 0;
            // Update T[i, i + 1] and T[i + 1, i + 1]
            // ptr[0] == T[i, i + 1]
            // ptr[1] == T[i + 1, i + 1]
            ptr = Tii + this->n;
            tmp = *ptr;
            ptr[0] = (*c) * tmp - (*s) * ptr[1];
            ptr[1] = (*s) * tmp + (*c) * ptr[1];
            // Update T[i, i + 2] and T[i + 1, i + 2]
            // ptr[0] == T[i, i + 2] == 0
            // ptr[1] == T[i + 1, i + 2]
            ptr += this->n;
            ptr[0] = -(*s) * ptr[1];
            ptr[1] *= (*c);

            // Move from T[i, i] to T[i + 1, i + 1]
            Tii += this->n + 1;
            // Increase c and s by 1
            c++;
            s++;


            // If we do not need to calculate the R matrix, then
            // only the cos and sin sequences are required.
            // In such case we only update T[i + 1, (i + 1):(i + 2)]
            // this->mat_T(i + 1, i + 1) = (*c) * this->mat_T(i + 1, i + 1) + (*s) * this->mat_T(i, i + 1);
            // this->mat_T(i + 1, i + 2) *= (*c);
        }
        // For i = n - 2
        r = std::sqrt(Tii[0] * Tii[0] + Tii[1] * Tii[1]);
        *c =  Tii[0] / r;
        *s = -Tii[1] / r;
        Tii[0] = r;
        Tii[1] = 0;
        ptr = Tii + this->n;  // points to T[i - 2, i - 1]
        tmp = *ptr;
        ptr[0] = (*c) * tmp - (*s) * ptr[1];
        ptr[1] = (*s) * tmp + (*c) * ptr[1];

        this->computed = true;
    }
};



#endif // UpperHessenbergQR_H
