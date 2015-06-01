#ifndef MATOP_H
#define MATOP_H

template <typename Scalar>
class MatOp
{
public:
    // Destructor
    virtual ~MatOp() {}

    // y_out = A * x_in
    virtual void prod(Scalar *x_in, Scalar *y_out) = 0;

    // Dimension of the matrix
    // In eigenvalue problems, they should be equal.
    virtual int rows() = 0;
    virtual int cols() = 0;
};

template <typename Scalar>
class MatOpWithTransProd: virtual public MatOp<Scalar>
{
public:
    // Destructor
    virtual ~MatOpWithTransProd() {}

    // y_out = A' * x_in
    virtual void trans_prod(Scalar *x_in, Scalar *y_out) = 0;
};

template <typename Scalar>
class MatOpWithRealShiftSolve: virtual public MatOp<Scalar>
{
public:
    // Destructor
    virtual ~MatOpWithRealShiftSolve() {}

    // setting sigma
    virtual void set_shift(Scalar sigma) = 0;
    // y_out = inv(A - sigma * I) * x_in
    virtual void shift_solve(Scalar *x_in, Scalar *y_out) = 0;
};

template <typename Scalar>
class MatOpWithComplexShiftSolve: public MatOpWithRealShiftSolve<Scalar>
{
public:
    // Destructor
    virtual ~MatOpWithComplexShiftSolve() {}

    // setting real shift
    virtual void set_shift(Scalar sigma)
    {
        this->set_shift(sigma, Scalar(0));
    }
    // setting complex shift
    virtual void set_shift(Scalar sigmar, Scalar sigmai) = 0;
};


#endif // MATOP_H
