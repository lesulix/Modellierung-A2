// -----------------------------------------------------------------------
// <copyright file="LU.cs" company="">
// Original CSparse code by Timothy A. Davis, http://www.suitesparse.com
// CSparse.NET code by Christian Woltering, http://csparse.codeplex.com/
// </copyright>
// -----------------------------------------------------------------------

namespace CSparse.Complex
{
    using System;
    using System.Numerics;

    /// <summary>
    /// LU decomposition.
    /// </summary>
    public class LU
    {
        SymbolicFactorization symFactor;
        NumericFactorization numFactor;
        int n;

        /// <summary>
        /// Creates a LU factorization.
        /// </summary>
        /// <param name="order">ordering method to use (0 to 3)</param>
        /// <param name="A">column-compressed matrix</param>
        /// <param name="tol">partial pivoting tolerance</param>
        /// <returns>LU factorization</returns>
        public static LU Create(int order, SparseMatrix A, double tol)
        {
            var lu = new LU();

            int n = lu.n = A.n;

            lu.symFactor = SymbolicAnalysis(order, A); // ordering and symbolic analysis
            lu.numFactor = Factorize(A, lu.symFactor, tol); // numeric LU factorization

            return lu;
        }

        /// <summary>
        /// Solves Ax=b, where A is square and nonsingular. b overwritten with
        /// solution. Partial pivoting if tol = 1.
        /// </summary>
        /// <param name="b">size n, b on input, x on output</param>
        /// <returns>true if successful, false on error</returns>
        public bool Solve(Complex[] b)
        {
            if (b == null || numFactor == null) return false; // check inputs

            Complex[] x = new Complex[n]; // get workspace

            Common.InversePermute(numFactor.pinv, b, x, n); // x = b(p)
            Triangular.SolveL(numFactor.L, x); // x = L\x
            Triangular.SolveU(numFactor.U, x); // x = U\x
            Common.InversePermute(symFactor.q, x, b, n); // b(q) = x

            return true;
        }

        // [L,U,pinv]=lu(A, [q lnz unz]). lnz and unz can be guess
        static NumericFactorization Factorize(SparseMatrix A, SymbolicFactorization S, double tol)
        {
            int i, n = A.n;
            int[] q = S.q;

            int lnz = S.lnz;
            int unz = S.unz;

            int[] pinv;

            NumericFactorization N = new NumericFactorization(); // allocate result

            SparseMatrix L, U;
            N.L = L = new SparseMatrix(n, n, lnz, true); // allocate result L
            N.U = U = new SparseMatrix(n, n, unz, true); // allocate result U
            N.pinv = pinv = new int[n]; // allocate result pinv

            Complex[] x = new Complex[n]; // get double workspace
            int[] xi = new int[2 * n]; // get int workspace

            for (i = 0; i < n; i++)
            {
                pinv[i] = -1; // no rows pivotal yet
            }

            lnz = unz = 0;

            int ipiv, top, p, col;
            Complex pivot;
            double a, t;

            int[] Li, Ui, Lp = L.p, Up = U.p;
            Complex[] Lx, Ux;

            for (int k = 0; k < n; k++) // compute L(:,k) and U(:,k)
            {
                // Triangular solve
                Lp[k] = lnz; // L(:,k) starts here
                Up[k] = unz; // U(:,k) starts here
                if ((lnz + n > L.nzmax && !L.Resize(2 * L.nzmax + n)) ||
                    (unz + n > U.nzmax && !U.Resize(2 * U.nzmax + n)))
                {
                    return null;
                }
                Li = L.i;
                Lx = L.x;
                Ui = U.i;
                Ux = U.x;
                col = q != null ? (q[k]) : k;
                top = Triangular.SolveSp(L, A, col, xi, x, pinv, true);  // x = L\A(:,col)

                // Find pivot
                ipiv = -1;
                a = -1;
                for (p = top; p < n; p++)
                {
                    i = xi[p]; // x(i) is nonzero
                    if (pinv[i] < 0) // row i is not yet pivotal
                    {
                        if ((t = Complex.Abs(x[i])) > a)
                        {
                            a = t; // largest pivot candidate so far
                            ipiv = i;
                        }
                    }
                    else // x(i) is the entry U(pinv[i],k)
                    {
                        Ui[unz] = pinv[i];
                        Ux[unz++] = x[i];
                    }
                }
                if (ipiv == -1 || a <= 0)
                {
                    return null;
                }

                if (pinv[col] < 0 && Complex.Abs(x[col]) >= a * tol)
                {
                    ipiv = col;
                }

                // Divide by pivot
                pivot = x[ipiv]; // the chosen pivot
                Ui[unz] = k; // last entry in U(:,k) is U(k,k)
                Ux[unz++] = pivot;
                pinv[ipiv] = k; // ipiv is the kth pivot row
                Li[lnz] = ipiv; // first entry in L(:,k) is L(k,k) = 1
                Lx[lnz++] = 1;
                for (p = top; p < n; p++) // L(k+1:n,k) = x / pivot
                {
                    i = xi[p];
                    if (pinv[i] < 0) // x(i) is an entry in L(:,k)
                    {
                        Li[lnz] = i; // save unpermuted row in L
                        Lx[lnz++] = x[i] / pivot; // scale pivot column
                    }
                    x[i] = 0; // x [0..n-1] = 0 for next k
                }
            }

            // Finalize L and U
            Lp[n] = lnz;
            Up[n] = unz;
            Li = L.i; // fix row indices of L for final pinv
            for (p = 0; p < lnz; p++)
            {
                Li[p] = pinv[Li[p]];
            }
            L.Resize(0); // remove extra space from L and U
            U.Resize(0);

            return N; // success
        }

        /// <summary>
        /// Symbolic ordering and analysis for LU.
        /// </summary>
        /// <param name="order"></param>
        /// <param name="A"></param>
        /// <returns></returns>
        static SymbolicFactorization SymbolicAnalysis(int order, SparseMatrix A)
        { // ori: cs_sqr
            int n = A.n;

            SymbolicFactorization S = new SymbolicFactorization(); // allocate result S
            S.q = Ordering.AMD(order, A); // fill-reducing ordering

            if (order != 0 && S.q == null) return null;

            S.unz = 4 * (A.p[n]) + n; // For LU factorization only
            S.lnz = S.unz; // Guess nnz(L) and nnz(U)

            return S; // return result S
        }
    }
}
