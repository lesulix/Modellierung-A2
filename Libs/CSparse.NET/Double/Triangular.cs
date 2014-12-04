// -----------------------------------------------------------------------
// <copyright file="Triangular.cs" company="">
// Original CSparse code by Timothy A. Davis, http://www.suitesparse.com
// CSparse.NET code by Christian Woltering, http://csparse.codeplex.com/
// </copyright>
// -----------------------------------------------------------------------

namespace CSparse.Double
{
    /// <summary>
    /// Solving triangular systems.
    /// </summary>
    public static class Triangular
    {
        /// <summary>
        /// Solves a lower triangular system Lx=b where x and b are dense. x=b on
        /// input, solution on output.
        /// </summary>
        /// <param name="L">column-compressed, lower triangular matrix</param>
        /// <param name="x">size n, right hand side on input, solution on output</param>
        /// <returns>true if successful, false on error</returns>
        public static bool SolveL(SparseMatrix L, double[] x)
        {
            if (x == null) return false; // check inputs

            int n = L.n;
            int[] Lp = L.p;
            int[] Li = L.i;
            double[] Lx = L.x;

            int p;
            for (int j = 0; j < n; j++)
            {
                x[j] /= Lx[Lp[j]];
                for (p = Lp[j] + 1; p < Lp[j + 1]; p++)
                {
                    x[Li[p]] -= Lx[p] * x[j];
                }
            }
            return true;
        }

        /// <summary>
        /// Solves an upper triangular system L'x=b where x and b are dense. x=b on
        /// input, solution on output.
        /// </summary>
        /// <param name="L">column-compressed, lower triangular matrix</param>
        /// <param name="x">size n, right hand side on input, solution on output</param>
        /// <returns>true if successful, false on error</returns>
        public static bool SolveLt(SparseMatrix L, double[] x)
        {
            if (x == null) return false; // check inputs

            int n = L.n;
            int[] Lp = L.p;
            int[] Li = L.i;
            double[] Lx = L.x;

            int p;
            for (int j = n - 1; j >= 0; j--)
            {
                for (p = Lp[j] + 1; p < Lp[j + 1]; p++)
                {
                    x[j] -= Lx[p] * x[Li[p]];
                }
                x[j] /= Lx[Lp[j]];
            }
            return true;
        }

        /// <summary>
        /// Solves an upper triangular system Ux=b, where x and b are dense vectors.
        /// The diagonal of U must be the last entry of each column.
        /// </summary>
        /// <param name="U">upper triangular matrix in column-compressed form</param>
        /// <param name="x">size n, right hand side on input, solution on output</param>
        /// <returns>true if successful, false on error</returns>
        public static bool SolveU(SparseMatrix U, double[] x)
        {
            if (x == null) return false; // check inputs

            int n = U.n;
            int[] Up = U.p;
            int[] Ui = U.i;
            double[] Ux = U.x;

            int p;
            for (int j = n - 1; j >= 0; j--)
            {
                x[j] /= Ux[Up[j + 1] - 1];
                for (p = Up[j]; p < Up[j + 1] - 1; p++)
                {
                    x[Ui[p]] -= Ux[p] * x[j];
                }
            }
            return true;
        }

        /// <summary>
        /// Solves a lower triangular system U'x=b, where x and b are dense vectors.
        /// The diagonal of U must be the last entry of each column.
        ///</summary>
        /// <param name="U">upper triangular matrix in column-compressed form</param>
        /// <param name="x">size n, right hand side on input, solution on output</param>
        /// <returns>true if successful, false on error</returns>
        public static bool SolveUt(SparseMatrix U, double[] x)
        {
            if (x == null) return false; // check inputs

            int n = U.n;
            int[] Up = U.p;
            int[] Ui = U.i;
            double[] Ux = U.x;

            int p;
            for (int j = 0; j < n; j++)
            {
                for (p = Up[j]; p < Up[j + 1] - 1; p++)
                {
                    x[j] -= Ux[p] * x[Ui[p]];
                }
                x[j] /= Ux[Up[j + 1] - 1];
            }
            return true;
        }

        /// <summary>
        /// Solve Gx=b(:,k), where G is either upper (lo=false) or lower (lo=true)
        /// triangular.
        /// </summary>
        /// <param name="G">lower or upper triangular matrix in column-compressed form</param>
        /// <param name="B">right hand side, b=B(:,k)</param>
        /// <param name="k">use kth column of B as right hand side</param>
        /// <param name="xi">size 2*n, nonzero pattern of x in xi[top..n-1]</param>
        /// <param name="x">size n, x in x[xi[top..n-1]]</param>
        /// <param name="pinv">mapping of rows to columns of G, ignored if null</param>
        /// <param name="lo">true if lower triangular, false if upper</param>
        /// <returns>top, -1 in error</returns>
        public static int SolveSp(SparseMatrix G, SparseMatrix B, int k, int[] xi, double[] x, int[] pinv, bool lo)
        {
            if (xi == null || x == null) return (-1);

            int[] Gp = G.p;
            int[] Gi = G.i;
            double[] Gx = G.x;
            int n = G.n;

            int[] Bp = B.p;
            int[] Bi = B.i;
            double[] Bx = B.x;

            int top = Common.Reach(G, B, k, xi, pinv); // xi[top..n-1]=Reach(B(:,k))

            int j, J, p, q, px;

            for (p = top; p < n; p++)
            {
                x[xi[p]] = 0; // clear x
            }

            for (p = Bp[k]; p < Bp[k + 1]; p++)
            {
                x[Bi[p]] = Bx[p]; // scatter B
            }

            for (px = top; px < n; px++)
            {
                j = xi[px]; // x(j) is nonzero
                J = pinv != null ? (pinv[j]) : j; // j maps to col J of G
                if (J < 0) continue; // column J is empty
                x[j] /= Gx[lo ? (Gp[J]) : (Gp[J + 1] - 1)]; // x(j) /= G(j,j)
                p = lo ? (Gp[J] + 1) : (Gp[J]); // lo: L(j,j) 1st entry
                q = lo ? (Gp[J + 1]) : (Gp[J + 1] - 1); // up: U(j,j) last entry
                for (; p < q; p++)
                {
                    x[Gi[p]] -= Gx[p] * x[j]; // x(i) -= G(i,j) * x(j)
                }
            }
            return (top); // return top of stack
        }
    }
}
