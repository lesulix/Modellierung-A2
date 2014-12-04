// -----------------------------------------------------------------------
// <copyright file="SparseMatrix.cs" company="">
// Original CSparse code by Timothy A. Davis, http://www.suitesparse.com
// CSparse.NET code by Christian Woltering, http://csparse.codeplex.com/
// </copyright>
// -----------------------------------------------------------------------

namespace CSparse.Double
{
    using System;

    /// <summary>
    /// Matrix in compressed-column form.
    /// </summary>
    public class SparseMatrix : Matrix
    {
        public double[] x;    // numerical values, size nzmax

        public SparseMatrix(int m, int n, int nzmax, bool values)
            : base (m, n, nzmax)
        {
            this.x = values ? new double[nzmax] : null;
        }

        #region Public functions

        /// <summary>
        /// Change the max # of entries sparse matrix
        /// </summary>
        /// <param name="nzmax"></param>
        /// <returns></returns>
        public override bool Resize(int nzmax)
        {
            if (nzmax <= 0)
            {
                nzmax = this.p[this.n];
            }

            Array.Resize<int>(ref this.i, nzmax);

            if (this.x != null)
            {
                Array.Resize<double>(ref this.x, nzmax);
            }

            this.nzmax = nzmax;

            return true;
        }

        /// <summary>
        /// Sparse matrix times dense column vector, y = A*x+y.
        /// </summary>
        /// <param name="xx">size n, vector x</param>
        /// <param name="yy">size m, vector y</param>
        /// <returns>true if successful, false on error</returns>
        public bool Axpy(double[] xx, double[] yy)
        {
            if (xx == null || yy == null) return false; // check inputs

            for (int j = 0; j < n; j++)
            {
                for (int k = p[j]; k < p[j + 1]; k++)
                {
                    yy[i[k]] += x[k] * xx[j];
                }
            }
            return true;
        }

        /// <summary>
        /// Sparse matrix times dense column vector, y = A*x.
        /// </summary>
        /// <param name="xx">size n, vector x</param>
        /// <param name="yy">size m, vector y</param>
        /// <returns>true if successful, false on error</returns>
        public bool Ax(double[] xx, out double[] yy)
        {
            yy = new double[m];
            if (xx == null) return false; // check inputs            

            for (int j = 0; j < n; j++)
            {
                for (int k = p[j]; k < p[j + 1]; k++)
                {
                    yy[i[k]] += x[k] * xx[j];
                }
            }
            return true;
        }

        /// <summary>
        /// Computes the transpose of a sparse matrix, C =A';
        /// </summary>
        /// <param name="values">pattern only if false, both pattern and values otherwise</param>
        /// <returns>C=A', null on error</returns>
        public override Matrix Transpose(bool values = true)
        {
            int pp, q, j;

            SparseMatrix C = new SparseMatrix(n, m, p[n], (values && x != null)); // allocate result

            int[] w = new int[m];
            int[] Cp = C.p;
            int[] Ci = C.i;
            double[] Cx = C.x;

            for (pp = 0; pp < p[n]; pp++)
            {
                w[i[pp]]++; // row counts
            }

            Common.CumulativeSum(Cp, w, m); // row pointers
            for (j = 0; j < n; j++)
            {
                for (pp = p[j]; pp < p[j + 1]; pp++)
                {
                    Ci[q = w[i[pp]]++] = j; // place A(i,j) as entry C(j,i)
                    if (Cx != null)
                    {
                        Cx[q] = x[pp];
                    }
                }
            }
            return C; // success; free w and return C
        }

        /// <summary>
        /// Removes and sums duplicate entries in a sparse matrix.
        /// </summary>
        /// <returns>true if successful, false on error</returns>
        public bool Cleanup()
        {
            int ii, jj, pp, q, nzz = 0;
            int[] w = new int[m]; // get workspace

            for (ii = 0; ii < m; ii++) w[ii] = -1; // row i not yet seen
            for (jj = 0; jj < n; jj++)
            {
                q = nzz; // column j will start at q
                for (pp = p[jj]; pp < p[jj + 1]; pp++)
                {
                    ii = i[pp]; // A(i,j) is nonzero
                    if (w[ii] >= q)
                    {
                        x[w[ii]] += x[pp]; // A(i,j) is a duplicate
                    }
                    else
                    {
                        w[ii] = nzz; // record where row i occurs
                        i[nzz] = ii; // keep A(i,j)
                        x[nzz++] = x[pp];
                    }
                }
                p[jj] = q; // record start of column j
            }
            p[n] = nzz; // finalize A
            //cs_free (w) ; // free workspace

            return this.Resize(0); // remove extra space from A
        }

        /// <summary>
        /// Removes numerically zero entries from a matrix.
        /// </summary>
        /// <returns>nz, new number of entries in A, -1 on error</returns>
        public int DropZeros()
        {
            Func<int, int, double, object, bool> cs_nonzero = (int i, int j, double aij, object other) =>
            {
                return (aij != 0);
            };

            return (Keep(cs_nonzero, null)); // keep all nonzero entries
        }

        /// <summary>
        /// Removes entries from a matrix with absolute value &lt;= tol.
        /// </summary>
        /// <param name="tol">drop tolerance</param>
        /// <returns>nz, new number of entries in A, -1 on error</returns>
        public int DropByTolerance(double tol)
        {
            Func<int, int, double, object, bool> cs_tol = (int i, int j, double aij, object ftol) =>
            {
                return (Math.Abs(aij) > (double)ftol);
            };

            return (Keep(cs_tol, tol)); // keep all large entries
        }

        /// <summary>
        /// Scatters and sums a sparse vector A(:,j) into a dense vector, x = x + beta * A(:,j).
        /// </summary>
        /// <param name="j">the column of A to use</param>
        /// <param name="beta">scalar multiplied by A(:,j)</param>
        /// <param name="w">size m, node i is marked if w[i] = mark</param>
        /// <param name="xx">size m, ignored if null</param>
        /// <param name="mark">mark value of w</param>
        /// <param name="C">pattern of x accumulated in C.i</param>
        /// <param name="nz">pattern of x placed in C starting at C.i[nz]</param>
        /// <returns>new value of nz, -1 on error</returns>
        public int Scatter(int j, double beta, int[] w, double[] xx, int mark, SparseMatrix C, int nzz)
        {
            int ii, pp;
            int[] Ci;
            if (w == null || C == null) return (-1); // check inputs
            Ci = C.i;
            for (pp = p[j]; pp < p[j + 1]; pp++)
            {
                ii = i[pp]; // A(i,j) is nonzero
                if (w[ii] < mark)
                {
                    w[ii] = mark; // i is new entry in column j
                    Ci[nzz++] = ii; // add i to pattern of C(:,j)
                    if (xx != null) xx[ii] = beta * x[pp]; // x(i) = beta*A(i,j)
                }
                else if (xx != null) xx[ii] += beta * x[pp]; // i exists in C(:,j) already
            }
            return (nzz);
        }

        /// <summary>
        /// Computes the 1-norm of a sparse matrix = max (sum (abs (A))), largest
        /// column sum.
        /// </summary>
        /// <returns>the 1-norm if successful, -1 on error</returns>
        public double Norm()
        {
            int pp, j;
            double norm = 0, s;
            if (x == null) return (-1); // check inputs

            for (j = 0; j < n; j++)
            {
                for (s = 0, pp = p[j]; pp < p[j + 1]; pp++) s += Math.Abs(x[pp]);
                norm = Math.Max(norm, s);
            }
            return (norm);
        }

        /// <summary>
        /// Permutes a sparse matrix, C = PAQ.
        /// </summary>
        /// <param name="A">m-by-n, column-compressed matrix</param>
        /// <param name="pinv">a permutation vector of length m</param>
        /// <param name="q">a permutation vector of length n</param>
        /// <param name="values">allocate pattern only if false, values and pattern otherwise</param>
        /// <returns>C = PAQ, null on error</returns>
        public override Matrix Permute(int[] pinv, int[] q, bool values = true)
        {
            int t, j, k, nz = 0, m, n;
            int[] Ap, Ai, Cp, Ci;
            double[] Cx, Ax;
            SparseMatrix C;

            m = this.m;
            n = this.n;
            Ap = this.p;
            Ai = this.i;
            Ax = this.x;

            C = new SparseMatrix(m, n, Ap[n], (values && Ax != null)); // alloc result
            if (C == null) return C; // out of memory
            Cp = C.p; Ci = C.i; Cx = C.x;
            for (k = 0; k < n; k++)
            {
                Cp[k] = nz; // column k of C is column q[k] of A
                j = q != null ? (q[k]) : k;
                for (t = Ap[j]; t < Ap[j + 1]; t++)
                {
                    if (Cx != null) Cx[nz] = Ax[t]; // row i of A is row pinv[i] of C
                    Ci[nz++] = pinv != null ? (pinv[Ai[t]]) : Ai[t];
                }
            }
            Cp[n] = nz; // finalize the last column of C
            return C;
        }

        /// <summary>
        /// Permutes a symmetric sparse matrix. C = PAP' where A and C are symmetric.
        /// </summary>
        /// <param name="A">column-compressed matrix (only upper triangular part is used)</param>
        /// <param name="pinv">size n, inverse permutation</param>
        /// <param name="values">allocate pattern only if false, values and pattern otherwise</param>
        /// <returns>C = PAP', null on error</returns>
        public SparseMatrix PermuteSym(int[] pinv, bool values)
        {
            int i, j, p, q, i2, j2;

            int n = this.n;
            int[] Ap = this.p;
            int[] Ai = this.i;
            double[] Ax = this.x;

            SparseMatrix C = new SparseMatrix(n, n, Ap[n], values && (Ax != null)); // alloc result
            int[] Cp = C.p;
            int[] Ci = C.i;
            double[] Cx = C.x;

            int[] w = new int[n]; // get workspace

            for (j = 0; j < n; j++) // count entries in each column of C
            {
                j2 = pinv != null ? pinv[j] : j; // column j of A is column j2 of C
                for (p = Ap[j]; p < Ap[j + 1]; p++)
                {
                    i = Ai[p];
                    if (i > j) continue; // skip lower triangular part of A
                    i2 = pinv != null ? pinv[i] : i; // row i of A is row i2 of C
                    w[Math.Max(i2, j2)]++; // column count of C
                }
            }
            Common.CumulativeSum(Cp, w, n); // compute column pointers of C
            for (j = 0; j < n; j++)
            {
                j2 = pinv != null ? pinv[j] : j; // column j of A is column j2 of C
                for (p = Ap[j]; p < Ap[j + 1]; p++)
                {
                    i = Ai[p];
                    if (i > j) continue; // skip lower triangular part of A
                    i2 = pinv != null ? pinv[i] : i; // row i of A is row i2 of C
                    Ci[q = w[Math.Max(i2, j2)]++] = Math.Min(i2, j2);
                    if (Cx != null) Cx[q] = Ax[p];
                }
            }
            return C; // success; free workspace, return C
        }

        /// <summary>
        /// Drops entries from a sparse matrix
        /// </summary>
        /// <param name="fkeep">drop aij if fkeep.fkeep(i,j,aij,other) is false</param>
        /// <param name="other">optional parameter to fkeep</param>
        /// <returns>nz, new number of entries in A, -1 on error</returns>
        public int Keep(Func<int, int, double, object, bool> fkeep, object other)
        {
            int j, pp, nzz = 0;
            if (fkeep == null) return (-1); // check inputs

            for (j = 0; j < n; j++)
            {
                pp = p[j]; // get current location of col j
                p[j] = nzz; // record new location of col j
                for (; pp < p[j + 1]; pp++)
                {
                    if (fkeep(i[pp], j, x != null ? x[pp] : 1, other))
                    {
                        if (x != null) x[nzz] = x[pp]; // keep A(i,j)
                        i[nzz++] = i[pp];
                    }
                }
            }
            p[n] = nzz; // finalize A
            this.Resize(0); // remove extra space from A
            return (nzz);
        }

        #endregion

        /// <summary>
        /// C = alpha*A + beta*B
        /// </summary>
        /// <param name="A">column-compressed matrix</param>
        /// <param name="B">column-compressed matrix</param>
        /// <param name="alpha">scalar alpha</param>
        /// <param name="beta">scalar beta</param>
        /// <returns>C=alpha*A + beta*B, null on error</returns>
        public static SparseMatrix Add(SparseMatrix A, SparseMatrix B, double alpha = 1.0, double beta = 1.0)
        {
            int p, j, nz = 0, anz, m, n, bnz;
            int[] Cp, Ci, Bp, w;
            bool values;
            double[] x, Bx, Cx;
            
            // check inputs
            if (A == null || B == null) return null;
            if (A.m != B.m || A.n != B.n) return null;

            m = A.m;
            anz = A.p[A.n];
            n = B.n;
            Bp = B.p;
            Bx = B.x;
            bnz = Bp[n];

            w = new int[m]; // get workspace
            values = (A.x != null) && (Bx != null);
            x = values ? new double[m] : null; // get workspace

            SparseMatrix C = new SparseMatrix(m, n, anz + bnz, values); // allocate result

            Cp = C.p; Ci = C.i; Cx = C.x;
            for (j = 0; j < n; j++)
            {
                Cp[j] = nz; // column j of C starts here
                nz = A.Scatter(j, alpha, w, x, j + 1, C, nz); // alpha*A(:,j)
                nz = B.Scatter(j, beta, w, x, j + 1, C, nz); // beta*B(:,j)
                if (values)
                {
                    for (p = Cp[j]; p < nz; p++)
                    {
                        Cx[p] = x[Ci[p]];
                    }
                }
            }
            Cp[n] = nz; // finalize the last column of C
            C.Resize(0); // remove extra space from C

            return C; // success
        }

        /// <summary>
        /// Sparse matrix multiplication, C = A*B
        /// </summary>
        /// <param name="A">column-compressed matrix</param>
        /// <param name="B">column-compressed matrix</param>
        /// <returns>C = A*B, null on error</returns>
        public static SparseMatrix Multiply(SparseMatrix A, SparseMatrix B)
        {
            int p, j, nz = 0, anz, m, n, bnz;
            bool values;
            int[] Cp, Ci, Bp, w, Bi;
            double[] x, Bx, Cx;
            
            // check inputs
            if (A == null || B == null) return null;
            if (A.n != B.m) return null;

            m = A.m;
            anz = A.p[A.n];
            n = B.n;
            Bp = B.p;
            Bi = B.i;
            Bx = B.x;
            bnz = Bp[n];

            w = new int[m]; // get workspace
            values = (A.x != null) && (Bx != null);
            x = values ? new double[m] : null; // get workspace

            SparseMatrix C = new SparseMatrix(m, n, anz + bnz, values); // allocate result

            Cp = C.p;
            for (j = 0; j < n; j++)
            {
                if (nz + m > C.nzmax && !C.Resize(2 * (C.nzmax) + m))
                {
                    return null; // out of memory
                }
                Ci = C.i; Cx = C.x; // C.i and C.x may be reallocated
                Cp[j] = nz; // column j of C starts here
                for (p = Bp[j]; p < Bp[j + 1]; p++)
                {
                    nz = A.Scatter(Bi[p], Bx != null ? Bx[p] : 1, w, x, j + 1, C, nz);
                }
                if (values)
                {
                    for (p = Cp[j]; p < nz; p++)
                    {
                        Cx[p] = x[Ci[p]];
                    }
                }
            }
            Cp[n] = nz; // finalize the last column of C
            C.Resize(0); // remove extra space from C

            return C; // success
        }
    }
}
