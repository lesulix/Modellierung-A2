// -----------------------------------------------------------------------
// <copyright file="TripletMatrix.cs" company="">
// Original CSparse code by Timothy A. Davis, http://www.suitesparse.com
// CSparse.NET code by Christian Woltering, http://csparse.codeplex.com/
// </copyright>
// -----------------------------------------------------------------------

namespace CSparse.Complex
{
    using System;
    using System.Numerics;

    /// <summary>
    /// Matrix in triplet format.
    /// </summary>
    public class TripletMatrix
    {
        public int nzmax;     // maximum number of entries
        public int m;         // number of rows
        public int n;         // number of columns
        public int[] p;       // column pointers (size n+1) or col indices (size nzmax)
        public int[] i;       // row indices, size nzmax
        public Complex[] x;    // numerical values, size nzmax
        public int nz;        // # of entries in triplet matrix, -1 for compressed-col

        public TripletMatrix(int m, int n, int nzmax, bool values)
        {
            this.m = m; // define dimensions and nzmax
            this.n = n;
            this.nzmax = (nzmax = Math.Max(nzmax, 1));
            this.nz = 0;
            this.p = new int[nzmax];
            this.i = new int[nzmax];
            this.x = values ? new Complex[nzmax] : null;
        }

        /// <summary>
        /// Change the max # of entries sparse matrix
        /// </summary>
        /// <param name="nzmax"></param>
        /// <returns></returns>
        public bool Resize(int nzmax)
        {
            if (nzmax <= 0)
            {
                nzmax = this.nz;
            }

            Array.Resize<int>(ref this.i, nzmax);
            Array.Resize<int>(ref this.p, nzmax);

            if (this.x != null)
            {
                Array.Resize<Complex>(ref this.x, nzmax);
            }

            this.nzmax = nzmax;

            return true;
        }

        /// <summary>
        /// Adds an entry to a triplet matrix. Memory-space and dimension of T are
        /// increased if necessary.
        /// </summary>
        /// <param name="ii">row index of new entry</param>
        /// <param name="jj">column index of new entry</param>
        /// <param name="xx">numerical value of new entry</param>
        /// <returns>true if successful, false otherwise</returns>
        public bool Entry(int ii, int jj, Complex xx)
        {
            if (ii < 0 || jj < 0) return false; // check inputs
            if (nz >= nzmax && !this.Resize(2 * nzmax)) return false;
            if (x != null) x[nz] = xx;
            i[nz] = ii;
            p[nz++] = jj;
            m = Math.Max(m, ii + 1);
            n = Math.Max(n, jj + 1);
            return true;
        }

        /// <summary>
        /// C = compressed-column form of a triplet matrix T. The columns of C are
        /// not sorted, and duplicate entries may be present in C.
        /// </summary>
        /// <returns>C if successful, null on error</returns>
        public SparseMatrix Compress()
        {
            int pp, k;

            SparseMatrix C = new SparseMatrix(m, n, nz, x != null); // allocate result
            int[] w = new int[n]; // get workspace

            int[] Cp = C.p;
            int[] Ci = C.i;
            Complex[] Cx = C.x;

            for (k = 0; k < nz; k++)
            {
                w[p[k]]++; // column counts
            }

            Common.CumulativeSum(Cp, w, n); // column pointers

            for (k = 0; k < nz; k++)
            {
                Ci[pp = w[p[k]]++] = i[k]; // A(i,j) is the pth entry in C
                if (Cx != null) Cx[pp] = x[k];
            }
            return C; // success; free w and return C
        }
    }
}
