// -----------------------------------------------------------------------
// <copyright file="Matrix.cs" company="">
// Original CSparse code by Timothy A. Davis, http://www.suitesparse.com
// CSparse.NET code by Christian Woltering, http://csparse.codeplex.com/
// </copyright>
// -----------------------------------------------------------------------

namespace CSparse
{
    using System;

    /// <summary>
    /// A symbolic matrix class.
    /// </summary>
    public class Matrix
    {
        public int nzmax; // maximum number of entries
        public int m;     // number of rows
        public int n;     // number of columns
        public int[] p;   // column pointers (size n+1) or col indices (size nzmax)
        public int[] i;   // row indices, size nzmax

        protected Matrix(int m, int n, int nzmax)
        {
            this.m = m;
            this.n = n;
            this.nzmax = (nzmax = Math.Max(nzmax, 1));
            this.p = new int[n + 1];
            this.i = new int[nzmax];
        }

        /// <summary>
        /// Drops entries from a sparse matrix (for Dulmage-Mendelsohn only).
        /// </summary>
        internal int KeepSymbolic(Func<int, int, int[], bool> fkeep, int[] other)
        {
            int j, pp, nzz = 0;
            if (fkeep == null) return (-1); // check inputs

            for (j = 0; j < n; j++)
            {
                pp = p[j]; // get current location of col j
                p[j] = nzz; // record new location of col j
                for (; pp < p[j + 1]; pp++)
                {
                    if (fkeep(i[pp], j, other))
                    {
                        i[nzz++] = i[pp];
                    }
                }
            }
            p[n] = nzz; // finalize A
            this.Resize(0); // remove extra space from A
            return (nzz);
        }

        public virtual bool Resize(int nzmax)
        {
            throw new NotImplementedException();
        }

        public virtual Matrix Permute(int[] pinv, int[] q, bool values)
        {
            throw new NotImplementedException();
        }

        public virtual Matrix Transpose(bool values)
        {
            throw new NotImplementedException();
        }
    }
}
