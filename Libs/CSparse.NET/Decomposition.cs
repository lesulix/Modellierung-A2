// -----------------------------------------------------------------------
// <copyright file="Decomposition.cs" company="">
// Original CSparse code by Timothy A. Davis, http://www.suitesparse.com
// CSparse.NET code by Christian Woltering, http://csparse.codeplex.com/
// </copyright>
// -----------------------------------------------------------------------

namespace CSparse
{
    /// <summary>
    /// Dulmage-Mendelsohn decomposition.
    /// </summary>
    public class Decomposition
    {
        public int[] p;   // size m, row permutation
        public int[] q;   // size n, column permutation
        public int[] r;   // size nb+1, block k is rows r[k] to r[k+1]-1 in A(p,q)
        public int[] s;   // size nb+1, block k is cols s[k] to s[k+1]-1 in A(p,q)
        public int nb;    // # of blocks in fine dmperm decomposition
        public int[] rr;  // coarse row decomposition
        public int[] cc;  // coarse column decomposition

        /// <summary>
        /// Create a new Decomposition instance. 
        /// </summary>
        public Decomposition(int m, int n)
        {
            this.p = new int[m];
            this.r = new int[m + 6];
            this.q = new int[n];
            this.s = new int[n + 6];

            this.rr = new int[5];  // coarse row decomposition
            this.cc = new int[5];  // coarse column decomposition
        }

        public int Blocks()
        {
            return nb;
        }

        public int StructuralRank()
        {
            return rr[3];
        }

        public int Singletons()
        {
            int k, ns = 0;
            for (k = 0; k < nb; k++)
            {
                if ((r[k + 1] == r[k] + 1) && (s[k + 1] == s[k] + 1))
                {
                    ns++;
                }
            }
            return ns;
        }
    }
}
