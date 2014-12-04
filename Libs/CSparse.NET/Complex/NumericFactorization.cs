// -----------------------------------------------------------------------
// <copyright file="NumericFactorization.cs" company="">
// Original CSparse code by Timothy A. Davis, http://www.suitesparse.com
// CSparse.NET code by Christian Woltering, http://csparse.codeplex.com/
// </copyright>
// -----------------------------------------------------------------------

namespace CSparse.Complex
{
    /// <summary>
    /// Numeric Cholesky, LU, or QR factorization.
    /// </summary>
    public class NumericFactorization
    {
        public SparseMatrix L; // L for LU and Cholesky, V for QR
        public SparseMatrix U; // U for LU, R for QR, not used for Cholesky
        public int[] pinv;     // partial pivoting for LU
        public double[] B;     // beta [0..n-1] for QR
    }
}
