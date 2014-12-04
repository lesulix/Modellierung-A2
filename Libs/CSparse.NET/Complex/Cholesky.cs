// -----------------------------------------------------------------------
// <copyright file="Cholesky.cs" company="">
// Original CSparse code by Timothy A. Davis, http://www.suitesparse.com
// CSparse.NET code by Christian Woltering, http://csparse.codeplex.com/
// </copyright>
// -----------------------------------------------------------------------

namespace CSparse.Complex
{
    using System;
    using System.Numerics;

    /// <summary>
    /// Cholesky decomposition.
    /// </summary>
    public class Cholesky
    {
        SymbolicFactorization symFactor;
        NumericFactorization numFactor;
        int n;

        public NumericFactorization N
        {
            get { return numFactor; }
        }

        public SymbolicFactorization S
        {
            get { return symFactor; }
        }

        /// <summary>
        /// Creates a Cholesky factorization.
        /// </summary>
        /// <param name="order">ordering method to use (0 or 1)</param>
        /// <param name="A">column-compressed matrix, symmetric positive definite, only
        /// upper triangular part is used</param>
        /// <returns>Cholesky factorization</returns>
        public static Cholesky Create(int order, SparseMatrix A)
        {
            var chol = new Cholesky();

            int n = chol.n = A.n;

            chol.symFactor = SymbolicAnalysis(order, A); // ordering and symbolic analysis
            chol.numFactor = Factorize(A, chol.symFactor); // numeric Cholesky factorization

            return chol;
        }

        /// <summary>
        /// Solves Ax=b where A is symmetric positive definite; b is overwritten with
        /// solution.
        /// </summary>
        /// <param name="b">right hand side, b is overwritten with solution</param>
        /// <returns>true if successful, false on error</returns>
        public bool Solve(Complex[] b)
        {
            if (b == null || numFactor == null) return false; // check inputs

            Complex[] x = new Complex[n];

            Common.InversePermute(symFactor.pinv, b, x, n); // x = P*b
            Triangular.SolveL(numFactor.L, x); // x = L\x
            Triangular.SolveLt(numFactor.L, x); // x = L'\x
            Common.Permute(symFactor.pinv, x, b, n); // b = P'*x

            return true;
        }

        /// <summary>
        /// Compute the Numeric Cholesky factorization, L = chol (A, [pinv parent cp]).
        /// </summary>
        /// <returns>Numeric Cholesky factorization</returns>
        public static NumericFactorization Factorize(SparseMatrix A, SymbolicFactorization S)
        {
            Complex d, lki;
            Complex[] Lx, x, Cx;
            int top, i, p, k;
            int[] Li, Lp, cp, pinv, s, c, parent, Cp, Ci;

            SparseMatrix L, C;
            NumericFactorization N = new NumericFactorization(); // allocate result

            int n = A.n;
            c = new int[n]; // get int workspace
            s = new int[n];
            x = new Complex[n]; // get double workspace

            cp = S.cp;
            pinv = S.pinv;
            parent = S.parent;

            C = pinv != null ? A.PermuteSym(pinv, true) : A;
            Cp = C.p;
            Ci = C.i;
            Cx = C.x;

            N.L = L = new SparseMatrix(n, n, cp[n], true); // allocate result
            Lp = L.p;
            Li = L.i;
            Lx = L.x;

            for (k = 0; k < n; k++)
            {
                Lp[k] = c[k] = cp[k];
            }
            for (k = 0; k < n; k++) // compute L(k,:) for L*L' = C
            {
                // Nonzero pattern of L(k,:)
                top = Common.EReach(C, k, parent, s, c); // find pattern of L(k,:)
                x[k] = 0;                           // x (0:k) is now zero
                for (p = Cp[k]; p < Cp[k + 1]; p++) // x = full(triu(C(:,k)))
                {
                    if (Ci[p] <= k) x[Ci[p]] = Cx[p];
                }
                d = x[k]; // d = C(k,k)
                x[k] = 0; // clear x for k+1st iteration

                // Triangular solve
                for (; top < n; top++) // solve L(0:k-1,0:k-1) * x = C(:,k)
                {
                    i = s[top];  // s [top..n-1] is pattern of L(k,:)
                    lki = x[i] / Lx[Lp[i]]; // L(k,i) = x (i) / L(i,i)
                    x[i] = 0;               // clear x for k+1st iteration
                    for (p = Lp[i] + 1; p < c[i]; p++)
                    {
                        x[Li[p]] -= Lx[p] * lki;
                    }
                    d -= lki * Complex.Conjugate(lki); // d = d - L(k,i)*L(k,i)
                    p = c[i]++;
                    Li[p] = k; // store L(k,i) in column i
                    Lx[p] = Complex.Conjugate(lki);
                }
                // Compute L(k,k)
                if (d.Real <= 0 || d.Imaginary != 0)
                {
                    return null; // not pos def
                }

                p = c[k]++;
                Li[p] = k; // store L(k,k) = sqrt (d) in column k
                Lx[p] = Complex.Sqrt(d);
            }
            Lp[n] = cp[n]; // finalize L

            return N; // success
        }

        // Sparse Cholesky update, L*L' + w*w'
        public bool Update(SparseMatrix w)
        {
            return UpDown(numFactor.L, 1, w, symFactor.parent);
        }

        // Sparse Cholesky downdate, L*L' - w*w'
        public bool Downdate(SparseMatrix w)
        {
            return UpDown(numFactor.L, -1, w, symFactor.parent);
        }

        // Sparse Cholesky update/downdate, L*L' + sigma*w*w' (sigma = +1 or -1)
        private bool UpDown(SparseMatrix L, int sigma, SparseMatrix C, int[] parent)
        {
            int n, p, f, j;
            int[] Lp, Li, Cp, Ci;
            Complex[] Lx, Cx, w;
            Complex alpha, gamma, w1, w2, phase;
            double beta = 1, beta2 = 1, delta;

            if (parent == null)
            {
                return false;  // check inputs
            }

            Lp = L.p; Li = L.i; Lx = L.x; n = L.n;
            Cp = C.p; Ci = C.i; Cx = C.x;

            if ((p = Cp[0]) >= Cp[1])
            {
                return true; // return if C empty
            }

            w = new Complex[n]; // get workspace

            f = Ci[p];
            for (; p < Cp[1]; p++)
            {
                f = Math.Min(f, Ci[p]); // f = min (find (C))
            }

            for (p = Cp[0]; p < Cp[1]; p++)
            {
                w[Ci[p]] = Cx[p]; // w = C
            }

            for (j = f; j != -1; j = parent[j]) // walk path f up to root
            {
                p = Lp[j];
                alpha = w[j] / Lx[p]; // alpha = w(j) / L(j,j)
                beta2 = beta * beta + sigma * (alpha * Complex.Conjugate(alpha)).Real;

                if (beta2 <= 0) break; // not positive definite

                beta2 = Math.Sqrt(beta2);
                delta = (sigma > 0) ? (beta / beta2) : (beta2 / beta);
                gamma = sigma * Complex.Conjugate(alpha) / (beta2 * beta);
                Lx[p] = delta * Lx[p] + ((sigma > 0) ? (gamma * w[j]) : 0);
                beta = beta2;

                phase = Complex.Abs(Lx[p]) / Lx[p];  // phase = abs(L(j,j)) / L(j,j)
                Lx[p] *= phase; // L(j,j) = L(j,j) * phase

                for (p++; p < Lp[j + 1]; p++)
                {
                    w1 = w[Li[p]];
                    w[Li[p]] = w2 = w1 - alpha * Lx[p];
                    Lx[p] = delta * Lx[p] + gamma * ((sigma > 0) ? w1 : w2);
                    Lx[p] *= phase; // L(i,j) = L(i,j) * phase
                }
            }

            return (beta2 > 0);
        }

        /// <summary>
        /// Ordering and symbolic analysis for a Cholesky factorization
        /// </summary>
        /// <param name="order"></param>
        /// <param name="A"></param>
        /// <returns></returns>
        static SymbolicFactorization SymbolicAnalysis(int order, SparseMatrix A)
        {
            int n = A.n;
            int[] c, post, P;

            SymbolicFactorization S = new SymbolicFactorization(); // allocate result S

            P = Ordering.AMD(order, A); // P = amd(A+A'), or natural
            S.pinv = Common.InversePermutation(P, n); // find inverse permutation

            if (order != 0 && S.pinv == null)
            {
                return null;
            }

            SparseMatrix C = A.PermuteSym(S.pinv, false); // C = spones(triu(A(P,P)))

            S.parent = Common.EliminationTree(C, false); // find etree of C
            post = Common.TreePostorder(S.parent, n); // postorder the etree
            c = Common.ColumnCounts(C, S.parent, post, false); // find column counts of chol(C)

            S.cp = new int[n + 1]; // allocate result S.cp
            S.unz = S.lnz = Common.CumulativeSum(S.cp, c, n); // find column pointers for L

            return S;
        }
    }
}
