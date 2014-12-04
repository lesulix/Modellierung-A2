// -----------------------------------------------------------------------
// <copyright file="QR.cs" company="">
// Original CSparse code by Timothy A. Davis, http://www.suitesparse.com
// CSparse.NET code by Christian Woltering, http://csparse.codeplex.com/
// </copyright>
// -----------------------------------------------------------------------

namespace CSparse.Double
{
    using System;

    /// <summary>
    /// QR decomposition.
    /// </summary>
    public class QR
    {
        SymbolicFactorization symFactor;
        NumericFactorization numFactor;
        int n, m;

        /// <summary>
        /// Creates a QR factorization.
        /// </summary>
        /// <param name="A">column-compressed matrix</param>
        /// <param name="order">ordering method to use (0 to 3)</param>
        /// <returns>QR factorization</returns>
        public static QR Create(SparseMatrix A, int order = 3)
        {
            var qr = new QR();

            qr.n = A.n;
            qr.m = A.m;

            if (qr.m >= qr.n)
            {
                qr.symFactor = SymbolicAnalysis(order, A); // ordering and symbolic analysis
                qr.numFactor = Factorize(A, qr.symFactor); // numeric QR factorization
            }
            else
            {
                SparseMatrix AT = (SparseMatrix)A.Transpose(true); // Ax=b is underdetermined
                qr.symFactor = SymbolicAnalysis(order, AT); // ordering and symbolic analysis
                qr.numFactor = Factorize(AT, qr.symFactor); // numeric QR factorization of A'
            }

            return qr;
        }

        /// <summary>
        /// Solve a least-squares problem (min ||Ax-b||_2, where A is m-by-n with m
        /// >= n) or underdetermined system (Ax=b, where m &lt; n)
        /// </summary>
        /// <param name="b">size max(m,n), b (size m) on input, x(size n) on output</param>
        /// <returns>true if successful, false on error</returns>
        public bool Solve(double[] b)
        {
            double[] x;

            if (b == null)
            {
                return false; // check inputs
            }

            if (m >= n)
            {
                x = new double[symFactor != null ? symFactor.m2 : 1]; // get workspace

                Common.InversePermute(symFactor.pinv, b, x, m); // x(0:m-1) = b(p(0:m-1)
                for (int k = 0; k < n; k++) // apply Householder refl. to x
                {
                    ApplyHouseholder(numFactor.L, k, numFactor.B[k], x);
                }
                Triangular.SolveU(numFactor.U, x); // x = R\x
                Common.InversePermute(symFactor.q, x, b, n); // b(q(0:n-1)) = x(0:n-1)
            }
            else
            {
                x = new double[symFactor != null ? symFactor.m2 : 1]; // get workspace

                Common.Permute(symFactor.q, b, x, m); // x(q(0:m-1)) = b(0:m-1)
                Triangular.SolveUt(numFactor.U, x); // x = R'\x
                for (int k = m - 1; k >= 0; k--) // apply Householder refl. to x
                {
                    ApplyHouseholder(numFactor.L, k, numFactor.B[k], x);
                }
                Common.Permute(symFactor.pinv, x, b, n); // b(0:n-1) = x(p(0:n-1))
            }

            return true;
        }

        /// <summary>
        /// Sparse QR factorization [V,beta,pinv,R] = qr (A)
        /// </summary>
        static NumericFactorization Factorize(SparseMatrix A, SymbolicFactorization S)
        {
            double[] Beta;
            int i, p, p1, top, len, col;

            int m = A.m;
            int n = A.n;
            int[] Ap = A.p;
            int[] Ai = A.i;
            double[] Ax = A.x;

            int[] q = S.q;
            int[] parent = S.parent;
            int[] pinv = S.pinv;
            int m2 = S.m2;

            int vnz = S.lnz;
            int rnz = S.unz;

            int[] leftmost = S.leftmost;

            int[] w = new int[m2 + n]; // get int workspace
            double[] x = new double[m2]; // get double workspace
           
            int s = m2; // offset into w

            SparseMatrix R, V;
            NumericFactorization N = new NumericFactorization(); // allocate result

            N.L = V = new SparseMatrix(m2, n, vnz, true); // allocate result V
            N.U = R = new SparseMatrix(m2, n, rnz, true); // allocate result R
            N.B = Beta = new double[n]; // allocate result Beta

            int[] Rp = R.p;
            int[] Ri = R.i;
            double[] Rx = R.x;

            int[] Vp = V.p;
            int[] Vi = V.i;
            double[] Vx = V.x;

            for (i = 0; i < m2; i++)
            {
                w[i] = -1; // clear w, to mark nodes
            }

            rnz = 0; vnz = 0;
            for (int k = 0; k < n; k++) // compute V and R
            {
                Rp[k] = rnz;      // R(:,k) starts here
                Vp[k] = p1 = vnz; // V(:,k) starts here
                w[k] = k;         // add V(k,k) to pattern of V
                Vi[vnz++] = k;
                top = n;
                col = q != null ? q[k] : k;
                for (p = Ap[col]; p < Ap[col + 1]; p++) // find R(:,k) pattern
                {
                    i = leftmost[Ai[p]]; // i = min(find(A(i,q)))
                    for (len = 0; w[i] != k; i = parent[i]) // traverse up to k
                    {
                        //len++;
                        w[s + len++] = i;
                        w[i] = k;
                    }
                    while (len > 0)
                    {
                        --top;
                        --len;
                        w[s + top] = w[s + len]; // push path on stack
                    }
                    i = pinv[Ai[p]]; // i = permuted row of A(:,col)
                    x[i] = Ax[p];    // x (i) = A(:,col)
                    if (i > k && w[i] < k) // pattern of V(:,k) = x (k+1:m)
                    {
                        Vi[vnz++] = i; // add i to pattern of V(:,k)
                        w[i] = k;
                    }
                }
                for (p = top; p < n; p++) // for each i in pattern of R(:,k)
                {
                    i = w[s + p]; // R(i,k) is nonzero
                    ApplyHouseholder(V, i, Beta[i], x); // apply (V(i),Beta(i)) to x
                    Ri[rnz] = i; // R(i,k) = x(i)
                    Rx[rnz++] = x[i];
                    x[i] = 0;
                    if (parent[i] == k)
                    {
                        vnz = V.Scatter(i, 0, w, null, k, V, vnz);
                    }
                }
                for (p = p1; p < vnz; p++) // gather V(:,k) = x
                {
                    Vx[p] = x[Vi[p]];
                    x[Vi[p]] = 0;
                }
                Ri[rnz] = k; // R(k,k) = norm (x)
                Rx[rnz++] = CreateHouseholder(Vx, p1, ref Beta[k], vnz - p1); // [v,beta]=house(x)
            }
            Rp[n] = rnz; // finalize R
            Vp[n] = vnz; // finalize V

            return N; // success
        }

        /// <summary>
        /// Create a Householder reflection [v,beta,s]=house(x), overwrite x with v,
        /// where (I-beta*v*v')*x = s*e1 and e1 = [1 0 ... 0]'.
        /// </summary>
        /// <remarks>
        /// Note that this CXSparse version is different than CSparse.  See Higham,
        /// Accuracy & Stability of Num Algorithms, 2nd ed, 2002, page 357.
        /// </remarks>
        static double CreateHouseholder(double[] x, int offset, ref double beta, int n)
        {
            double s = 0;
            int i;
            if (x == null) return -1; // check inputs

            // s = norm(x)
            for (i = 0; i < n; i++)
            {
                s += x[offset + i] * x[offset + i];
            }

            s = Math.Sqrt(s);
            if (s == 0)
            {
                beta = 0;
                x[offset] = 1;
            }
            else
            {
                // s = sign(x[0]) * norm (x) ;
                if (x[offset] != 0)
                {
                    s *= x[offset] / Math.Abs(x[offset]);
                }
                x[offset] += s;
                beta = 1 / (s * x[offset]);
            }
            return (-s);
        }

        /// <summary>
        /// Apply the ith Householder vector to x.
        /// </summary>
        static bool ApplyHouseholder(SparseMatrix V, int i, double beta, double[] x)
        {
            int p;
            double tau = 0;

            if (x == null) return false;     // check inputs

            int[] Vp = V.p;
            int[] Vi = V.i;
            double[] Vx = V.x;

            for (p = Vp[i]; p < Vp[i + 1]; p++)   // tau = v'*x
            {
                tau += Vx[p] * x[Vi[p]];
            }
            tau *= beta;                           // tau = beta*(v'*x)
            for (p = Vp[i]; p < Vp[i + 1]; p++)   // x = x - v*tau
            {
                x[Vi[p]] -= Vx[p] * tau;
            }
            return true;
        }

        /// <summary>
        /// Symbolic ordering and analysis for QR
        /// </summary>
        static SymbolicFactorization SymbolicAnalysis(int order, SparseMatrix A)
        { // ori: cs_sqr
            int n = A.n, k;
            bool ok = true;
            int[] post;

            SymbolicFactorization S = new SymbolicFactorization(); // allocate result S
            S.q = Ordering.AMD(order, A); // fill-reducing ordering

            if (order != 0 && S.q == null) return null;

            SparseMatrix C = order > 0 ? (SparseMatrix)A.Permute(null, S.q, false) : A;
            S.parent = Common.EliminationTree(C, true); // etree of C'*C, where C=A(:,q)
            post = Common.TreePostorder(S.parent, n);
            S.cp = Common.ColumnCounts(C, S.parent, post, true); // col counts chol(C'*C)

            ok = C != null && S.parent != null && S.cp != null && CountV(C, S);

            if (ok)
            {
                for (S.unz = 0, k = 0; k < n; k++)
                {
                    S.unz += S.cp[k];
                }
            }

            ok = ok && S.lnz >= 0 && S.unz >= 0; // int overflow guard

            return (ok ? S : null); // return result S
        }

        /// <summary>
        /// Compute nnz(V) = S.lnz, S.pinv, S.leftmost, S.m2 from A and S.parent
        /// </summary>
        static bool CountV(SparseMatrix A, SymbolicFactorization S)
        {
            int i, k, p, pa, n = A.n, m = A.m;
            int[] Ap = A.p, Ai = A.i, head,
                tail, nque, pinv, leftmost, w, parent = S.parent;

            S.pinv = pinv = new int[m + n]; // allocate pinv,
            S.leftmost = leftmost = new int[m]; // and leftmost

            w = new int[m]; // get workspace
            head = new int[n];
            tail = new int[n];
            nque = new int[n]; // Initialized to 0's

            for (k = 0; k < n; k++) head[k] = -1; // queue k is empty
            for (k = 0; k < n; k++) tail[k] = -1;
            for (i = 0; i < m; i++) leftmost[i] = -1;
            for (k = n - 1; k >= 0; k--)
            {
                for (p = Ap[k]; p < Ap[k + 1]; p++)
                {
                    leftmost[Ai[p]] = k; // leftmost[i] = min(find(A(i,:)))
                }
            }
            for (i = m - 1; i >= 0; i--) // scan rows in reverse order
            {
                pinv[i] = -1; // row i is not yet ordered
                k = leftmost[i];
                if (k == -1) continue; // row i is empty
                if (nque[k]++ == 0) tail[k] = i; // first row in queue k
                w[i] = head[k]; // put i at head of queue k
                head[k] = i;
            }
            S.lnz = 0;
            S.m2 = m;
            for (k = 0; k < n; k++) // find row permutation and nnz(V)
            {
                i = head[k]; // remove row i from queue k
                S.lnz++; // count V(k,k) as nonzero
                if (i < 0) i = S.m2++; // add a fictitious row
                pinv[i] = k; // associate row i with V(:,k)
                if (--nque[k] <= 0) continue; // skip if V(k+1:m,k) is empty
                S.lnz += nque[k]; // nque [k] is nnz (V(k+1:m,k))
                if ((pa = parent[k]) != -1) // move all rows to parent of k
                {
                    if (nque[pa] == 0) tail[pa] = tail[k];
                    w[tail[k]] = head[pa];
                    head[pa] = w[i];
                    nque[pa] += nque[k];
                }
            }
            for (i = 0; i < m; i++)
            {
                if (pinv[i] < 0)
                {
                    pinv[i] = k++;
                }
            }

            return true;
        }
    }
}
