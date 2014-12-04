// -----------------------------------------------------------------------
// <copyright file="DulmageMendelsohn.cs" company="">
// Original CSparse code by Timothy A. Davis, http://www.suitesparse.com
// CSparse.NET code by Christian Woltering, http://csparse.codeplex.com/
// </copyright>
// -----------------------------------------------------------------------

namespace CSparse
{
    using System;

    /// <summary>
    /// Dulmage-Mendelsohn analysis.
    /// </summary>
    public static class DulmageMendelsohn
    {
        #region Maximum matching

        /// <summary>
        /// Find an augmenting path starting at column k and extend the match if found.
        /// </summary>
        static void Augment(int k, Matrix A, int[] jmatch, int jmatch_offset, int[] cheap, 
            int[] w, int[] js, int[] iss, int[] ps)
        {
            bool found = false;
            int p, i = -1, head = 0, j;
            int[] Ap = A.p, Ai = A.i;

            js[0] = k; // start with just node k in jstack
            while (head >= 0)
            {
                // Start (or continue) depth-first-search at node j
                j = js[head]; // get j from top of jstack
                if (w[j] != k) // 1st time j visited for kth path
                {
                    w[j] = k; // mark j as visited for kth path
                    for (p = cheap[j]; p < Ap[j + 1] && !found; p++)
                    {
                        i = Ai[p]; // try a cheap assignment (i,j)
                        found = (jmatch[jmatch_offset + i] == -1);
                    }
                    cheap[j] = p; // start here next time j is traversed*/
                    if (found)
                    {
                        iss[head] = i; // column j matched with row i
                        break; // end of augmenting path
                    }
                    ps[head] = Ap[j]; // no cheap match: start dfs for j
                }

                // Depth-first-search of neighbors of j
                for (p = ps[head]; p < Ap[j + 1]; p++)
                {
                    i = Ai[p]; // consider row i
                    if (w[jmatch[jmatch_offset + i]] == k)
                    {
                        continue; // skip jmatch [i] if marked
                    }
                    ps[head] = p + 1; // pause dfs of node j
                    iss[head] = i; // i will be matched with j if found
                    head++;
                    js[head] = jmatch[jmatch_offset + i]; // start dfs at column jmatch [i]
                    break;
                }
                if (p == Ap[j + 1])
                {
                    head--; // node j is done; pop from stack
                }
            }
            // augment the match if path found:
            if (found)
            {
                for (p = head; p >= 0; p--)
                {
                    jmatch[jmatch_offset + iss[p]] = js[p];
                }
            }
        }

        /// <summary>
        /// Find a maximum transveral (zero-free diagonal). Seed optionally selects a
        /// randomized algorithm.
        /// </summary>
        /// <param name="A">column-compressed matrix</param>
        /// <param name="seed">: natural, -1: reverse, randomized otherwise</param>
        /// <returns>row and column matching, size m+n</returns>
        static int[] MaximumTransveral(Matrix A, int seed) 
        {
            int i, j, k, p, n2 = 0, m2 = 0;
            int[] jimatch, w, cheap, js, iss, ps, Cp, q;
            Matrix C;

            int n = A.n;
            int m = A.m;
            int[] Ap = A.p;
            int[] Ai = A.i;
            
            //[jmatch [0..m-1]; imatch [0..n-1]]
            w = jimatch = new int[m + n]; // allocate result

            for (k = 0, j = 0; j < n; j++) // count nonempty rows and columns
            {
                n2 += (Ap[j] < Ap[j + 1]) ? 1 : 0;
                for (p = Ap[j]; p < Ap[j + 1]; p++)
                {
                    w[Ai[p]] = 1;
                    k += (j == Ai[p]) ? 1 : 0; // count entries already on diagonal
                }
            }


            if (k == Math.Min(m, n)) // quick return if diagonal zero-free
            {
                for (i = 0; i < k; i++) jimatch[i] = i;
                for (; i < m; i++) jimatch[i] = -1;
                for (j = 0; j < k; j++) jimatch[m + j] = j;
                for (; j < n; j++) jimatch[m + j] = -1;

                return jimatch;
            }

            for (i = 0; i < m; i++) m2 += w[i];

            C = (m2 < n2) ? A.Transpose(false) : A; // transpose if needed

            if (C == null) return jimatch;

            n = C.n;
            m = C.m;
            Cp = C.p;

            int jmatch_offset = (m2 < n2) ? n : 0;
            int imatch_offset = (m2 < n2) ? 0 : m;

            w = new int[n]; // get workspace

            cheap = new int[n];
            js = new int[n];
            iss = new int[n];
            ps = new int[n];

            for (j = 0; j < n; j++) cheap[j] = Cp[j]; // for cheap assignment
            for (j = 0; j < n; j++) w[j] = -1; // all columns unflagged
            for (i = 0; i < m; i++) jimatch[jmatch_offset + i] = -1; // nothing matched yet

            q = Common.RandomPermute(n, seed); // q = random permutation
            for (k = 0; k < n; k++) // augment, starting at column q[k]
            {
                Augment(q != null ? q[k] : k, C, jimatch, jmatch_offset, cheap, w, js, iss, ps);
            }

            for (j = 0; j < n; j++)
            {
                jimatch[imatch_offset + j] = -1; // find row match
            }

            for (i = 0; i < m; i++)
            {
                if (jimatch[jmatch_offset + i] >= 0)
                {
                    jimatch[imatch_offset + jimatch[jmatch_offset + i]] = i;
                }
            }

            return jimatch;
        }

        #endregion

        #region Block triangular form

        /// <summary>
        /// Finds the strongly connected components of a square matrix.
        /// </summary>
        /// <param name="A">column-compressed matrix (A.p modified then restored)</param>
        /// <returns>strongly connected components, null on error</returns>
        static Decomposition FindScc(Matrix A)
        {
            // matrix A temporarily modified, then restored

            int n, i, k, b, nb = 0, top;
            int[] xi, p, r, Ap, ATp;
            Matrix AT;

            n = A.n;
            Ap = A.p;

            AT = A.Transpose(false); // AT = A'
            ATp = AT.p;

            xi = new int[2 * n + 1]; // get workspace

            Decomposition D = new Decomposition(n, 0); // allocate result
            p = D.p;
            r = D.r;

            top = n;
            for (i = 0; i < n; i++) // first dfs(A) to find finish times (xi)
            {
                if (!(Ap[i] < 0))
                {
                    top = Common.DepthFirstSearch(i, A, top, xi, xi, n, null);
                }
            }
            for (i = 0; i < n; i++)
            {
                //CS_MARK(Ap, i);
                Ap[i] = -(Ap[i]) - 2; // restore A; unmark all nodes
            }
            top = n;
            nb = n;
            for (k = 0; k < n; k++) // dfs(A') to find strongly connnected comp
            {
                i = xi[k]; // get i in reverse order of finish times
                if (ATp[i] < 0)
                {
                    continue; // skip node i if already ordered
                }
                r[nb--] = top; // node i is the start of a component in p
                top = Common.DepthFirstSearch(i, AT, top, p, xi, n, null);
            }
            r[nb] = 0; // first block starts at zero; shift r up
            for (k = nb; k <= n; k++) r[k - nb] = r[k];
            D.nb = nb = n - nb; // nb = # of strongly connected components
            for (b = 0; b < nb; b++) // sort each block in natural order
            {
                for (k = r[b]; k < r[b + 1]; k++) xi[p[k]] = b;
            }
            for (b = 0; b <= nb; b++)
            {
                xi[n + b] = r[b];
            }
            for (i = 0; i < n; i++)
            {
                p[xi[n + xi[i]]++] = i;
            }
            return D;
        }

        #endregion

        #region Dulmage-Mendelsohn

        // breadth-first search for coarse decomposition (C0,C1,R1 or R0,R3,C3)
        static bool BreadthFirstSearch(Matrix A, int n, int[] wi, int[] wj, int[] queue,
            int[] jimatch, int imatch_offset, int jmatch_offset, int mark)
        {
            // cs_bfs
            int[] Ap, Ai;
            int head = 0, tail = 0, j, i, p, j2;
            Matrix C;
            for (j = 0; j < n; j++) // place all unmatched nodes in queue
            {
                if (jimatch[imatch_offset + j] >= 0) continue; // skip j if matched
                wj[j] = 0; // j in set C0 (R0 if transpose)
                queue[tail++] = j; // place unmatched col j in queue
            }
            if (tail == 0) return true; // quick return if no unmatched nodes
            C = (mark == 1) ? A : A.Transpose(false);
            if (C == null) return false; // bfs of C=A' to find R3,C3 from R0
            Ap = C.p; Ai = C.i;
            while (head < tail) // while queue is not empty
            {
                j = queue[head++]; // get the head of the queue
                for (p = Ap[j]; p < Ap[j + 1]; p++)
                {
                    i = Ai[p];
                    if (wi[i] >= 0) continue; // skip if i is marked
                    wi[i] = mark; // i in set R1 (C3 if transpose)
                    j2 = jimatch[jmatch_offset + i]; // traverse alternating path to j2
                    if (wj[j2] >= 0) continue; // skip j2 if it is marked
                    wj[j2] = mark; // j2 in set C1 (R3 if transpose)
                    queue[tail++] = j2; // add j2 to queue
                }
            }
            //if (mark != 1) SparseMatrix.spfree(C); // free A' if it was created
            return true;
        }

        // collect matched rows and columns into p and q
        static void Matched(int n, int[] wj, int[] imatch, int imatch_offset, int[] p, int[] q,
            int[] cc, int[] rr, int set, int mark)
        {
            int kc = cc[set];
            int kr = rr[set - 1];
            for (int j = 0; j < n; j++)
            {
                if (wj[j] != mark) continue; // skip if j is not in C set
                p[kr++] = imatch[imatch_offset + j];
                q[kc++] = j;
            }
            cc[set + 1] = kc;
            rr[set] = kr;
        }

        // collect unmatched rows into the permutation vector p
        static void Unmatched(int m, int[] wi, int[] p, int[] rr, int set)
        {
            int i, kr = rr[set];
            for (i = 0; i < m; i++) if (wi[i] == 0) p[kr++] = i;
            rr[set + 1] = kr;
        }

        // return 1 if row i is in R2
        static bool RowPrune(int i, int j, int[] rr)
        {
            return (i >= rr[1] && i < rr[2]);
        }

        /// <summary>
        /// Compute coarse and then fine Dulmage-Mendelsohn decomposition. seed
        /// optionally selects a randomized algorithm.
        /// </summary>
        /// <param name="A">column-compressed matrix</param>
        /// <param name="seed">0: natural, -1: reverse, random order oterwise</param>
        /// <returns>Dulmage-Mendelsohn analysis, null on error</returns>
        public static Decomposition Analyse(Matrix A, int seed)
        {
            int i, j, k, cnz, nc, nb1, nb2;
            int[] Cp, ps, rs;
            bool ok;

            // Maximum matching
            int m = A.m;
            int n = A.n;

            Decomposition D = new Decomposition(m, n); // allocate result
            int[] p = D.p;
            int[] q = D.q;
            int[] r = D.r;
            int[] s = D.s;
            int[] cc = D.cc;
            int[] rr = D.rr;

            int[] jimatch = MaximumTransveral(A, seed); // max transversal

            if (jimatch == null) return null;

            // Coarse decomposition
            for (j = 0; j < n; j++) s[j] = -1; // unmark all cols for bfs
            for (i = 0; i < m; i++) r[i] = -1; // unmark all rows for bfs

            BreadthFirstSearch(A, n, r, s, q, jimatch, m, 0, 1); // find C1, R1 from C0*/
            ok = BreadthFirstSearch(A, m, s, r, p, jimatch, 0, m, 3); // find R3, C3 from R0*/

            if (!ok) return null;

            Unmatched(n, s, q, cc, 0); // unmatched set C0
            Matched(n, s, jimatch, m, p, q, cc, rr, 1, 1); // set R1 and C1
            Matched(n, s, jimatch, m, p, q, cc, rr, 2, -1); // set R2 and C2
            Matched(n, s, jimatch, m, p, q, cc, rr, 3, 3); // set R3 and C3
            Unmatched(m, r, p, rr, 3); // unmatched set R0

            // Fine decomposition
            int[] pinv = Common.InversePermutation(p, m); // pinv=p'

            Matrix C = A.Permute(pinv, q, false); // C=A(p,q) (it will hold A(R2,C2))

            Cp = C.p;
            nc = cc[3] - cc[2]; // delete cols C0, C1, and C3 from C
            if (cc[2] > 0)
            {
                for (j = cc[2]; j <= cc[3]; j++)
                {
                    Cp[j - cc[2]] = Cp[j];
                }
            }
            C.n = nc;
            if (rr[2] - rr[1] < m) // delete rows R0, R1, and R3 from C
            {
                C.KeepSymbolic(RowPrune, rr);
                cnz = Cp[nc];
                int[] Ci = C.i;
                if (rr[1] > 0)
                {
                    for (k = 0; k < cnz; k++)
                    {
                        Ci[k] -= rr[1];
                    }
                }
            }
            C.m = nc;
            Decomposition scc = FindScc(C); // find strongly connected components of C*/

            // Combine coarse and fine decompositions
            ps = scc.p; // C(ps,ps) is the permuted matrix
            rs = scc.r; // kth block is rs[k]..rs[k+1]-1
            nb1 = scc.nb; // # of blocks of A(R2,C2)
            for (k = 0; k < nc; k++) s[k] = q[ps[k] + cc[2]];
            for (k = 0; k < nc; k++) q[k + cc[2]] = s[k];
            for (k = 0; k < nc; k++) r[k] = p[ps[k] + rr[1]];
            for (k = 0; k < nc; k++) p[k + rr[1]] = r[k];
            nb2 = 0; // create the fine block partitions
            r[0] = s[0] = 0;
            if (cc[2] > 0) nb2++; // leading coarse block A (R1, [C0 C1])
            for (k = 0; k < nb1; k++) // coarse block A (R2,C2)
            {
                r[nb2] = rs[k] + rr[1]; // A (R2,C2) splits into nb1 fine blocks
                s[nb2] = rs[k] + cc[2];
                nb2++;
            }
            if (rr[2] < m)
            {
                r[nb2] = rr[2]; // trailing coarse block A ([R3 R0], C3)
                s[nb2] = cc[3];
                nb2++;
            }
            r[nb2] = m;
            s[nb2] = n;
            D.nb = nb2;

            return D;
        }

        #endregion
    }
}
