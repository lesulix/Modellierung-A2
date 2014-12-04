﻿// -----------------------------------------------------------------------
// <copyright file="Ordering.cs" company="">
// Original CSparse code by Timothy A. Davis, http://www.suitesparse.com
// CSparse.NET code by Christian Woltering, http://csparse.codeplex.com/
// </copyright>
// -----------------------------------------------------------------------

namespace CSparse.Double
{
    using System;

    /// <summary>
    /// Minimum degree ordering.
    /// </summary>
    public static class Ordering
    {
        static int FLIP(int i) { return -(i) - 2; }

        #region Approximate Minimum degree

        // clear w
        static int Clear(int mark, int lemax, int[] w, int n)
        {
            int k;
            if (mark < 2 || (mark + lemax < 0))
            {
                for (k = 0; k < n; k++)
                {
                    if (w[k] != 0)
                    {
                        w[k] = 1;
                    }
                }
                mark = 2;
            }
            return (mark); // at this point, w [0..n-1] < mark holds
        }

        // keep off-diagonal entries; drop diagonal entries
        static bool KeepOffDiag(int i, int j, double aij, object other)
        {
            return (i != j);
        }

        /// <summary>
        /// Minimum degree ordering of A+A' (if A is symmetric) or A'A.
        /// </summary>
        /// <param name="order">0:natural, 1:Chol, 2:LU, 3:QR</param>
        /// <param name="A">column-compressed matrix</param>
        /// <returns>amd(A+A') if A is symmetric, or amd(A'A) otherwise, null on 
        /// error or for natural ordering</returns>
        public static int[] AMD(int order, SparseMatrix A)
        {
            SparseMatrix C, A2, AT;
            int[] Cp, Ci, P, W, nv, next, head, elen, degree, w,
                hhead, ATp, ATi;

            int d, dk, dext, lemax = 0, e, elenk, eln, i, j, k, k1,
                k2, k3, jlast, ln, dense, nzmax, mindeg = 0, nvi, nvj, nvk, mark, wnvi,
                cnz, nel = 0, p, p1, p2, p3, p4, pj, pk, pk1, pk2, pn, q, n, m;
            bool ok;
            int h;

            // Construct matrix C
            if (order <= 0 || order > 3) return null; // check
            AT = (SparseMatrix)A.Transpose(false); // compute A'
            if (AT == null) return null;
            m = A.m; n = A.n;
            dense = Math.Max(16, 10 * (int)Math.Sqrt(n)); // find dense threshold
            dense = Math.Min(n - 2, dense);
            if (order == 1 && n == m)
            {
                C = SparseMatrix.Add(A, AT, 0, 0); // C = A+A'
            }
            else if (order == 2)
            {
                ATp = AT.p; // drop dense columns from AT
                ATi = AT.i;
                for (p2 = 0, j = 0; j < m; j++)
                {
                    p = ATp[j]; // column j of AT starts here
                    ATp[j] = p2; // new column j starts here
                    if (ATp[j + 1] - p > dense) continue; // skip dense col j
                    for (; p < ATp[j + 1]; p++) ATi[p2++] = ATi[p];
                }
                ATp[m] = p2; // finalize AT
                A2 = (SparseMatrix)AT.Transpose(false); // A2 = AT'
                C = A2 != null ? SparseMatrix.Multiply(AT, A2) : null; // C=A'*A with no dense rows
            }
            else
            {
                C = SparseMatrix.Multiply(AT, A); // C=A'*A
            }

            if (C == null) return null;
            C.Keep(KeepOffDiag, null); // drop diagonal entries
            Cp = C.p;
            cnz = Cp[n];

            // add elbow room to C
            if (!C.Resize(cnz + cnz / 5 + 2 * n)) return null;

            P = new int[n + 1]; // allocate result
            W = new int[n + 1]; // get workspace
            w = new int[n + 1];
            degree = new int[n + 1];

            elen = new int[n + 1]; // Initialized to 0's

            // Initialize quotient graph
            for (k = 0; k < n; k++)
            {
                W[k] = Cp[k + 1] - Cp[k];
            }
            W[n] = 0;
            nzmax = C.nzmax;
            Ci = C.i;

            for (i = 0; i <= n; i++)
            {
                P[i] = -1;
                w[i] = 1; // node i is alive
                degree[i] = W[i]; // degree of node i
            }

            next = new int[n + 1];
            hhead = new int[n + 1];
            head = new int[n + 1];
            nv = new int[n + 1];

            Array.Copy(P, next, n + 1);
            Array.Copy(P, head, n + 1); // degree list i is empty
            Array.Copy(P, hhead, n + 1); // hash list i is empty
            Array.Copy(w, nv, n + 1); // node i is just one node

            mark = Clear(0, 0, w, n); // clear w
            elen[n] = -2; // n is a dead element
            Cp[n] = -1; // n is a root of assembly tree
            w[n] = 0; // n is a dead element

            // Initialize degree lists
            for (i = 0; i < n; i++)
            {
                d = degree[i];
                if (d == 0) // node i is empty
                {
                    elen[i] = -2; // element i is dead
                    nel++;
                    Cp[i] = -1; // i is a root of assembly tree
                    w[i] = 0;
                }
                else if (d > dense) // node i is dense
                {
                    nv[i] = 0; // absorb i into element n
                    elen[i] = -1; // node i is dead
                    nel++;
                    Cp[i] = FLIP(n);
                    nv[n]++;
                }
                else
                {
                    if (head[d] != -1) P[head[d]] = i;
                    next[i] = head[d]; // put node i in degree list d
                    head[d] = i;
                }
            }
            while (nel < n) // while (selecting pivots) do
            {
                // Select node of minimum approximate degree
                for (k = -1; mindeg < n && (k = head[mindeg]) == -1; mindeg++) ;
                if (next[k] != -1) P[next[k]] = -1;
                head[mindeg] = next[k]; // remove k from degree list
                elenk = elen[k]; // elenk = |Ek|
                nvk = nv[k]; // # of nodes k represents
                nel += nvk; // nv[k] nodes of A eliminated

                // Garbage collection
                if (elenk > 0 && cnz + mindeg >= nzmax)
                {
                    for (j = 0; j < n; j++)
                    {
                        if ((p = Cp[j]) >= 0) // j is a live node or element
                        {
                            Cp[j] = Ci[p]; // save first entry of object
                            Ci[p] = FLIP(j); // first entry is now CS_FLIP(j)
                        }
                    }
                    for (q = 0, p = 0; p < cnz; ) // scan all of memory
                    {
                        if ((j = FLIP(Ci[p++])) >= 0) // found object j
                        {
                            Ci[q] = Cp[j]; // restore first entry of object
                            Cp[j] = q++; // new pointer to object j
                            for (k3 = 0; k3 < W[j] - 1; k3++) Ci[q++] = Ci[p++];
                        }
                    }
                    cnz = q; // Ci [cnz...nzmax-1] now free
                }

                // Construct new element
                dk = 0;
                nv[k] = -nvk; // flag k as in Lk
                p = Cp[k];
                pk1 = (elenk == 0) ? p : cnz; // do in place if elen[k] == 0
                pk2 = pk1;
                for (k1 = 1; k1 <= elenk + 1; k1++)
                {
                    if (k1 > elenk)
                    {
                        e = k; // search the nodes in k
                        pj = p; // list of nodes starts at Ci[pj]*/
                        ln = W[k] - elenk; // length of list of nodes in k
                    }
                    else
                    {
                        e = Ci[p++]; // search the nodes in e
                        pj = Cp[e];
                        ln = W[e]; // length of list of nodes in e
                    }
                    for (k2 = 1; k2 <= ln; k2++)
                    {
                        i = Ci[pj++];
                        if ((nvi = nv[i]) <= 0) continue; // node i dead, or seen
                        dk += nvi; // degree[Lk] += size of node i
                        nv[i] = -nvi; // negate nv[i] to denote i in Lk
                        Ci[pk2++] = i; // place i in Lk
                        if (next[i] != -1) P[next[i]] = P[i];
                        if (P[i] != -1) // remove i from degree list
                        {
                            next[P[i]] = next[i];
                        }
                        else
                        {
                            head[degree[i]] = next[i];
                        }
                    }
                    if (e != k)
                    {
                        Cp[e] = FLIP(k); // absorb e into k
                        w[e] = 0; // e is now a dead element
                    }
                }
                if (elenk != 0) cnz = pk2; // Ci [cnz...nzmax] is free
                degree[k] = dk; // external degree of k - |Lk\i|
                Cp[k] = pk1; // element k is in Ci[pk1..pk2-1]
                W[k] = pk2 - pk1;
                elen[k] = -2; // k is now an element

                // Find set differences
                mark = Clear(mark, lemax, w, n); // clear w if necessary
                for (pk = pk1; pk < pk2; pk++) // scan 1: find |Le\Lk|
                {
                    i = Ci[pk];
                    if ((eln = elen[i]) <= 0) continue; // skip if elen[i] empty
                    nvi = -nv[i]; // nv [i] was negated
                    wnvi = mark - nvi;
                    for (p = Cp[i]; p <= Cp[i] + eln - 1; p++) // scan Ei
                    {
                        e = Ci[p];
                        if (w[e] >= mark)
                        {
                            w[e] -= nvi; // decrement |Le\Lk|
                        }
                        else if (w[e] != 0) // ensure e is a live element
                        {
                            w[e] = degree[e] + wnvi; // 1st time e seen in scan 1
                        }
                    }
                }

                // Degree update
                for (pk = pk1; pk < pk2; pk++) // scan2: degree update
                {
                    i = Ci[pk]; // consider node i in Lk
                    p1 = Cp[i];
                    p2 = p1 + elen[i] - 1;
                    pn = p1;
                    for (h = 0, d = 0, p = p1; p <= p2; p++) // scan Ei
                    {
                        e = Ci[p];
                        if (w[e] != 0) // e is an unabsorbed element
                        {
                            dext = w[e] - mark; // dext = |Le\Lk|
                            if (dext > 0)
                            {
                                d += dext; // sum up the set differences
                                Ci[pn++] = e; // keep e in Ei
                                h += e; // compute the hash of node i
                            }
                            else
                            {
                                Cp[e] = FLIP(k); // aggressive absorb. e.k
                                w[e] = 0; // e is a dead element
                            }
                        }
                    }
                    elen[i] = pn - p1 + 1; // elen[i] = |Ei|
                    p3 = pn;
                    p4 = p1 + W[i];
                    for (p = p2 + 1; p < p4; p++) // prune edges in Ai
                    {
                        j = Ci[p];
                        if ((nvj = nv[j]) <= 0) continue; // node j dead or in Lk
                        d += nvj; // degree(i) += |j|
                        Ci[pn++] = j; // place j in node list of i
                        h += j; // compute hash for node i
                    }
                    if (d == 0) // check for mass elimination
                    {
                        Cp[i] = FLIP(k); // absorb i into k
                        nvi = -nv[i];
                        dk -= nvi; // |Lk| -= |i|
                        nvk += nvi; // |k| += nv[i]
                        nel += nvi;
                        nv[i] = 0;
                        elen[i] = -1; // node i is dead
                    }
                    else
                    {
                        degree[i] = Math.Min(degree[i], d); // update degree(i)
                        Ci[pn] = Ci[p3]; // move first node to end
                        Ci[p3] = Ci[p1]; // move 1st el. to end of Ei
                        Ci[p1] = k; // add k as 1st element in of Ei
                        W[i] = pn - p1 + 1; // new len of adj. list of node i
                        h = ((h < 0) ? (-h) : h) % n; // finalize hash of i
                        next[i] = hhead[h]; // place i in hash bucket
                        hhead[h] = i;
                        P[i] = h; // save hash of i in last[i]
                    }
                } // scan2 is done
                degree[k] = dk; // finalize |Lk|
                lemax = Math.Max(lemax, dk);
                mark = Clear(mark + lemax, lemax, w, n); // clear w

                // Supernode detection
                for (pk = pk1; pk < pk2; pk++)
                {
                    i = Ci[pk];
                    if (nv[i] >= 0) continue; // skip if i is dead
                    h = P[i]; // scan hash bucket of node i
                    i = hhead[h];
                    hhead[h] = -1; // hash bucket will be empty
                    for (; i != -1 && next[i] != -1; i = next[i], mark++)
                    {
                        ln = W[i];
                        eln = elen[i];
                        for (p = Cp[i] + 1; p <= Cp[i] + ln - 1; p++) w[Ci[p]] = mark;
                        jlast = i;
                        for (j = next[i]; j != -1; ) // compare i with all j
                        {
                            ok = (W[j] == ln) && (elen[j] == eln);
                            for (p = Cp[j] + 1; ok && p <= Cp[j] + ln - 1; p++)
                            {
                                if (w[Ci[p]] != mark) ok = false; // compare i and j
                            }
                            if (ok) // i and j are identical
                            {
                                Cp[j] = FLIP(i); // absorb j into i
                                nv[i] += nv[j];
                                nv[j] = 0;
                                elen[j] = -1; // node j is dead
                                j = next[j]; // delete j from hash bucket
                                next[jlast] = j;
                            }
                            else
                            {
                                jlast = j; // j and i are different
                                j = next[j];
                            }
                        }
                    }
                }

                // Finalize new element
                for (p = pk1, pk = pk1; pk < pk2; pk++) // finalize Lk
                {
                    i = Ci[pk];
                    if ((nvi = -nv[i]) <= 0) continue; // skip if i is dead
                    nv[i] = nvi; // restore nv[i]
                    d = degree[i] + dk - nvi; // compute external degree(i)
                    d = Math.Min(d, n - nel - nvi);
                    if (head[d] != -1) P[head[d]] = i;
                    next[i] = head[d]; // put i back in degree list
                    P[i] = -1;
                    head[d] = i;
                    mindeg = Math.Min(mindeg, d); // find new minimum degree
                    degree[i] = d;
                    Ci[p++] = i; // place i in Lk
                }
                nv[k] = nvk; // # nodes absorbed into k
                if ((W[k] = p - pk1) == 0) // length of adj list of element k
                {
                    Cp[k] = -1; // k is a root of the tree
                    w[k] = 0; // k is now a dead element
                }
                if (elenk != 0) cnz = p; // free unused space in Lk
            }

            // Postordering
            for (i = 0; i < n; i++) Cp[i] = FLIP(Cp[i]); // fix assembly tree
            for (j = 0; j <= n; j++) head[j] = -1;
            for (j = n; j >= 0; j--) // place unordered nodes in lists
            {
                if (nv[j] > 0) continue; // skip if j is an element
                next[j] = head[Cp[j]]; // place j in list of its parent
                head[Cp[j]] = j;
            }
            for (e = n; e >= 0; e--) // place elements in lists
            {
                if (nv[e] <= 0) continue; // skip unless e is an element
                if (Cp[e] != -1)
                {
                    next[e] = head[Cp[e]]; // place e in list of its parent
                    head[Cp[e]] = e;
                }
            }
            for (k = 0, i = 0; i <= n; i++) // postorder the assembly tree
            {
                if (Cp[i] == -1) k = Common.TreeDepthFirstSearch(i, k, head, next, P, w);
            }
            return P;
        }

        #endregion
    }
}
