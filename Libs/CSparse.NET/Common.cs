// -----------------------------------------------------------------------
// <copyright file="Common.cs" company="">
// Original CSparse code by Timothy A. Davis, http://www.suitesparse.com
// CSparse.NET code by Christian Woltering, http://csparse.codeplex.com/
// </copyright>
// -----------------------------------------------------------------------

namespace CSparse
{
    using System;

    /// <summary>
    /// Common utility functions.
    /// </summary>
    public static class Common
    {
        /// <summary>
        /// Depth-first-search of the graph of a matrix, starting at node j.
        /// </summary>
        /// <param name="j">starting node</param>
        /// <param name="G">graph to search (G.p modified, then restored)</param>
        /// <param name="top">stack[top..n-1] is used on input</param>
        /// <param name="xi">size n, stack containing nodes traversed</param>
        /// <param name="pstack">size n, work array</param>
        /// <param name="pstack_offset">the index of the first element in array pstack</param>
        /// <param name="pinv">mapping of rows to columns of G, ignored if null</param>
        /// <returns>new value of top, -1 on error</returns>
        public static int DepthFirstSearch(int j, Matrix G, int top, int[] xi,
            int[] pstack, int pstack_offset, int[] pinv)
        {
            int i, p, p2, jnew, head = 0;
            bool done;
            int[] Gp, Gi;
            if (xi == null || pstack == null) return (-1); // check inputs
            Gp = G.p; Gi = G.i;
            xi[0] = j; // initialize the recursion stack
            while (head >= 0)
            {
                j = xi[head]; // get j from the top of the recursion stack
                jnew = pinv != null ? (pinv[j]) : j;
                if (!(Gp[j] < 0))
                {
                    //CS_MARK(Gp, j);
                    Gp[j] = -(Gp[j]) - 2; // mark node j as visited
                    pstack[pstack_offset + head] = (jnew < 0) ? 0 : // CS_UNFLIP(Gp[jnew]);
                        ((Gp[jnew] < 0) ? -Gp[jnew] - 2 : Gp[jnew]);
                }
                done = true; // node j done if no unvisited neighbors
                p2 = (jnew < 0) ? 0 : // CS_UNFLIP(Gp[jnew + 1]);
                    ((Gp[jnew + 1] < 0) ? -Gp[jnew + 1] - 2 : Gp[jnew + 1]);

                for (p = pstack[pstack_offset + head]; p < p2; p++) // examine all neighbors of j
                {
                    i = Gi[p]; // consider neighbor node i
                    if (Gp[i] < 0) continue; // skip visited node i
                    pstack[pstack_offset + head] = p; // pause depth-first search of node j
                    xi[++head] = i; // start dfs at node i
                    done = false; // node j is not done
                    break; // break, to start dfs (i)
                }
                if (done) // depth-first search at node j is done
                {
                    head--; // remove j from the recursion stack
                    xi[--top] = j; // and place in the output stack
                }
            }
            return (top);
        }

        /// <summary>
        /// Depth-first search and postorder of a tree rooted at node j
        /// </summary>
        /// <param name="j">postorder of a tree rooted at node j</param>
        /// <param name="k">number of nodes ordered so far</param>
        /// <param name="head">head[i] is first child of node i; -1 on output</param>
        /// <param name="next">next[i] is next sibling of i or -1 if none</param>
        /// <param name="post">postordering</param>
        /// <param name="stack">size n, work array</param>
        /// <returns>new value of k, -1 on error</returns>
        public static int TreeDepthFirstSearch(int j, int k, int[] head, int[] next, int[] post, int[] stack)
        {
            int i, p, top = 0;

            if (head == null || next == null || post == null || stack == null)
                return (-1); // check inputs

            stack[0] = j; // place j on the stack
            while (top >= 0) // while (stack is not empty)
            {
                p = stack[top]; // p = top of stack
                i = head[p]; // i = youngest child of p
                if (i == -1)
                {
                    top--; // p has no unordered children left
                    post[k++] = p; // node p is the kth postordered node
                }
                else
                {
                    head[p] = next[i]; // remove i from children of p
                    top++;
                    stack[top] = i; // start dfs on child node i
                }
            }
            return (k);
        }

        /// <summary>
        /// Compute the elimination tree of A or A'A (without forming A'A).
        /// </summary>
        /// <param name="A">column-compressed matrix</param>
        /// <param name="ata">analyze A if false, A'A oterwise</param>
        /// <returns>elimination tree, null on error</returns>
        public static int[] EliminationTree(Matrix A, bool ata)
        {
            int i, k, p, m, n, inext;
            int[] Ap, Ai, w, parent, ancestor, prev;

            m = A.m; n = A.n; Ap = A.p; Ai = A.i;
            parent = new int[n]; // allocate result
            w = new int[n + (ata ? m : 0)]; // get workspace
            if (w == null || parent == null) return null;
            ancestor = w; prev = w;
            int prev_offset = n;
            if (ata) for (i = 0; i < m; i++) prev[prev_offset + i] = -1;
            for (k = 0; k < n; k++)
            {
                parent[k] = -1; // node k has no parent yet
                ancestor[k] = -1; // nor does k have an ancestor
                for (p = Ap[k]; p < Ap[k + 1]; p++)
                {
                    i = ata ? (prev[prev_offset + Ai[p]]) : (Ai[p]);
                    for (; i != -1 && i < k; i = inext) // traverse from i to k
                    {
                        inext = ancestor[i]; // inext = ancestor of i
                        ancestor[i] = k; // path compression
                        if (inext == -1) parent[i] = k; // no anc., parent is k
                    }
                    if (ata) prev[prev_offset + Ai[p]] = k;
                }
            }
            return parent;
        }

        /// <summary>
        /// Find nonzero pattern of Cholesky L(k,1:k-1) using etree and triu(A(:,k))
        /// </summary>
        public static int EReach(Matrix A, int k, int[] parent, int[] s, int[] w)
        {
            int i, p, n, len;

            if (parent == null || s == null || w == null) return -1;   // check inputs

            int top = n = A.n;
            int[] Ap = A.p;
            int[] Ai = A.i;

            //CS_MARK(w, k);
            w[k] = -w[k] - 2; // mark node k as visited

            for (p = Ap[k]; p < Ap[k + 1]; p++)
            {
                i = Ai[p]; // A(i,k) is nonzero
                if (i > k)
                {
                    continue; // only use upper triangular part of A
                }
                for (len = 0; !(w[i] < 0); i = parent[i]) // traverse up etree
                {
                    s[len++] = i; // L(k,i) is nonzero
                    //CS_MARK(w, i);       
                    w[i] = -w[i] - 2; // mark i as visited
                }
                while (len > 0)
                {
                    s[--top] = s[--len]; // push path onto stack
                }
            }
            for (p = top; p < n; p++)
            {
                //CS_MARK(w, s[p]);
                w[s[p]] = -w[s[p]] - 2; // unmark all nodes
            }
            //CS_MARK(w, k);
            w[k] = -w[k] - 2; // unmark node k


            return top; // s [top..n-1] contains pattern of L(k,:)
        }


        // xi [top...n-1] = nodes reachable from graph of G*P' via nodes in B(:,k).
        // xi [n...2n-1] used as workspace
        public static int Reach(Matrix G, Matrix B, int k, int[] xi, int[] pinv)
        {
            if (xi == null) return (-1); // check inputs

            int n = G.n;
            int[] Bp = B.p;
            int[] Bi = B.i;
            int[] Gp = G.p;
            int top = n;
            int p;

            for (p = Bp[k]; p < Bp[k + 1]; p++)
            {
                //if (!CS_MARKED(Gp, Bi[p]))
                if (!(Gp[Bi[p]] < 0)) // start a dfs at unmarked node i
                {
                    top = Common.DepthFirstSearch(Bi[p], G, top, xi, xi, n, pinv);
                }
            }

            for (p = top; p < n; p++)
            {
                //CS_MARK(Gp, xi[p]);
                Gp[xi[p]] = -(Gp[xi[p]]) - 2; // restore G
            }

            return (top);
        }

        /// <summary>
        /// Postorders a tree of forest.
        /// </summary>
        /// <param name="parent">defines a tree of n nodes</param>
        /// <param name="n">length of parent</param>
        /// <returns>post[k]=i, null on error</returns>
        public static int[] TreePostorder(int[] parent, int n)
        {
            int j, k = 0;
            int[] post, w, next, stack;
            if (parent == null) return null; // check inputs
            post = new int[n]; // allocate result
            w = new int[n]; // get workspace
            next = new int[n];
            stack = new int[n];

            for (j = 0; j < n; j++) w[j] = -1; // empty linked lists
            for (j = n - 1; j >= 0; j--) // traverse nodes in reverse order*/
            {
                if (parent[j] == -1) continue; // j is a root
                next[j] = w[parent[j]]; // add j to list of its parent
                w[parent[j]] = j;
            }
            for (j = 0; j < n; j++)
            {
                if (parent[j] != -1) continue; // skip j if it is not a root
                k = TreeDepthFirstSearch(j, k, w, next, post, stack);
            }
            return post; // success; free w, return post
        }

        /// <summary>
        /// Column counts for Cholesky (LL'=A or LL'=A'A) and QR, given parent & post ordering.
        /// </summary>
        /// <param name=""></param>
        /// <returns></returns>
        public static int[] ColumnCounts(Matrix A, int[] parent, int[] post, bool ata)
        {
            int i, j, k, J, p, q, jleaf = 0;
            int[] ATp, ATi, colcount, delta, head = null, next = null;

            if (parent == null || post == null) return (null); // check inputs

            int m = A.m;
            int n = A.n;

            delta = colcount = new int[n]; // allocate result

            Matrix AT = A.Transpose(false); // AT = A'

            // w is ancestor
            int[] w = new int[n]; // get workspace

            int[] maxfirst = new int[n];
            int[] prevleaf = new int[n];
            int[] first = new int[n];

            for (k = 0; k < n; k++)
            {
                w[k] = -1; // clear workspace w [0..s-1]
            }
            Array.Copy(w, maxfirst, n);
            Array.Copy(w, prevleaf, n);
            Array.Copy(w, first, n);

            for (k = 0; k < n; k++) // find first [j]
            {
                j = post[k];
                delta[j] = (first[j] == -1) ? 1 : 0; // delta[j]=1 if j is a leaf
                for (; j != -1 && first[j] == -1; j = parent[j])
                {
                    first[j] = k;
                }
            }
            ATp = AT.p;
            ATi = AT.i;

            if (ata) // Init ata
            {
                head = new int[n + 1];
                next = new int[m];

                Array.Copy(w, head, n);
                head[n] = -1;

                for (k = 0; k < n; k++)
                {
                    w[post[k]] = k; // invert post
                }
                for (i = 0; i < m; i++)
                {
                    for (k = n, p = ATp[i]; p < ATp[i + 1]; p++)
                    {
                        k = Math.Min(k, w[ATi[p]]);
                    }
                    next[i] = head[k]; // place row i in linked list k
                    head[k] = i;
                }
            }

            for (i = 0; i < n; i++) w[i] = i; // each node in its own set
            for (k = 0; k < n; k++)
            {
                j = post[k]; // j is the kth node in postordered etree
                if (parent[j] != -1) delta[parent[j]]--; // j is not a root

                //int HEAD(k,j) (ata ? head [k] : j)
                //int NEXT(J)   (ata ? next [J] : -1)
                for (J = (ata ? head[k] : j); J != -1; J = (ata ? next[J] : -1)) // J=j for LL'=A case
                {
                    for (p = ATp[J]; p < ATp[J + 1]; p++)
                    {
                        i = ATi[p];
                        q = IsLeaf(i, j, first, maxfirst, prevleaf, w, ref jleaf);
                        if (jleaf >= 1) delta[j]++; // A(i,j) is in skeleton
                        if (jleaf == 2) delta[q]--; // account for overlap in q
                    }
                }
                if (parent[j] != -1) w[j] = parent[j];
            }
            for (j = 0; j < n; j++) // sum up delta's of each child
            {
                if (parent[j] != -1) colcount[parent[j]] += colcount[j];
            }
            return colcount; // success: free workspace
        }

        /// <summary>
        /// Determines if j is a leaf of the skeleton matrix and find lowest common
        /// ancestor (lca).
        /// </summary>
        /// <param name=""></param>
        /// <returns>Least common ancestor (jprev,j)</returns>
        static int IsLeaf(int i, int j, int[] first, int[] maxfirst, int[] prevleaf,
            int[] ancestor, ref int jleaf)
        {
            int q, s, sparent, jprev;
            if (first == null || maxfirst == null || prevleaf == null || ancestor == null)
            {
                return (-1);
            }
            jleaf = 0;
            if (i <= j || first[j] <= maxfirst[i])
            {
                return (-1); // j not a leaf
            }
            maxfirst[i] = first[j]; // update max first[j] seen so far
            jprev = prevleaf[i]; // jprev = previous leaf of ith subtree
            prevleaf[i] = j;
            jleaf = (jprev == -1) ? 1 : 2; // j is first or subsequent leaf
            if (jleaf == 1) return (i); // if 1st leaf, q = root of ith subtree
            for (q = jprev; q != ancestor[q]; q = ancestor[q]) ;
            for (s = jprev; s != q; s = sparent)
            {
                sparent = ancestor[s]; // path compression
                ancestor[s] = q;
            }
            return (q);
        }

        /// <summary>
        /// Permutes a vector, x=P*b, for dense vectors x and b.
        /// </summary>
        /// <param name="p">permutation vector, p=null denotes identity</param>
        /// <param name="b">input vector</param>
        /// <param name="x">output vector, x=P*b</param>
        /// <param name="n">length of p, b and x</param>
        /// <returns>true if successful, false otherwise</returns>
        public static bool Permute<T>(int[] p, T[] b, T[] x, int n)
        {
            int k;
            if (x == null || b == null) return false; // check inputs
            for (k = 0; k < n; k++) x[k] = b[p != null ? p[k] : k];
            return true;
        }

        /// <summary>
        /// Permutes a vector, x = P'b.
        /// </summary>
        /// <param name="p">permutation vector, p=null denotes identity</param>
        /// <param name="b">input vector</param>
        /// <param name="x">output vector, x = P'b</param>
        /// <param name="n">length of p, b, and x</param>
        /// <returns>true if successful, false on error</returns>
        public static bool InversePermute<T>(int[] p, T[] b, T[] x, int n)
        {
            int k;
            if (x == null || b == null) return false; // check inputs
            for (k = 0; k < n; k++) x[p != null ? p[k] : k] = b[k];
            return true;
        }

        /// <summary>
        /// Returns a random permutation vector, the identity perm, or p = n-1:-1:0.
        /// seed = -1 means p = n-1:-1:0. seed = 0 means p = identity. otherwise p =
        /// random permutation.
        /// </summary>
        /// <param name="n">length of p</param>
        /// <param name="seed">0: natural, -1: reverse, random p oterwise</param>
        /// <returns>p, null on error or for natural order</returns>
        public static int[] RandomPermute(int n, int seed)
        {
            int k, j, t;
            int[] p;
            if (seed == 0) return null; // return p = NULL (identity)
            p = new int[n]; // allocate result
            if (p == null) return null; // out of memory
            for (k = 0; k < n; k++) p[k] = n - k - 1;
            if (seed == -1) return (p); // return reverse permutation
            var rand = new Random(seed); // get new random number seed
            for (k = 0; k < n; k++)
            {
                j = k + (rand.Next() % (n - k)); // j = rand integer in range k to n-1
                t = p[j]; // swap p[k] and p[j]
                p[j] = p[k];
                p[k] = t;
            }
            return (p);
        }

        /// <summary>
        /// Inverts a permutation vector. Returns pinv[i] = k if p[k] = i on input.
        /// </summary>
        /// <param name="p">a permutation vector if length n</param>
        /// <param name="n">length of p</param>
        /// <returns>pinv, null on error</returns>
        public static int[] InversePermutation(int[] p, int n)
        {
            int k;
            int[] pinv;
            if (p == null) return null; // p = NULL denotes identity
            pinv = new int[n]; // allocate result
            if (pinv == null) return null; // out of memory
            for (k = 0; k < n; k++) pinv[p[k]] = k; // invert the permutation
            return (pinv); // return result
        }

        /// <summary>
        /// p [0..n] = cumulative sum of c [0..n-1], and then copy p [0..n-1] into c
        /// </summary>
        /// <param name="p">size n+1, cumulative sum of c</param>
        /// <param name="c">size n, overwritten with p [0..n-1] on output</param>
        /// <param name="n">length of c</param>
        /// <returns>sum (c), null on error</returns>
        public static int CumulativeSum(int[] p, int[] c, int n)
        {
            int i, nz = 0, nz2 = 0;
            if (p == null || c == null) return (-1); // check inputs
            for (i = 0; i < n; i++)
            {
                p[i] = nz;
                nz += c[i];
                nz2 += c[i]; // also in double to avoid int overflow
                c[i] = p[i]; // also copy p[0..n-1] back into c[0..n-1]
            }
            p[n] = nz;
            return nz2; // return sum (c [0..n-1])
        }
    }
}
