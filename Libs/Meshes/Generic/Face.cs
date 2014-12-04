#region License
// Copyright (c) 2006 Alexander Kolliopoulos
//
// This software is provided 'as-is', without any express or implied
// warranty. In no event will the authors be held liable for any damages
// arising from the use of this software.
//
// Permission is granted to anyone to use this software for any purpose,
// including commercial applications, and to alter it and redistribute it
// freely, subject to the following restrictions:
//
// 1. The origin of this software must not be misrepresented; you must not
//    claim that you wrote the original software. If you use this software
//    in a product, an acknowledgment in the product documentation would
//    be appreciated but is not required.
//
// 2. Altered source versions must be plainly marked as such, and must not
//    be misrepresented as being the original software.
//
// 3. This notice may not be removed or altered from any source distribution.
#endregion

using System;
using System.Collections.Generic;

namespace Meshes.Generic
{
    public partial class Mesh<TEdgeTraits, TFaceTraits, THalfedgeTraits, TVertexTraits>
    {
        /// <summary>
        /// A face of the mesh.
        /// </summary>
        [Serializable]
        public class Face
        {
            #region Fields
            /// <summary>
            /// The custom traits for the face.
            /// </summary>
            public TFaceTraits Traits;

            private Halfedge halfedge;
            private int index;
            #endregion

            #region Constructors
            /// <summary>
            /// Creates a face with traits set to their default value.
            /// </summary>
            internal Face() { }

            /// <summary>
            /// Creates a face with the given traits.
            /// </summary>
            /// <param name="faceTraits">Traits for this face.</param>
            internal Face(TFaceTraits faceTraits)
            {
                Traits = faceTraits;
            }
            #endregion

            #region Properties
            /// <summary>
            /// The number of edges for the face.
            /// </summary>
            public int EdgeCount
            {
                get
                {
                    int count = 0;
                    foreach (Edge e in Edges)
                    {
                        ++count;
                    }
                    return count;
                }
            }

            /// <summary>
            /// The number of adjacent faces for the face.
            /// </summary>
            public int FaceCount
            {
                get
                {
                    int count = 0;
                    foreach (Face f in Faces)
                    {
                        ++count;
                    }
                    return count;
                }
            }

            /// <summary>
            /// A halfedge that belongs to the face.
            /// </summary>
            public Halfedge Halfedge
            {
                get
                {
                    return halfedge;
                }
                internal set
                {
                    halfedge = value;
                }
            }

            /// <summary>
            /// The number of halfedges for the face.
            /// </summary>
            public int HalfedgeCount
            {
                get
                {
                    int count = 0;
                    foreach (Halfedge h in Halfedges)
                    {
                        ++count;
                    }
                    return count;
                }
            }

            /// <summary>
            /// The index of this in the mesh's interal face list.
            /// </summary>
            public int Index
            {
                get
                {
                    return index;
                }
                internal set
                {
                    index = value;
                }
            }

            /// <summary>
            /// The mesh the face belongs to.
            /// </summary>
            public Mesh<TEdgeTraits, TFaceTraits, THalfedgeTraits, TVertexTraits> Mesh
            {
                get
                {
                    return halfedge.Mesh;
                }
            }

            /// <summary>
            /// Checks if the face is on the boundary of the mesh.
            /// </summary>
            public bool OnBoundary
            {
                get
                {
                    foreach (Halfedge h in Halfedges)
                    {
                        if (h.Opposite.OnBoundary)
                        {
                            return true;
                        }
                    }
                    return false;
                }
            }

            /// <summary>
            /// The number of vertices for the face.
            /// </summary>
            public int VertexCount
            {
                get
                {
                    int count = 0;
                    foreach (Vertex v in Vertices)
                    {
                        ++count;
                    }
                    return count;
                }
            }
            #endregion

            #region Iterators
            /// <summary>
            /// An iterator for edges on the face.
            /// </summary>
            public IEnumerable<Edge> Edges
            {
                get
                {
                    foreach (Halfedge h in Halfedges)
                    {
                        yield return h.Edge;
                    }
                }
            }

            /// <summary>
            /// An iterator for faces adjacent to the face.
            /// </summary>
            public IEnumerable<Face> Faces
            {
                get
                {
                    foreach (Halfedge h in Halfedges)
                    {
                        yield return h.Opposite.Face;
                    }
                }
            }

            /// <summary>
            /// An iterator for halfedges for the face.
            /// </summary>
            public IEnumerable<Halfedge> Halfedges
            {
                get
                {
                    Halfedge h = this.halfedge;

                    do
                    {
                        yield return h;
                        h = h.Next;
                    } while (h != this.halfedge);
                }
            }

            /// <summary>
            /// An iterator for vertices for the face.
            /// </summary>
            public IEnumerable<Vertex> Vertices
            {
                get
                {
                    foreach (Halfedge h in Halfedges)
                    {
                        yield return h.ToVertex;
                    }
                }
            }
            #endregion

            #region Methods
            /// <summary>
            /// Searches for the edge associated with the specified face.
            /// </summary>
            /// <param name="face">A face sharing an edge with this face.</param>
            /// <returns>The edge if it is found, otherwise null.</returns>
            public Edge FindEdgeTo(Face face)
            {
                foreach (Halfedge h in Halfedges)
                {
                    if (h.Opposite.Face == face)
                    {
                        return h.Edge;
                    }
                }
                return null;
            }

            /// <summary>
            /// Searches for the halfedge pointing to the specified vertex from this face.
            /// </summary>
            /// <param name="vertex">The vertex the halfedge to find points to.</param>
            /// <returns>The halfedge if it is found, otherwise null.</returns>
            public Halfedge FindHalfedgeTo(Vertex vertex)
            {
                foreach (Halfedge h in Halfedges)
                {
                    if (h.ToVertex == vertex)
                    {
                        return h;
                    }
                }
                return null;
            }

            /// <summary>
            /// Searches for an indexed edge by iterating.
            /// </summary>
            /// <param name="index">The index of the edge to return.</param>
            /// <exception cref="ArgumentOutOfRangeException">Thrown when <paramref name="index"/> is negative or too large.</exception>
            /// <returns>The specified edge.</returns>
            public Edge GetEdge(int index)
            {
                int count = 0;
                foreach (Edge e in Edges)
                {
                    if (count == index)
                    {
                        return e;
                    }
                    ++count;
                }
                throw new ArgumentOutOfRangeException("index");
            }

            /// <summary>
            /// Searches for an indexed face by iterating.
            /// </summary>
            /// <param name="index">The index of the face to return.</param>
            /// <exception cref="ArgumentOutOfRangeException">Thrown when <paramref name="index"/> is negative or too large.</exception>
            /// <returns>The specified face.</returns>
            public Face GetFace(int index)
            {
                int count = 0;
                foreach (Face f in Faces)
                {
                    if (count == index)
                    {
                        return f;
                    }
                    ++count;
                }
                throw new ArgumentOutOfRangeException("index");
            }

            /// <summary>
            /// Searches for an indexed halfedge by iterating.
            /// </summary>
            /// <param name="index">The index of the halfedge to return.</param>
            /// <exception cref="ArgumentOutOfRangeException">Thrown when <paramref name="index"/> is negative or too large.</exception>
            /// <returns>The specified halfedge.</returns>
            public Halfedge GetHalfedge(int index)
            {
                int count = 0;
                foreach (Halfedge h in Halfedges)
                {
                    if (count == index)
                    {
                        return h;
                    }
                    ++count;
                }
                throw new ArgumentOutOfRangeException("index");
            }

            /// <summary>
            /// Searches for an indexed vertex by iterating.
            /// </summary>
            /// <param name="index">The index of the vertex to return.</param>
            /// <exception cref="ArgumentOutOfRangeException">Thrown when <paramref name="index"/> is negative or too large.</exception>
            /// <returns>The specified vertex.</returns>
            public Vertex GetVertex(int index)
            {
                int count = 0;
                foreach (Vertex v in Vertices)
                {
                    if (count == index)
                    {
                        return v;
                    }
                    ++count;
                }
                throw new ArgumentOutOfRangeException("index");
            }
            #endregion
        }
    }
}
