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
using System.Linq;
using System.Collections.Generic;

namespace Meshes.Generic
{
    public partial class Mesh<TEdgeTraits, TFaceTraits, THalfedgeTraits, TVertexTraits>
    {
        /// <summary>
        /// A vertex of the mesh.
        /// </summary>
        [Serializable]
        public class Vertex
        {
            #region Fields
            /// <summary>
            /// The custom traits for the vertex.
            /// </summary>
            public TVertexTraits Traits;

            //private Halfedge halfedge;
            private Mesh<TEdgeTraits, TFaceTraits, THalfedgeTraits, TVertexTraits> mesh;
            private int index;
            #endregion

            #region Constructors
            /// <summary>
            /// Creates a vertex with traits set to their default value.
            /// </summary>
            internal Vertex() { }

            /// <summary>
            /// Creates a vertex with the given traits.
            /// </summary>
            /// <param name="vertexTraits">Traits for this vertex.</param>
            internal Vertex(TVertexTraits vertexTraits)
            {
                Traits = vertexTraits;
            }
            #endregion

            #region Properties
            /// <summary>
            /// The number of edges connected to the vertex.
            /// </summary>
            public int EdgeCount
            {
                get { return this.Edges.Count(); }
            }

            /// <summary>
            /// The number of faces with the vertex.
            /// </summary>
            public int FaceCount
            {
                get { return this.Faces.Count(); }
            }

            /// <summary>
            /// A halfedge that originates from the vertex.
            /// </summary>
            public Halfedge Halfedge
            {
                get;
                internal set;
            }

            /// <summary>
            /// The number of halfedges from the vertex.
            /// </summary>
            public int HalfedgeCount
            {
                get{ return this.Halfedges.Count(); }
            }

            /// <summary>
            /// The index of this in the mesh's interal vertex list.
            /// </summary>
            public int Index
            {
                get { return index; }
                internal set { index = value; }
            }

            /// <summary>
            /// The mesh the vertex belongs to.
            /// </summary>
            public Mesh<TEdgeTraits, TFaceTraits, THalfedgeTraits, TVertexTraits> Mesh
            {
                get { return mesh; }
                internal set { mesh = value; }
            }

            /// <summary>
            /// Checks if the vertex is on the boundary of the mesh.
            /// </summary>
            public bool OnBoundary
            {
                get
                {
                    if (this.Halfedge == null)
                    {
                        return true;
                    }

                    // Search adjacent faces for any that are null
                    foreach (Halfedge h in this.Halfedges)
                    {
                        if (h.OnBoundary)
                        {
                            return true;
                        }
                    }

                    return false;
                }
            }

            /// <summary>
            /// The number of vertices in the one ring neighborhood.
            /// </summary>
            public int VertexCount()
            {
                int count = 0;
                foreach (Vertex v in this.Vertices)
                {
                    ++count;
                }
                return count;
            }
            
            #endregion

            #region Iterators
            /// <summary>
            /// An iterator for edges connected to the vertex.
            /// </summary>
            public IEnumerable<Edge> Edges
            {
                get
                {                    
                    foreach (Halfedge h in this.Halfedges)
                    {
                        yield return h.Edge;
                    }
                }
            }

            /// <summary>
            /// An iterator for the faces with the vertex.
            /// </summary>
            public IEnumerable<Face> Faces
            {
                get
                {
                    foreach (Halfedge h in this.Halfedges)
                    {
                        if (h.Face != null)
                        {
                            yield return h.Face;
                        }
                    }
                }
            }

            /// <summary>
            /// An iterator for the halfedges originating from the vertex.
            /// Notice: the return order is CCW
            /// </summary>
            public IEnumerable<Halfedge> Halfedges
            {
                get
                {
                    Halfedge h = this.Halfedge;

                    if (h != null)
                    {
                        do
                        {
                            yield return h;
                            // use this for CW order
                            //h = h.Opposite.Next;                            
                            // use this for CCW order
                            h = h.Previous.Opposite;
                        } while (h != this.Halfedge);
                    }
                }
            }

            /// <summary>
            /// An iterator for the vertices in the one ring neighborhood.
            /// </summary>
            public IEnumerable<Vertex> Vertices
            {
                get
                {
                    foreach (Halfedge h in this.Halfedges)
                    {
                        yield return h.ToVertex;
                    }
                }
            }
            #endregion

            #region Methods
            /// <summary>
            /// Searches for the edge associated with the specified vertex.
            /// </summary>
            /// <param name="vertex">A vertex sharing an edge with this vertex.</param>
            /// <returns>The edge if it is found, otherwise null.</returns>
            public Edge FindEdgeTo(Vertex vertex)
            {
                foreach (Halfedge h in this.Halfedges)
                {
                    if (h.ToVertex == vertex)
                    {
                        return h.Edge;
                    }
                }
                return null;
            }

            /// <summary>
            /// Searches for the halfedge pointing to the specified face from this vertex.
            /// </summary>
            /// <param name="face">The face the halfedge to find points to.</param>
            /// <returns>The halfedge if it is found, otherwise null.</returns>
            public Halfedge FindHalfedgeTo(Face face)
            {
                foreach (Halfedge h in this.Halfedges)
                {
                    if (h.Face == face)
                    {
                        return h;
                    }
                }
                return null;
            }

            /// <summary>
            /// Searches for a halfedge pointing to a vertex from this vertex.
            /// </summary>
            /// <param name="vertex">A vertex pointed to by the halfedge to search for.</param>
            /// <returns>The halfedge from this vertex to the specified vertex. If none exists, returns null.</returns>
            public Halfedge FindHalfedgeTo(Vertex vertex)
            {
                foreach (Halfedge h in this.Halfedges)
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
