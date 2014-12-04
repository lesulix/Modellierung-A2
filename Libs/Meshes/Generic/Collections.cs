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
using System.Diagnostics;

namespace Meshes.Generic
{
    public partial class Mesh<TEdgeTraits, TFaceTraits, THalfedgeTraits, TVertexTraits>
    {
        #region EdgeCollection class
        /// <summary>
        /// Type allowing edges to be accessed like an array.
        /// </summary>
        [Serializable]
        public class EdgeCollection : IEnumerable<Edge>
        {
            #region Fields
            readonly Mesh<TEdgeTraits, TFaceTraits, THalfedgeTraits, TVertexTraits> mesh;
            #endregion

            #region Constructors
            internal EdgeCollection(Mesh<TEdgeTraits, TFaceTraits, THalfedgeTraits, TVertexTraits> m)
            {
                mesh = m;
            }
            #endregion

            #region Properties
            /// <summary>
            /// Accesses the edges in a mesh.
            /// </summary>
            /// <param name="index">The index of the edge.</param>
            /// <returns>The indexed <see cref="Edge"/>.</returns>
            public Edge this[int index]
            {
                get
                {
                    return mesh.edges[index];
                }
            }

            /// <summary>
            /// The number of edges in the mesh.
            /// </summary>
            public int Count
            {
                get
                {
                    return mesh.edges.Count;
                }
            }
            #endregion

            #region Iterators
            /// <summary>
            /// Provides an enumerator for the edges of the mesh.
            /// </summary>
            /// <returns>An edge of the mesh.</returns>
            public IEnumerator<Edge> GetEnumerator()
            {
                foreach (Edge e in mesh.edges)
                {
                    yield return e;
                }
            }

            /// <summary>
            /// Useless IEnumerable.GetEnumerator() implementation.
            /// </summary>
            System.Collections.IEnumerator System.Collections.IEnumerable.GetEnumerator()
            {
                return GetEnumerator();
            }
            #endregion
        }
        #endregion

        #region FaceCollection class
        /// <summary>
        /// Type allowing faces to be accessed like an array.
        /// </summary>
        [Serializable]
        public class FaceCollection : IEnumerable<Face>
        {
            #region Fields
            readonly Mesh<TEdgeTraits, TFaceTraits, THalfedgeTraits, TVertexTraits> mesh;
            #endregion

            #region Constructors
            internal FaceCollection(Mesh<TEdgeTraits, TFaceTraits, THalfedgeTraits, TVertexTraits> m)
            {
                mesh = m;
            }
            #endregion

            #region Properties
            /// <summary>
            /// Accesses the faces in a mesh.
            /// </summary>
            /// <param name="index">The index of the face.</param>
            /// <returns>The indexed <see cref="Face"/>.</returns>
            public Face this[int index]
            {
                get
                {
                    return mesh.faces[index];
                }
            }

            /// <summary>
            /// The number of faces in the mesh.
            /// </summary>
            public int Count
            {
                get
                {
                    return mesh.faces.Count;
                }
            }
            #endregion

            #region Iterators
            /// <summary>
            /// Provides an enumerator for the faces of the mesh.
            /// </summary>
            /// <returns>A face of the mesh.</returns>
            public IEnumerator<Face> GetEnumerator()
            {
                foreach (Face f in mesh.faces)
                {
                    yield return f;
                }
            }

            /// <summary>
            /// Useless IEnumerable.GetEnumerator() implementation.
            /// </summary>
            System.Collections.IEnumerator System.Collections.IEnumerable.GetEnumerator()
            {
                return GetEnumerator();
            }
            #endregion

            #region Methods
            /// <summary>
            /// Adds a face to the mesh with default face traits.
            /// </summary>
            /// <param name="faceVertices">The vertices of the face in counterclockwise order.</param>
            /// <returns>The face created by this method.</returns>
            /// <exception cref="BadTopologyException">
            /// Thrown when fewer than three vertices are given or the vertices cannot form a valid face.
            /// </exception>
            /// <exception cref="ArgumentNullException">Thrown when a null vertex is given.</exception>
            public Face Add(params Vertex[] faceVertices)
            {
                return Add(default(TFaceTraits), faceVertices);
            }

            /// <summary>
            /// Adds a face to the mesh with the specified face traits.
            /// </summary>
            /// <param name="faceTraits">The custom traits for the face to add to the mesh.</param>
            /// <param name="faceVertices">The vertices of the face in counterclockwise order.</param>
            /// <returns>The face created by this method.</returns>
            /// <exception cref="BadTopologyException">
            /// Thrown when fewer than three vertices are given or the vertices cannot form a valid face.
            /// </exception>
            /// <exception cref="ArgumentNullException">Thrown when a null vertex is given.</exception>
            public Face Add(TFaceTraits faceTraits, params Vertex[] faceVertices)
            {
                if (mesh.trianglesOnly)
                {
                    return AddTriangles(faceTraits, faceVertices)[0];
                }
                else
                {
                    return CreateFace(faceTraits, faceVertices);
                }
            }

            /// <summary>
            /// Adds triangular faces to the mesh with default face traits.
            /// </summary>
            /// <param name="faceVertices">The vertices of the faces in counterclockwise order.</param>
            /// <returns>An array of faces created by this method.</returns>
            /// <exception cref="BadTopologyException">
            /// Thrown when fewer than three vertices are given or the vertices cannot form a valid face.
            /// </exception>
            /// <exception cref="ArgumentNullException">Thrown when a null vertex is given.</exception>
            public Face[] AddTriangles(params Vertex[] faceVertices)
            {
                return AddTriangles(default(TFaceTraits), faceVertices);
            }

            /// <summary>
            /// Adds triangular faces to the mesh with the specified face traits.
            /// </summary>
            /// <param name="faceTraits">The custom traits for the faces to add to the mesh.</param>
            /// <param name="faceVertices">The vertices of the faces in counterclockwise order.</param>
            /// <returns>An array of faces created by this method.</returns>
            /// <exception cref="BadTopologyException">
            /// Thrown when fewer than three vertices are given or the vertices cannot form a valid face.
            /// </exception>
            /// <exception cref="ArgumentNullException">Thrown when a null vertex is given.</exception>
            public Face[] AddTriangles(TFaceTraits faceTraits, params Vertex[] faceVertices)
            {
                int n = faceVertices.Length;

                // Require at least 3 vertices
                if (n < 3)
                {
                    throw new BadTopologyException("Cannot create a polygon with fewer than three vertices.");
                }

                Face[] addedFaces = new Face[n - 2];

                // Triangulate the face
                for (int i = 0; i < n - 2; ++i)
                {
                    addedFaces[i] = CreateFace(faceTraits, faceVertices[0], faceVertices[i + 1], faceVertices[i + 2]);
                }

                return addedFaces;
            }

            /// <summary>
            /// Adds a face to the mesh with the specified face traits.
            /// </summary>
            /// <param name="faceTraits">The custom traits for the face to add to the mesh.</param>
            /// <param name="faceVertices">The vertices of the face in counterclockwise order.</param>
            /// <returns>The face created by this method.</returns>
            /// <exception cref="BadTopologyException">
            /// Thrown when fewer than three vertices are given or the vertices cannot form a valid face.
            /// </exception>
            /// <exception cref="ArgumentNullException">Thrown when a null vertex is given.</exception>
            private Face CreateFace(TFaceTraits faceTraits, params Vertex[] faceVertices)
            {
                int n = faceVertices.Length;

                // Require at least 3 vertices
                if (n < 3)
                {
                    throw new BadTopologyException("Cannot create a polygon with fewer than three vertices.");
                }

                Edge e;
                Face f;
                Halfedge[] faceHalfedges = new Halfedge[n];
                bool[] isNewEdge = new bool[n], isUsedVertex = new bool[n];

                // Make sure input is (mostly) acceptable before making any changes to the mesh
                for (int i = 0; i < n; ++i)
                {
                    int j = (i + 1) % n;

                    if (faceVertices[i] == null)
                    {
                        throw new ArgumentNullException("Can't add a null vertex to a face.");
                    }
                    if (!faceVertices[i].OnBoundary)
                    {
                        throw new BadTopologyException("Can't add an edge to a vertex on the interior of a mesh.");
                    }

                    // Find existing halfedges for this face
                    faceHalfedges[i] = faceVertices[i].FindHalfedgeTo(faceVertices[j]);
                    isNewEdge[i] = (faceHalfedges[i] == null);
                    isUsedVertex[i] = (faceVertices[i].Halfedge != null);

                    if (!isNewEdge[i] && !faceHalfedges[i].OnBoundary)
                    {
                        throw new BadTopologyException("Can't add more than two faces to an edge.");
                    }
                }

                // Create face
                f = new Face(faceTraits);
                mesh.AppendToFaceList(f);

                // Create new edges
                for (int i = 0; i < n; ++i)
                {
                    int j = (i + 1) % n;

                    if (isNewEdge[i])
                    {
                        // Create new edge
                        e = new Edge();
                        mesh.AppendToEdgeList(e);

                        // Create new halfedges
                        faceHalfedges[i] = new Halfedge();
                        mesh.AppendToHalfedgeList(faceHalfedges[i]);

                        faceHalfedges[i].Opposite = new Halfedge();
                        mesh.AppendToHalfedgeList(faceHalfedges[i].Opposite);

                        // Connect opposite halfedge to inner halfedge
                        faceHalfedges[i].Opposite.Opposite = faceHalfedges[i];

                        // Connect edge to halfedges
                        e.Halfedge0 = faceHalfedges[i];

                        // Connect halfedges to edge
                        faceHalfedges[i].Edge = e;
                        faceHalfedges[i].Opposite.Edge = e;

                        // Connect halfedges to vertices
                        faceHalfedges[i].ToVertex = faceVertices[j];
                        faceHalfedges[i].Opposite.ToVertex = faceVertices[i];

                        // Connect vertex to outgoing halfedge if it doesn't have one yet
                        if (faceVertices[i].Halfedge == null)
                        {
                            faceVertices[i].Halfedge = faceHalfedges[i];
                        }
                    }

                    if (faceHalfedges[i].Face != null)
                    {
                        throw new BadTopologyException("An inner halfedge already has a face assigned to it.");
                    }

                    // Connect inner halfedge to face
                    faceHalfedges[i].Face = f;

#if DEBUG
                    Debug.Assert(faceHalfedges[i].FromVertex == faceVertices[i] && faceHalfedges[i].ToVertex == faceVertices[j]);
#endif
                }

                // Connect next/previous halfedges
                for (int i = 0; i < n; ++i)
                {
                    int j = (i + 1) % n;

                    // Outer halfedges
                    if (isNewEdge[i] && isNewEdge[j] && isUsedVertex[j])  // Both edges are new and vertex has faces connected already
                    {
                        Halfedge closeHalfedge = null;

                        // Find the closing halfedge of the first available opening
                        foreach (Halfedge h in faceVertices[j].Halfedges)
                        {
                            if (h.Face == null)
                            {
                                closeHalfedge = h;
                                break;
                            }
                        }

                        Debug.Assert(closeHalfedge != null);

                        Halfedge openHalfedge = closeHalfedge.Previous;

                        // Link new outer halfedges into this opening
                        faceHalfedges[i].Opposite.Previous = openHalfedge;
                        openHalfedge.Next = faceHalfedges[i].Opposite;
                        faceHalfedges[j].Opposite.Next = closeHalfedge;
                        closeHalfedge.Previous = faceHalfedges[j].Opposite;
                    }
                    else if (isNewEdge[i] && isNewEdge[j])  // Both edges are new
                    {
                        faceHalfedges[i].Opposite.Previous = faceHalfedges[j].Opposite;
                        faceHalfedges[j].Opposite.Next = faceHalfedges[i].Opposite;
                    }
                    else if (isNewEdge[i] && !isNewEdge[j])  // This is new, next is old
                    {
                        faceHalfedges[i].Opposite.Previous = faceHalfedges[j].Previous;
                        faceHalfedges[j].Previous.Next = faceHalfedges[i].Opposite;
                    }
                    else if (!isNewEdge[i] && isNewEdge[j])  // This is old, next is new
                    {
                        faceHalfedges[i].Next.Previous = faceHalfedges[j].Opposite;
                        faceHalfedges[j].Opposite.Next = faceHalfedges[i].Next;
                    }
                    else if (!isNewEdge[i] && !isNewEdge[j] && faceHalfedges[i].Next != faceHalfedges[j])  // Relink faces before adding new edges if they are in the way of a new face
                    {
                        Halfedge closeHalfedge = faceHalfedges[i].Opposite;

                        // Find the closing halfedge of the opening opposite the opening halfedge i is on
                        do
                        {
                            closeHalfedge = closeHalfedge.Previous.Opposite;
                        } 
                        while (closeHalfedge.Face != null && closeHalfedge != faceHalfedges[j] && closeHalfedge != faceHalfedges[i].Opposite);

                        if (closeHalfedge == faceHalfedges[j] || closeHalfedge == faceHalfedges[i].Opposite)
                        {
                            throw new BadTopologyException("Unable to find an opening to relink an existing face.");
                        }

                        Halfedge openHalfedge = closeHalfedge.Previous;

                        // Remove group of faces between two openings, close up gap to form one opening
                        openHalfedge.Next = faceHalfedges[i].Next;
                        faceHalfedges[i].Next.Previous = openHalfedge;

                        // Insert group of faces into target opening
                        faceHalfedges[j].Previous.Next = closeHalfedge;
                        closeHalfedge.Previous = faceHalfedges[j].Previous;
                    }

                    // Inner halfedges
                    faceHalfedges[i].Next = faceHalfedges[j];
                    faceHalfedges[j].Previous = faceHalfedges[i];
                }

                // Connect face to an inner halfedge
                f.Halfedge = faceHalfedges[0];

                return f;
            }
            #endregion
        }
        #endregion

        #region HalfedgeCollection class
        /// <summary>
        /// Type allowing halfedges to be accessed like an array.
        /// </summary>
        [Serializable]
        public class HalfedgeCollection : IEnumerable<Halfedge>
        {
            #region Fields
            readonly Mesh<TEdgeTraits, TFaceTraits, THalfedgeTraits, TVertexTraits> mesh;
            #endregion

            #region Constructors
            internal HalfedgeCollection(Mesh<TEdgeTraits, TFaceTraits, THalfedgeTraits, TVertexTraits> m)
            {
                mesh = m;
            }
            #endregion

            #region Properties
            /// <summary>
            /// Accesses the halfedges in a mesh.
            /// </summary>
            /// <param name="index">The index of the halfedge.</param>
            /// <returns>The indexed <see cref="Halfedge"/>.</returns>
            public Halfedge this[int index]
            {
                get { return mesh.halfedges[index]; }
            }

            /// <summary>
            /// The number of halfedges in the mesh.
            /// </summary>
            public int Count
            {
                get { return mesh.halfedges.Count; }
            }
            #endregion

            #region Iterators
            /// <summary>
            /// Provides an enumerator for the halfedges of the mesh.
            /// </summary>
            /// <returns>A halfedge of the mesh.</returns>
            public IEnumerator<Halfedge> GetEnumerator()
            {
                foreach (Halfedge h in mesh.halfedges)
                {
                    yield return h;
                }
            }

            /// <summary>
            /// Useless IEnumerable.GetEnumerator() implementation.
            /// </summary>
            System.Collections.IEnumerator System.Collections.IEnumerable.GetEnumerator()
            {
                return GetEnumerator();
            }
            #endregion
        }
        #endregion

        #region VertexCollection class
        /// <summary>
        /// Type allowing vertices to be accessed like an array.
        /// </summary>
        [Serializable]
        public class VertexCollection : IEnumerable<Vertex>
        {
            #region Fields
            readonly Mesh<TEdgeTraits, TFaceTraits, THalfedgeTraits, TVertexTraits> mesh;
            #endregion

            #region Constructors
            internal VertexCollection(Mesh<TEdgeTraits, TFaceTraits, THalfedgeTraits, TVertexTraits> m)
            {
                mesh = m;
            }
            #endregion

            #region Properties
            /// <summary>
            /// Accesses the vertices in a mesh.
            /// </summary>
            /// <param name="index">The index of the vertex.</param>
            /// <returns>The indexed <see cref="Vertex"/>.</returns>
            public Vertex this[int index]
            {
                get { return mesh.vertices[index]; }
            }

            /// <summary>
            /// The number of vertices in the mesh.
            /// </summary>
            public int Count
            {
                get { return mesh.vertices.Count; }
            }
            #endregion

            #region Iterators
            /// <summary>
            /// Provides an enumerator for the vertices of the mesh.
            /// </summary>
            /// <returns>A vertex of the mesh.</returns>
            public IEnumerator<Vertex> GetEnumerator()
            {
                foreach (Vertex v in mesh.vertices)
                {
                    yield return v;
                }
            }

            /// <summary>
            /// Useless IEnumerable.GetEnumerator() implementation.
            /// </summary>
            System.Collections.IEnumerator System.Collections.IEnumerable.GetEnumerator()
            {
                return GetEnumerator();
            }
            #endregion

            #region Methods
            /// <summary>
            /// Adds a vertex to the mesh.
            /// </summary>
            /// <returns>The vertex created by this method.</returns>
            public Vertex Add()
            {
                Vertex v = new Vertex();
                mesh.AppendToVertexList(v);
                return v;
            }

            /// <summary>
            /// Adds a vertex to the mesh with the specified vertex traits.
            /// </summary>
            /// <param name="vertexTraits">The custom traits for the vertex to add to the mesh.</param>
            /// <returns>The vertex created by this method.</returns>
            public Vertex Add(TVertexTraits vertexTraits)
            {
                Vertex v = new Vertex(vertexTraits);
                mesh.AppendToVertexList(v);
                return v;
            }
            #endregion
        }
        #endregion
    }
}