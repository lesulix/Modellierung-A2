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
using System.Diagnostics;
using System.IO;
using System.Runtime.Serialization;
using System.Runtime.Serialization.Formatters;
using System.Runtime.Serialization.Formatters.Binary;
using System.Text;

namespace Meshes.Generic
{
    /// <summary>
    /// A halfedge mesh data structure that stores mesh topology.
    /// </summary>
    /// <typeparam name="TEdgeTraits">The custom traits type for the edges.</typeparam>
    /// <typeparam name="TFaceTraits">The custom traits type for the faces.</typeparam>
    /// <typeparam name="THalfedgeTraits">The custom traits type for the halfeges.</typeparam>
    /// <typeparam name="TVertexTraits">The custom traits type for the vertices.</typeparam>
    /// <remarks>
    /// The trait classes allow the user to specify custom trait types on mesh elements.
    /// </remarks>
    [Serializable]
    public partial class Mesh<TEdgeTraits, TFaceTraits, THalfedgeTraits, TVertexTraits>
    {
        #region Fields
        List<Edge> edges = new List<Edge>();
        List<Face> faces = new List<Face>();
        List<Halfedge> halfedges = new List<Halfedge>();
        List<Vertex> vertices = new List<Vertex>();

        /// <summary>
        /// If set to true, adding a face with more than three vertices will cause
        /// it to be triangulated regardless of which method is used.
        /// </summary>
        protected bool trianglesOnly;
        #endregion

        #region Constructors
        /// <summary>
        /// Initializes a new, empty mesh.
        /// </summary>
        public Mesh()
        {
            Edges = new EdgeCollection(this);
            Faces = new FaceCollection(this);
            Halfedges = new HalfedgeCollection(this);
            Vertices = new VertexCollection(this);
        }

        #endregion

        #region Indexed properties
        /// <summary>
        /// The edges of the mesh.
        /// </summary>
        public readonly EdgeCollection Edges;

        /// <summary>
        /// The faces of the mesh.
        /// </summary>
        public readonly FaceCollection Faces;

        /// <summary>
        /// The halfedges of the mesh.
        /// </summary>
        public readonly HalfedgeCollection Halfedges;

        /// <summary>
        /// The vertices of the mesh.
        /// </summary>
        public readonly VertexCollection Vertices;
        #endregion

        #region Methods
        /// <summary>
        /// Adds an edge to the edge list.
        /// </summary>
        /// <param name="edge">The edge to add.</param>
        protected void AppendToEdgeList(Edge edge)
        {
            edge.Index = edges.Count;
            edges.Add(edge);
        }

        /// <summary>
        /// Adds a face to the face list.
        /// </summary>
        /// <param name="face">The face to add.</param>
        protected void AppendToFaceList(Face face)
        {
            face.Index = faces.Count;
            faces.Add(face);
        }

        /// <summary>
        /// Adds a halfedge to the halfedge list.
        /// </summary>
        /// <param name="halfedge">The halfedge to add.</param>
        protected void AppendToHalfedgeList(Halfedge halfedge)
        {
            halfedge.Index = halfedges.Count;
            halfedges.Add(halfedge);
        }

        /// <summary>
        /// Adds a vertex to the vertex list.
        /// </summary>
        /// <param name="vertex">The vertex to add.</param>
        protected void AppendToVertexList(Vertex vertex)
        {
            vertex.Index = vertices.Count;
            vertex.Mesh = this;
            vertices.Add(vertex);
        }

        /// <summary>
        /// Removes all elements from the mesh.
        /// </summary>
        /// <remarks>
        /// This is not exposed publicly because it's too easy to cause damage when using dynamic traits.
        /// </remarks>
        protected virtual void Clear()
        {
            edges.Clear();
            faces.Clear();
            halfedges.Clear();
            vertices.Clear();
        }

        /// <summary>
        /// Saves the mesh object to the specified stream.
        /// </summary>
        /// <param name="stream">The <see cref="Stream"/> where the mesh will be saved.</param>
        public void Save(Stream stream)
        {
            IFormatter formatter = new BinaryFormatter();
            formatter.Serialize(stream, this);
        }

        /// <summary>
        /// Saves the mesh object to the specified file.
        /// </summary>
        /// <param name="fileName">The location of the file where you want to save the mesh.</param>
        public void Save(string fileName)
        {
            if (File.Exists(fileName))
            {
                File.Delete(fileName);
            }

            using (Stream stream = File.OpenWrite(fileName))
            {
                Save(stream);
            }
        }

        /// <summary>
        /// Returns a string representing the connections of vertices for each face in the mesh.
        /// </summary>
        /// <returns>The traits for each vertex of each face on a line of the string.</returns>
        public override string ToString()
        {
            StringBuilder s = new StringBuilder();

            foreach (Face f in Faces)
            {
                foreach (Vertex v in f.Vertices)
                {
                    s.Append(v.Traits.ToString());
                    s.Append(" -> ");
                }
                s.Append(Environment.NewLine);
            }

            return s.ToString();
        }

        /// <summary>
        /// Triangulates a mesh.
        /// </summary>
        /// <returns>A triangulated copy of the mesh.</returns>
        /// <remarks>
        /// Any edge and halfedge traits are not copied to the new mesh. Face traits are copied
        /// to all faces triangulated from a face.
        /// </remarks>
        public Mesh<TEdgeTraits, TFaceTraits, THalfedgeTraits, TVertexTraits> TriangularCopy()
        {
            Mesh<TEdgeTraits, TFaceTraits, THalfedgeTraits, TVertexTraits> triangulatedMesh = new Mesh<TEdgeTraits, TFaceTraits, THalfedgeTraits, TVertexTraits>();
            Dictionary<Vertex, Vertex> newVertices = new Dictionary<Vertex, Vertex>();

            foreach (Vertex v in Vertices)
            {
                newVertices[v] = triangulatedMesh.Vertices.Add(v.Traits);
            }

            foreach (Face f in Faces)
            {
                Vertex[] vertices = new Vertex[f.VertexCount];
                int i = 0;
                foreach (Vertex v in f.Vertices)
                {
                    vertices[i] = newVertices[v];
                    ++i;
                }
                triangulatedMesh.Faces.AddTriangles(f.Traits, vertices);
            }

            return triangulatedMesh;
        }

        /// <summary>
        /// Trims internal data structures to their current size.
        /// </summary>
        /// <remarks>
        /// Call this method to prevent excess memory usage when the mesh is done being built.
        /// </remarks>
        public void TrimExcess()
        {
            edges.TrimExcess();
            faces.TrimExcess();
            halfedges.TrimExcess();
            vertices.TrimExcess();
        }

        /// <summary>
        /// Checks halfedge connections to verify that a valid mesh was constructed.
        /// </summary>
        /// <remarks>
        /// Checking for proper topology in every case when a face is added would slow down
        /// mesh construction significantly, so this method may be called once when a mesh
        /// is complete to ensure that topology is manifold (with boundary).
        /// If the mesh is non-manifold, a BadTopologyException will be thrown.
        /// </remarks>
        public void VerifyTopology()
        {
            foreach (Halfedge h in halfedges)
            {
#if DEBUG
                Debug.Assert(h == h.Opposite.Opposite);
                Debug.Assert(h.Edge == h.Opposite.Edge);
                Debug.Assert(h.ToVertex.Halfedge.Opposite.ToVertex == h.ToVertex);
#endif

                if (h.Previous.Next != h)
                {
                    throw new BadTopologyException("A halfedge's previous next is not itself.");
                }

                if (h.Next.Previous != h)
                {
                    throw new BadTopologyException("A halfedge's next previous is not itself.");
                }

                if (h.Next.Face != h.Face)
                {
                    throw new BadTopologyException("Adjacent halfedges do not belong to the same face.");
                }

                // Make sure each halfedge is reachable from the vertex it originates from
                bool found = false;

                foreach (Halfedge hIt in h.FromVertex.Halfedges)
                {
                    if (hIt == h)
                    {
                        found = true;
                        break;
                    }
                }

                if (found == false)
                {
                    throw new BadTopologyException("A halfedge is not reachable from the vertex it originates from.");
                }
            }
        }

        /// <summary>
        /// Creates a new, identical mesh.
        /// </summary>
        /// <returns>A deep copy of the mesh.</returns>
        public virtual Mesh<TEdgeTraits, TFaceTraits, THalfedgeTraits, TVertexTraits> MemoryCopy()
        {
            // Use serialization to create a deep copy
            using (MemoryStream ms = new MemoryStream(Vertices.Count * 300))
            {
                BinaryFormatter bf = new BinaryFormatter();
                bf.Serialize(ms, this);
                ms.Seek(0, SeekOrigin.Begin);
                return (Mesh<TEdgeTraits, TFaceTraits, THalfedgeTraits, TVertexTraits>)bf.Deserialize(ms);
            }
        }
        #endregion

        #region Static methods

        /// <summary>
        /// Determines if two faces are adjacent.
        /// </summary>
        /// <param name="faceA">One of the faces to search for.</param>
        /// <param name="faceB">The other face to search for.</param>
        /// <returns>True if the faces are adjacent, false if they are not.</returns>
        public static bool FacesShareEdge(Face faceA, Face faceB)
        {
            foreach (Face f in faceA.Faces)
            {
                if (f == faceB)
                {
                    return true;
                }
            }

            return false;
        }

        /// <summary>
        /// Creates a mesh from the specified file.
        /// </summary>
        /// <param name="fileName">A string that contains the name of the file from which to create the HalfedgeMesh object.</param>
        /// <returns>The mesh object this method creates.</returns>
        public static Mesh<TEdgeTraits, TFaceTraits, THalfedgeTraits, TVertexTraits> FromFile(string fileName)
        {
            Mesh<TEdgeTraits, TFaceTraits, THalfedgeTraits, TVertexTraits> m = null;
            using (Stream stream = File.OpenRead(fileName))
            {
                m = FromStream(stream);
            }
            return m;
        }

        /// <summary>
        /// Creates a mesh from the specified data stream.
        /// </summary>
        /// <param name="stream">A Stream object that contains the data for this HalfedgeMesh object.</param>
        /// <returns>The mesh object this method creates.</returns>
        public static Mesh<TEdgeTraits, TFaceTraits, THalfedgeTraits, TVertexTraits> FromStream(Stream stream)
        {
            IFormatter formatter = new BinaryFormatter();
            return (Mesh<TEdgeTraits, TFaceTraits, THalfedgeTraits, TVertexTraits>)formatter.Deserialize(stream);
        }

        #endregion
    }
}
