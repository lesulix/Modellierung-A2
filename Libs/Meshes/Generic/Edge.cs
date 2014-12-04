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
        /// An edge of the mesh.
        /// </summary>
        [Serializable]
        public class Edge
        {
            #region Fields
            /// <summary>
            /// The custom traits for the edge.
            /// </summary>
            public readonly TEdgeTraits Traits;
            #endregion

            #region Constructors
            /// <summary>
            /// Creates an edge with traits set to their default value.
            /// </summary>
            internal Edge() { }

            /// <summary>
            /// Creates an edge with the given traits.
            /// </summary>
            /// <param name="edgeTraits">Traits for this edge.</param>
            internal Edge(TEdgeTraits edgeTraits)
            {
                Traits = edgeTraits;
            }
            #endregion

            #region Properties
            /// <summary>
            /// One face adjacent to the edge, or null if there is no face.
            /// </summary>
            public Face Face0
            {
                get { return Halfedge0.Face; }
            }

            /// <summary>
            /// The other face adjacent to the edge, or null if there is no face.
            /// </summary>
            public Face Face1
            {
                get                {                    return Halfedge0.Opposite.Face;                }
            }

            /// <summary>
            /// One halfedge that corresponds to the edge.
            /// </summary>
            public Halfedge Halfedge0
            {
                get;
                internal set;
            }

            /// <summary>
            /// The other halfedge that corresponds to the edge.
            /// </summary>
            public Halfedge Halfedge1
            {
                get { return Halfedge0.Opposite; }
                internal set { Halfedge0.Opposite = value; }
            }

            /// <summary>
            /// The index of this in the mesh's interal edge list.
            /// </summary>
            public int Index
            {
                get;
                internal set;
            }

            /// <summary>
            /// Checks if the edge is on the boundary of the mesh.
            /// </summary>
            public bool OnBoundary
            {
                get
                {
                    //return halfedge.OnBoundary || halfedge.Opposite.OnBoundary;
                    return Halfedge0.FromVertex.OnBoundary && Halfedge0.ToVertex.OnBoundary;
                }
            }

            /// <summary>
            /// The mesh the edge belongs to.
            /// </summary>
            public Mesh<TEdgeTraits, TFaceTraits, THalfedgeTraits, TVertexTraits> Mesh
            {
                get                {                    return Halfedge0.Mesh;                }
            }

            /// <summary>
            /// One vertex on the edge.
            /// </summary>
            public Vertex Vertex0
            {
                get { return Halfedge0.ToVertex; }
            }

            /// <summary>
            /// The other vertex on the edge.
            /// </summary>
            public Vertex Vertex1
            {
                get { return Halfedge0.Opposite.ToVertex; }
            }
            #endregion
        }
    }
}
