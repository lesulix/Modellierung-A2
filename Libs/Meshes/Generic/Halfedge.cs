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
        /// A halfedge of the mesh.
        /// </summary>
        [Serializable]
        public class Halfedge
        {
            #region Fields
            /// <summary>
            /// The custom traits for the halfedge.
            /// </summary>
            public THalfedgeTraits Traits;
            
            #endregion

            #region Constructors
            /// <summary>
            /// Creates a halfedge with traits set to their default value.
            /// </summary>
            internal Halfedge() { }

            /// <summary>
            /// Creates a halfedge with the given traits.
            /// </summary>
            /// <param name="halfedgeTraits">Traits for this halfedge.</param>
            internal Halfedge(THalfedgeTraits halfedgeTraits)
            {
                Traits = halfedgeTraits;
            }
            #endregion

            #region Properties
            /// <summary>
            /// The edge corresponding to the halfedge.
            /// </summary>
            public Edge Edge
            {
                get;
                internal set;
            }

            /// <summary>
            /// The face corresponding to the halfedge.
            /// </summary>
            public Face Face
            {
                get;
                internal set;
            }

            /// <summary>
            /// The index of this in the mesh's interal halfedge list.
            /// </summary>
            public int Index
            {
                get;
                internal set;
            }

            /// <summary>
            /// The mesh the halfedge belongs to.
            /// </summary>
            public Mesh<TEdgeTraits, TFaceTraits, THalfedgeTraits, TVertexTraits> Mesh
            {
                get { return this.ToVertex.Mesh; }
            }

            /// <summary>
            /// Checks if the halfedge is on the boundary of the mesh.
            /// </summary>
            public bool OnBoundary
            {
                get { return this.Face == null; }
            }

            /// <summary>
            /// The opposite halfedge.
            /// </summary>
            public Halfedge Opposite
            {
                get;
                internal set;
            }

            /// <summary>
            /// The next halfedge.
            /// </summary>
            public Halfedge Next
            {
                get;
                internal set;
            }

            /// <summary>
            /// The previous halfedge.
            /// </summary>
            public Halfedge Previous
            {
                get;
                internal set;
            }

            /// <summary>
            /// The vertex pointed to by the halfedge.
            /// </summary>
            public Vertex ToVertex
            {
                get;
                internal set;
            }

            /// <summary>
            /// The vertex the halfedge originates from.
            /// </summary>
            public Vertex FromVertex
            {
                get { return this.Opposite.ToVertex; }
            }
            #endregion
        }
    }
}
