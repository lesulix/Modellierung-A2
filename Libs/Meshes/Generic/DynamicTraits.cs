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

namespace Meshes.Generic
{
    public partial class Mesh<TEdgeTraits, TFaceTraits, THalfedgeTraits, TVertexTraits>
    {
        #region EdgeDynamicTrait
        /// <summary>
        /// Creates a dynamic trait on the edges of an existing mesh.
        /// </summary>
        /// <typeparam name="TraitType">The type of the dynamic trait.</typeparam>
        /// <remarks>
        /// Edges to get the dynamic trait must be created and assigned to the mesh before
        /// the dynamic trait is created on the mesh.
        /// </remarks>
        [Serializable]
        public class EdgeDynamicTrait<TraitType>
        {
            #region Fields
            TraitType[] trait;
            Mesh<TEdgeTraits, TFaceTraits, THalfedgeTraits, TVertexTraits> mesh;
            #endregion

            #region Constructors
            /// <summary>
            /// Creates a new edge dynamic trait on the specified mesh.
            /// </summary>
            /// <param name="mesh">The mesh to create the dynamic trait on.</param>
            public EdgeDynamicTrait(Mesh<TEdgeTraits, TFaceTraits, THalfedgeTraits, TVertexTraits> mesh)
            {
                this.mesh = mesh;
                trait = new TraitType[mesh.Edges.Count];
            }
            #endregion

            #region Properties
            /// <summary>
            /// The dynamic trait for the specified edge.
            /// </summary>
            /// <param name="edge">The edge associated with the dynamic trait.</param>
            /// <value>The dynamic trait.</value>
            /// <exception cref="MismatchedMeshException">Thrown when <paramref name="edge"/> is from a mesh other than that of the dynamic trait.</exception>
            /// <exception cref="ArgumentOutOfRangeException">Thrown when <paramref name="edge"/> was not in the mesh when the dynamic trait was created.</exception>
            public TraitType this[Edge edge]
            {
                get
                {
                    if (edge.Mesh != mesh)
                    {
                        throw new MismatchedMeshException("The dynamic trait is not assigned to the mesh that the edge belongs to.");
                    }

                    try
                    {
                        return trait[edge.Index];
                    }
                    catch (IndexOutOfRangeException)
                    {
                        throw new ArgumentOutOfRangeException("Cannot access dynamic trait of an edge that wasn't present when the dynamic trait was created.");
                    }
                }
                set
                {
                    if (edge.Mesh != mesh)
                    {
                        throw new MismatchedMeshException("The dynamic trait is not assigned to the mesh that the edge belongs to.");
                    }

                    try
                    {
                        trait[edge.Index] = value;
                    }
                    catch (IndexOutOfRangeException)
                    {
                        throw new ArgumentOutOfRangeException("Cannot access dynamic trait of an edge that wasn't present when the dynamic trait was created.");
                    }
                }
            }

            /// <summary>
            /// The mesh that the dynamic trait belongs to.
            /// </summary>
            /// <value>The mesh.</value>
            public Mesh<TEdgeTraits, TFaceTraits, THalfedgeTraits, TVertexTraits> Mesh
            {
                get
                {
                    return mesh;
                }
            }
            #endregion
        }
        #endregion

        #region FaceDynamicTrait
        /// <summary>
        /// Creates a dynamic trait on the faces of an existing mesh.
        /// </summary>
        /// <typeparam name="TraitType">The type of the dynamic trait.</typeparam>
        /// <remarks>
        /// Faces to get the dynamic trait must be created and assigned to the mesh before
        /// the dynamic trait is created on the mesh.
        /// </remarks>
        [Serializable]
        public class FaceDynamicTrait<TraitType>
        {
            #region Fields
            TraitType[] trait;
            Mesh<TEdgeTraits, TFaceTraits, THalfedgeTraits, TVertexTraits> mesh;
            #endregion

            #region Constructors
            /// <summary>
            /// Creates a new face dynamic trait on the specified mesh.
            /// </summary>
            /// <param name="mesh">The mesh to create the dynamic trait on.</param>
            public FaceDynamicTrait(Mesh<TEdgeTraits, TFaceTraits, THalfedgeTraits, TVertexTraits> mesh)
            {
                this.mesh = mesh;
                trait = new TraitType[mesh.Faces.Count];
            }
            #endregion

            #region Properties
            /// <summary>
            /// The dynamic trait for the specified face.
            /// </summary>
            /// <param name="face">The face associated with the dynamic trait.</param>
            /// <value>The dynamic trait.</value>
            /// <exception cref="MismatchedMeshException">Thrown when <paramref name="face"/> is from a mesh other than that of the dynamic trait.</exception>
            /// <exception cref="ArgumentOutOfRangeException">Thrown when <paramref name="face"/> was not in the mesh when the dynamic trait was created.</exception>
            public TraitType this[Face face]
            {
                get
                {
                    if (face.Mesh != mesh)
                    {
                        throw new MismatchedMeshException("The dynamic trait is not assigned to the mesh that the face belongs to.");
                    }

                    try
                    {
                        return trait[face.Index];
                    }
                    catch (IndexOutOfRangeException)
                    {
                        throw new ArgumentOutOfRangeException("Cannot access dynamic trait of a face that wasn't present when the dynamic trait was created.");
                    }
                }
                set
                {
                    if (face.Mesh != mesh)
                    {
                        throw new MismatchedMeshException("The dynamic trait is not assigned to the mesh that the face belongs to.");
                    }

                    try
                    {
                        trait[face.Index] = value;
                    }
                    catch (IndexOutOfRangeException)
                    {
                        throw new ArgumentOutOfRangeException("Cannot access dynamic trait of a face that wasn't present when the dynamic trait was created.");
                    }
                }
            }

            /// <summary>
            /// The mesh that the dynamic trait belongs to.
            /// </summary>
            /// <value>The mesh.</value>
            public Mesh<TEdgeTraits, TFaceTraits, THalfedgeTraits, TVertexTraits> Mesh
            {
                get
                {
                    return mesh;
                }
            }
            #endregion
        }
        #endregion

        #region HalfedgeDynamicTrait
        /// <summary>
        /// Creates a dynamic trait on the halfedges of an existing mesh.
        /// </summary>
        /// <typeparam name="TraitType">The type of the dynamic trait.</typeparam>
        /// <remarks>
        /// Halfedges to get the dynamic trait must be created and assigned to the mesh before
        /// the dynamic trait is created on the mesh.
        /// </remarks>
        [Serializable]
        public class HalfedgeDynamicTrait<TraitType>
        {
            #region Fields
            TraitType[] trait;
            Mesh<TEdgeTraits, TFaceTraits, THalfedgeTraits, TVertexTraits> mesh;
            #endregion

            #region Constructors
            /// <summary>
            /// Creates a new halfedge dynamic trait on the specified mesh.
            /// </summary>
            /// <param name="mesh">The mesh to create the dynamic trait on.</param>
            public HalfedgeDynamicTrait(Mesh<TEdgeTraits, TFaceTraits, THalfedgeTraits, TVertexTraits> mesh)
            {
                this.mesh = mesh;
                trait = new TraitType[mesh.Halfedges.Count];
            }
            #endregion

            #region Properties
            /// <summary>
            /// The dynamic trait for the specified halfedge.
            /// </summary>
            /// <param name="halfedge">The halfedge associated with the dynamic trait.</param>
            /// <value>The dynamic trait.</value>
            /// <exception cref="MismatchedMeshException">Thrown when <paramref name="halfedge"/> is from a mesh other than that of the dynamic trait.</exception>
            /// <exception cref="ArgumentOutOfRangeException">Thrown when <paramref name="halfedge"/> was not in the mesh when the dynamic trait was created.</exception>
            public TraitType this[Halfedge halfedge]
            {
                get
                {
                    if (halfedge.Mesh != mesh)
                    {
                        throw new MismatchedMeshException("The dynamic trait is not assigned to the mesh that the halfedge belongs to.");
                    }

                    try
                    {
                        return trait[halfedge.Index];
                    }
                    catch (IndexOutOfRangeException)
                    {
                        throw new ArgumentOutOfRangeException("Cannot access dynamic trait of a halfedge that wasn't present when the dynamic trait was created.");
                    }
                }
                set
                {
                    if (halfedge.Mesh != mesh)
                    {
                        throw new MismatchedMeshException("The dynamic trait is not assigned to the mesh that the halfedge belongs to.");
                    }

                    try
                    {
                        trait[halfedge.Index] = value;
                    }
                    catch (IndexOutOfRangeException)
                    {
                        throw new ArgumentOutOfRangeException("Cannot access dynamic trait of a halfedge that wasn't present when the dynamic trait was created.");
                    }
                }
            }

            /// <summary>
            /// The mesh that the dynamic trait belongs to.
            /// </summary>
            /// <value>The mesh.</value>
            public Mesh<TEdgeTraits, TFaceTraits, THalfedgeTraits, TVertexTraits> Mesh
            {
                get
                {
                    return mesh;
                }
            }
            #endregion
        }
        #endregion

        #region VertexDynamicTrait
        /// <summary>
        /// Creates a dynamic trait on the vertices of an existing mesh.
        /// </summary>
        /// <typeparam name="TraitType">The type of the dynamic trait.</typeparam>
        /// <remarks>
        /// Vertices to get the dynamic trait must be created and assigned to the mesh before
        /// the dynamic trait is created on the mesh.
        /// </remarks>
        [Serializable]
        public class VertexDynamicTrait<TraitType>
        {
            #region Fields
            TraitType[] trait;
            Mesh<TEdgeTraits, TFaceTraits, THalfedgeTraits, TVertexTraits> mesh;
            #endregion

            #region Constructors
            /// <summary>
            /// Creates a new vertex dynamic trait on the specified mesh.
            /// </summary>
            /// <param name="mesh">The mesh to create the dynamic trait on.</param>
            public VertexDynamicTrait(Mesh<TEdgeTraits, TFaceTraits, THalfedgeTraits, TVertexTraits> mesh)
            {
                this.mesh = mesh;
                trait = new TraitType[mesh.Vertices.Count];
            }
            #endregion

            #region Properties
            /// <summary>
            /// The dynamic trait for the specified vertex.
            /// </summary>
            /// <param name="vertex">The vertex associated with the dynamic trait.</param>
            /// <value>The dynamic trait.</value>
            /// <exception cref="MismatchedMeshException">Thrown when <paramref name="vertex"/> is from a mesh other than that of the dynamic trait.</exception>
            /// <exception cref="ArgumentOutOfRangeException">Thrown when <paramref name="vertex"/> was not in the mesh when the dynamic trait was created.</exception>
            public TraitType this[Vertex vertex]
            {
                get
                {
                    if (vertex.Mesh != mesh)
                    {
                        throw new MismatchedMeshException("The dynamic trait is not assigned to the mesh that the vertex belongs to.");
                    }

                    try
                    {
                        return trait[vertex.Index];
                    }
                    catch (IndexOutOfRangeException)
                    {
                        throw new ArgumentOutOfRangeException("Cannot access dynamic trait of a vertex that wasn't present when the dynamic trait was created.");
                    }
                }
                set
                {
                    if (vertex.Mesh != mesh)
                    {
                        throw new MismatchedMeshException("The dynamic trait is not assigned to the mesh that the vertex belongs to.");
                    }

                    try
                    {
                        trait[vertex.Index] = value;
                    }
                    catch (IndexOutOfRangeException)
                    {
                        throw new ArgumentOutOfRangeException("Cannot access dynamic trait of a vertex that wasn't present when the dynamic trait was created.");
                    }
                }
            }

            /// <summary>
            /// The mesh that the dynamic trait belongs to.
            /// </summary>
            /// <value>The mesh.</value>
            public Mesh<TEdgeTraits, TFaceTraits, THalfedgeTraits, TVertexTraits> Mesh
            {
                get
                {
                    return mesh;
                }
            }
            #endregion
        }
        #endregion
    }
}
