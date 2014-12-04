#define CURVATURE
#region License
// Copyright (c) 2006 Alexander Kolliopoulos
//
// See license.txt for license information.
#endregion


using System;
using Point = SharpDX.Vector2;
using Point3D = SharpDX.Vector3;
using Vector3D = SharpDX.Vector3;
using Box3D = SharpDX.BoundingBox;

namespace Meshes
{
    #region Traits
    /// <summary>
    /// Halfedge traits with per-face vertex texture coordinates and normals.
    /// </summary>
    [Serializable]
    public struct HalfedgeTraits
    {
        /// <summary>
        /// Voronoi-area in the corner of the FromVertex of this halfedge
        /// </summary>
        public double VoronoiRegionArea;

        /// <summary>
        /// Stores the cotangens of the value of the 
        /// </summary>
        public double Cotan;
        
        /// <summary>
        /// U,V Texture coordinate.
        /// </summary>
        public Point TextureCoordinate;

        /// <summary>
        /// Normal for display.
        /// </summary>
        public Vector3D Normal;
    }

    /// <summary>
    /// Facde traits with per-face flat normals.
    /// </summary>
    [Serializable]
    public struct FaceTraits
    {
        /// <summary>
        /// Normal for display.
        /// </summary>
        public Vector3D Normal;

        /// <summary>
        /// Area of the face
        /// </summary>
        public double Area;
    }

    /// <summary>
    /// Mesh traits with bounding sphere and bounding box.
    /// </summary>
    [Serializable]
    public struct MeshTraits
    {
        /// <summary>
        /// Bounding sphere.
        /// </summary>
        //public Sphere BoundingSphere;

        /// <summary>
        /// Bounding box.
        /// </summary>
        public Box3D BoundingBox;

        /// <summary>
        /// Indicates whether the mesh loaded has face vertex normals.
        /// </summary>
        public bool HasFaceVertexNormals;

        /// <summary>
        /// Indicates whether the mesh loaded has texture coordinates.
        /// </summary>
        public bool HasTextureCoordinates;

        /// <summary>
        /// Indicates whether the mesh has tangent space.
        /// </summary>
        public bool HasTangents;
    }

    /// <summary>
    /// Null traits with no members.
    /// </summary>
    [Serializable]
    public struct NullTraits { }


    /// <summary>
    /// Vertex traits with position, normal, and principle curvatures.
    /// </summary>
    [Serializable]
    public struct VertexTraits
    {
        /// <summary>
        /// Point position.
        /// </summary>
        public Point3D Position;

        /// <summary>
        /// Vertex normal for computations.
        /// </summary>
        public Vector3D Normal;

        /// <summary>
        /// Tangent
        /// </summary>
        public Vector3D Tangent;

        /// <summary>
        /// Bi-Tangent
        /// </summary>
        public Vector3D BiTangent;

        /// <summary>
        /// U,V Texture coordinate.
        /// </summary>
        public Point TextureCoordinate;

#if CURVATURE
        /// <summary>
        /// Maximum principle curvature direction.
        /// </summary>
        public Vector3D MaxCurvatureDirection;

        /// <summary>
        /// Minimum principle curvature direction.
        /// </summary>
        public Vector3D MinCurvatureDirection;

        /// <summary>
        /// Maximum principle curvature.
        /// </summary>
        public double MaxCurvature;

        /// <summary>
        /// Minimum principle curvature.
        /// </summary>
        public double MinCurvature;
#endif

        /// <summary>
        /// Initializes vertex traits with the specified position.
        /// </summary>
        /// <param name="x">The x-coordinate.</param>
        /// <param name="y">The y-coordinate.</param>
        /// <param name="z">The z-coordinate.</param>
        public VertexTraits(float x, float y, float z)
            : this(new Point3D(x, y, z))
        {
        }

        public VertexTraits(Point3D p)
        {
            Position = p;
            Normal = new Vector3D();
            TextureCoordinate = new Point();
            Tangent = default(Vector3D);
            BiTangent = default(Vector3D);

            //IsFixed = false;
            //IsActive = false;
            //IsDifferential = false;
            //IsStiff = false;

            MaxCurvature = 1.0;
            MinCurvature = 0.0;
            MaxCurvatureDirection = new Vector3D();
            MinCurvatureDirection = new Vector3D();
        }
    }
    #endregion
}
