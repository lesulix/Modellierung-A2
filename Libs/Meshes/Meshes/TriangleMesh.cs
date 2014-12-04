﻿#region License
// Copyright (c) 2006 Alexander Kolliopoulos
//
// See license.txt for license information.
#endregion

#define CURVATURE_

using System;
using System.Linq;
using System.Collections.Generic;
using System.IO;
using System.Runtime.Serialization.Formatters.Binary;
using System.Text;
using Meshes.Generic;
using System.Globalization;
using Point = SharpDX.Vector2;
using Point3D = SharpDX.Vector3;
using Vector3D = SharpDX.Vector3;
using Box3D = SharpDX.BoundingBox;

namespace Meshes
{

    /// <summary>
    /// A triangular mesh with custom traits for point positions, normals, and curvature.
    /// </summary>
    [Serializable]
    public class TriangleMesh : Mesh<NullTraits, FaceTraits, HalfedgeTraits, VertexTraits>
    {
        #region ObjFileProcessorState class
        /// <summary>
        /// Stores state during processing of an OBJ file.
        /// </summary>
        private class ObjFileProcessorState
        {
            public List<Vector3D> VertexNormals = new List<Vector3D>();
            public List<Point> VertexTextureCoords = new List<Point>();
        }
        #endregion

        #region Fields
        /// <summary>
        /// The custom traits for the mesh.
        /// </summary>
        public MeshTraits Traits;

        string fileName = "";

        #endregion

        #region Constructors
        /// <summary>
        /// Initializes an empty mesh.
        /// </summary>
        public TriangleMesh()
        {
            trianglesOnly = true;
        }

        //public TriangleMesh3D(TriangleMesh3D m)
        //{
        //    this = m.Copy();
        //    fileName = m.fileName;
        //    Traits = m.Traits;
        //    trianglesOnly = m.trianglesOnly;
        //}
        #endregion

        #region Properties

        public string StatusText
        {
            get;
            internal set;
        }

        /// <summary>
        /// Gets or sets the name of the file for this mesh.
        /// </summary>
        /// <remarks>
        /// This is set automatically when a mesh is loaded from a file.
        /// </remarks>
        public string FileName
        {
            get { return fileName; }
            internal set { fileName = value; }
        }
        #endregion

        #region Methods
        /// <summary>
        /// Removes all elements from the mesh.
        /// </summary>
        protected override void Clear()
        {
            base.Clear();
            Traits = new MeshTraits();
        }

        /// <summary>
        /// Computes all traits on the mesh.
        /// </summary>
        public void ComputeAllTraits()
        {
            HalfedgeDynamicTrait<double> cornerArea = new HalfedgeDynamicTrait<double>(this);
            VertexDynamicTrait<double> pointArea = new VertexDynamicTrait<double>(this);

            ComputeBoundingSphereAndBox();
            ComputePointCornerArea(cornerArea, pointArea);
            ComputeNormals(cornerArea, pointArea);
#if CURVATURE
            ComputePrincipleCurvatures(cornerArea, pointArea);
#endif

            if (!Traits.HasFaceVertexNormals)
            {
                foreach (Halfedge h in Halfedges)
                {
                    h.Traits.Normal = h.ToVertex.Traits.Normal;
                }
            }

            if (Traits.HasTextureCoordinates)
            {
                this.ComputeTangents();
            }
        }

        /// <summary>
        /// Computes an approximate bounding sphere and axis-aligned bounding box.
        /// </summary>
        public void ComputeBoundingSphereAndBox()
        {
            if (Vertices.Count == 0)
            {
                // Traits.BoundingSphere.Center = Vector3D.Zero;
                //Traits.BoundingSphere.Radius = 0.0f;
            }
            else
            {
                Traits.BoundingBox = Box3D.FromPoints(Vertices.Select(x => x.Traits.Position).ToArray());
                //foreach (var v in Vertices)
                //{
                //    Traits.BoundingBox.ExtendBy((Vector3D)v.Traits.Position);
                //}
            }
        }

        /// <summary>
        /// Computes the angular area for each face vertex and the point area for each vertex.
        /// </summary>
        /// <param name="cornerArea">A halfedge dynamic trait to store face vertex angular areas.</param>
        /// <param name="pointArea">A vertex dynamic trait to store vertex angular areas.</param>
        public void ComputePointCornerArea(HalfedgeDynamicTrait<double> cornerArea, VertexDynamicTrait<double> pointArea)
        {
            Halfedge fh0, fh1, fh2;
            Vertex fv0, fv1, fv2;
            Vector3D e0, e1, e2;
            double length0, length1, length2, oneOverTotalLength;

            foreach (Face f in Faces)
            {
                // Get halfedges for this face
                fh0 = f.Halfedge;
                fh1 = fh0.Next;
                fh2 = fh1.Next;

                // Get vertices for this face
                fv0 = fh0.ToVertex;
                fv1 = fh1.ToVertex;
                fv2 = fh2.ToVertex;

                // Edge vectors
                e0 = fv2.Traits.Position - fv1.Traits.Position;
                e1 = fv0.Traits.Position - fv2.Traits.Position;
                e2 = fv1.Traits.Position - fv0.Traits.Position;

                // Triangle area
                double area = 0.5f * Vector3D.Cross(e0, e1).Length();

                // Edge lengths
                length0 = e0.Length();
                length1 = e1.Length();
                length2 = e2.Length();

                // Approximate corner area by fraction of triangle area given by opposite edge length
                oneOverTotalLength = 1.0 / (length0 + length1 + length2);
                cornerArea[fh0] = length2 * oneOverTotalLength;
                cornerArea[fh1] = length0 * oneOverTotalLength;
                cornerArea[fh2] = length1 * oneOverTotalLength;

                // Add corner areas to areas for vertices
                pointArea[fv0] += cornerArea[fh0];
                pointArea[fv1] += cornerArea[fh1];
                pointArea[fv2] += cornerArea[fh2];
            }
        }

#if CURVATURE
        /// <summary>
        /// Computes principle curvatures on the vertices.
        /// </summary>
        public void ComputePrincipleCurvatures()
        {
            HalfedgeDynamicTrait<double> cornerArea = new HalfedgeDynamicTrait<double>(this);
            VertexDynamicTrait<double> pointArea = new VertexDynamicTrait<double>(this);

            ComputePointCornerArea(cornerArea, pointArea);
            ComputePrincipleCurvatures(cornerArea, pointArea);
        }

        /// <summary>
        /// Computes principle curvatures on the vertices.
        /// </summary>
        /// <param name="cornerArea">A halfedge dynamic trait with face vertex angular areas.</param>
        /// <param name="pointArea">A vertex dynamic trait with vertex angular areas.</param>
        /// <remarks>
        /// Portions of this method are based on code from the C++ trimesh2 library
        /// (from TriMesh_curvature.cc).
        /// </remarks>
        public void ComputePrincipleCurvatures(HalfedgeDynamicTrait<double> cornerArea, VertexDynamicTrait<double> pointArea)
        {
            // Add dynamic trait for principle curvature computation
            VertexDynamicTrait<double> curv = new VertexDynamicTrait<double>(this);

            // Initialize principle curvatures to zero
            foreach (Vertex v in Vertices)
            {
                v.Traits.MaxCurvature = 0.0f;
                v.Traits.MinCurvature = 0.0f;
            }

            // Initialize a coordinate system for each vertex
            foreach (Vertex v in Vertices)
            {
                if (v.Traits.Normal.LengthSquared() > 0.0f)    // Ignore isolated points
                {
                    // Vector that points from this vertex to an adjacent one
                    v.Traits.MaxCurvatureDirection = v.Halfedge.ToVertex.Traits.Position - v.Traits.Position;
                    v.Traits.MaxCurvatureDirection.Normalize();

                    // Get a vector orthogonal to this vector and the vertex normal
                    v.Traits.MinCurvatureDirection = Vector3D.Cross(v.Traits.Normal, v.Traits.MaxCurvatureDirection);
                }
            }

            Halfedge[] fh = new Halfedge[3];
            Vertex[] fv = new Vertex[3];
            Vector3D[] e = new Vector3D[3];
            Vector3D t, b, dn, faceNormal;

            // Compute curvature for each face
            foreach (Face f in Faces)
            {
                // Get halfedges for this face
                fh[0] = f.Halfedge;
                fh[1] = fh[0].Next;
                fh[2] = fh[1].Next;

                // Get vertices for this face
                fv[0] = fh[0].ToVertex;
                fv[1] = fh[1].ToVertex;
                fv[2] = fh[2].ToVertex;

                // Edge vectors
                e[0] = fv[2].Traits.Position - fv[1].Traits.Position;
                e[1] = fv[0].Traits.Position - fv[2].Traits.Position;
                e[2] = fv[1].Traits.Position - fv[0].Traits.Position;

                t = e[0];
                t.Normalize();

                faceNormal = Vector3D.Cross(e[0], e[1]);
                faceNormal.Normalize();

                b = Vector3D.Cross(faceNormal, t);
                b.Normalize();

                // Estimate curvature by variation of normals along edges
                double[] m = new double[3];
                double[,] w = new double[3, 3];

                for (int i = 0; i < 3; ++i)
                {
                    double u = Vector3D.Dot(e[i], t);
                    double v = Vector3D.Dot(e[i], b);

                    w[0, 0] += u * u;
                    w[0, 1] += u * v;
                    w[2, 2] += v * v;

                    dn = fv[(i + 2) % 3].Traits.Normal - fv[(i + 1) % 3].Traits.Normal;

                    double dnu = Vector3D.Dot(dn, t);
                    double dnv = Vector3D.Dot(dn, b);

                    m[0] += dnu * u;
                    m[1] += dnu * v + dnv * u;
                    m[2] += dnv * v;
                }

                w[1, 1] = w[0, 0] + w[2, 2];
                w[1, 2] = w[0, 1];

                // Least squares solution
                double[] diag = new double[3];
                if (Curvature.LdlTransposeDecomp(w, diag))
                {
                    Curvature.LdlTransposeSolveInPlace(w, diag, m);

                    // Adjust curvature for vertices of this face
                    for (int i = 0; i < 3; ++i)
                    {
                        double c1, c12, c2;
                        Curvature.ProjectCurvature(t, b, m[0], m[1], m[2], fv[i].Traits.MaxCurvatureDirection, fv[i].Traits.MinCurvatureDirection, out c1, out c12, out c2);

                        double weight = cornerArea[fh[i]] / pointArea[fv[i]];
                        fv[i].Traits.MaxCurvature += weight * c1;
                        curv[fv[i]] += weight * c12;
                        fv[i].Traits.MinCurvature += weight * c2;
                    }
                }
            }

            // Compute curvature for each vertex
            foreach (Vertex v in Vertices)
            {
                if (v.Traits.Normal.LengthSquared() > 0.0f)    // Ignore isolated points
                {
                    Curvature.DiagonalizeCurvature(v.Traits.MaxCurvatureDirection, v.Traits.MinCurvatureDirection, v.Traits.MaxCurvature, curv[v], v.Traits.MinCurvature, v.Traits.Normal,
                        out v.Traits.MaxCurvatureDirection, out v.Traits.MinCurvatureDirection, out v.Traits.MaxCurvature, out v.Traits.MinCurvature);
                }
            }
        }
#endif
        /// <summary>
        /// Compute a tangent-space for the mesh, if it has a parameterization
        /// </summary>
        /// <param name="mesh"></param>
        public void ComputeTangents()
        {
            if (this.Traits.HasTextureCoordinates)
            {
                var pos = this.Vertices.Select(x => x.Traits.Position).ToList();
                var nor = this.Vertices.Select(x => x.Traits.Normal).ToList();
                var tex = this.Vertices.Select(x => x.Traits.TextureCoordinate).ToList();
                var ind = this.Faces.SelectMany(x => x.Vertices.Select(xx => xx.Index)).ToList();

                List<Point3D> tg1, tg2;

                ComputeTangents(pos, nor, tex, ind, out tg1, out tg2);

                for (int i = 0; i < this.Vertices.Count; i++)
                {
                    this.Vertices[i].Traits.Tangent = tg1[i];
                    this.Vertices[i].Traits.BiTangent = tg2[i];
                }

                this.Traits.HasTangents = true;
            }
        }

        /// <summary>
        /// Computes vertex normals.
        /// </summary>
        public void ComputeNormals()
        {
            HalfedgeDynamicTrait<double> cornerArea = new HalfedgeDynamicTrait<double>(this);
            VertexDynamicTrait<double> pointArea = new VertexDynamicTrait<double>(this);

            ComputePointCornerArea(cornerArea, pointArea);
            ComputeNormals(cornerArea, pointArea);
        }

        /// <summary>
        /// Computes vertex normals.
        /// </summary>
        /// <param name="cornerArea">A halfedge dynamic trait with face vertex angular areas.</param>
        /// <param name="pointArea">A vertex dynamic trait with vertex angular areas.</param>
        public void ComputeNormals(HalfedgeDynamicTrait<double> cornerArea, VertexDynamicTrait<double> pointArea)
        {
            this.Traits.HasFaceVertexNormals = true;
            FaceDynamicTrait<Vector3D> normal = new FaceDynamicTrait<Vector3D>(this);

            Vertex[] fv = new Vertex[3];

            // Compute normal for each face
            foreach (Face f in Faces)
            {
                int i = 0;
                foreach (Vertex v in f.Vertices)
                {
                    fv[i] = v;
                    ++i;
                }

                // dynamic traits
                // Compute normal for this face
                var n = Vector3D.Cross(fv[2].Traits.Position - fv[1].Traits.Position, fv[0].Traits.Position - fv[2].Traits.Position);
                n.Normalize();                

                f.Traits.Normal = n;
                normal[f] = n;

            }

            // Compute normal for each vertex
            foreach (Halfedge h in Halfedges)
            {
                if (!h.OnBoundary)  // Ignore halfedges that don't have a face
                {
                    var weight = (float)(cornerArea[h] / pointArea[h.ToVertex]);
                    h.ToVertex.Traits.Normal += weight * normal[h.Face];
                    //h.ToVertex.Traits.Normal += weight * h.Traits.Normal;
                }
            }

            // Normalize normals
            foreach (TriangleMesh.Vertex v in Vertices)
            {
                if (v.Traits.Normal.LengthSquared() > 0.0f)    // Ignore isolated points
                {
                    v.Traits.Normal.Normalize();
                }
            }
        }

        /// <summary>
        /// Creates a new, identical mesh.
        /// </summary>
        /// <returns>A deep copy of the mesh.</returns>
        public TriangleMesh Copy()
        {
            // Use serialization to create a deep copy
            using (MemoryStream ms = new MemoryStream(Vertices.Count * 300))
            {
                BinaryFormatter bf = new BinaryFormatter();
                bf.Serialize(ms, this);
                ms.Seek(0, SeekOrigin.Begin);
                return (TriangleMesh)bf.Deserialize(ms);
            }
        }

        /// <summary>
        /// Loads an OBJ file.
        /// </summary>
        /// <param name="fileName">The name of the file to load.</param>
        protected void LoadObjFile(string fileName, bool hasVertexNormals = true, bool hasTextureCoords = false)
        {
            using (Stream stream = File.OpenRead(fileName))
            {
                LoadObjStream(stream, hasVertexNormals, hasTextureCoords);
            }
            this.fileName = fileName;
        }

        /// <summary>
        /// Loads an OBJ file from a stream.
        /// </summary>
        /// <param name="stream">A stream with OBJ file data.</param>
        protected void LoadObjStream(Stream stream, bool hasVertexNormals = true, bool hasTextureCoords = false)
        {
            Traits.HasFaceVertexNormals = hasVertexNormals;
            Traits.HasTextureCoordinates = hasTextureCoords;

            StreamReader sr = new StreamReader(stream);
            ObjFileProcessorState state = new ObjFileProcessorState();
            string line;

            while ((line = sr.ReadLine()) != null)
            {
                ProcessObjLine(line, state);
            }

            // add per-vertex properties
            //if (state.VertexNormals.Count == this.Vertices.Count)
            //{
            //    for (int i = 0; i < this.Vertices.Count; i++)
            //    {
            //        this.Vertices[i].Traits.Normal = state.VertexNormals[i];
            //    }

            //    this.Traits.HasFaceVertexNormals = true;
            //}
            //else
            //{
            //    this.Traits.HasFaceVertexNormals = false;
            //}

            //if (state.VertexTextureCoords.Count == this.Vertices.Count)
            //{
            //    for (int i = 0; i < this.Vertices.Count; i++)
            //    {
            //        this.Vertices[i].Traits.TextureCoordinate = state.VertexTextureCoords[i];
            //    }

            //    this.Traits.HasTextureCoordinates = true;
            //}
            //else
            //{
            //    this.Traits.HasTextureCoordinates = false;
            //}

            TrimExcess();
        }

        /// <summary>
        /// Processes a line from an OBJ file.
        /// </summary>
        /// <param name="line">A line from an OBJ file.</param>
        /// <param name="state">An object that manages state between calls.</param>
        private void ProcessObjLine(string line, ObjFileProcessorState state)
        {
            // Trim out comments (allow comments trailing on a line)
            int commentStart = line.IndexOf('#');
            if (commentStart != -1)
            {
                line = line.Substring(0, commentStart);
            }

            // Tokenize line
            string[] tokens = line.Split((char[])null, StringSplitOptions.RemoveEmptyEntries);

            // Process line based on the keyword used
            if (tokens.Length > 0)
            {
                int? v;
                float x, y, z;
                Vertex[] faceVertices;
                int?[] vt, vn;

                switch (tokens[0])
                {
                    case "v":   // Vertex
                        if (tokens.Length < 4)
                        {
                            throw new IOException("Vertices in the OBJ file must have 3 coordinates.");
                        }

                        x = Single.Parse(tokens[1], CultureInfo.InvariantCulture);
                        y = Single.Parse(tokens[2], CultureInfo.InvariantCulture);
                        z = Single.Parse(tokens[3], CultureInfo.InvariantCulture);                        

                        Vertices.Add(new VertexTraits(x, y, z));
                        break;

                    case "vt":   // Vertex texture
                        if (tokens.Length != 3 && tokens.Length != 4)
                        {
                            throw new IOException("Texture coordinates in the OBJ file must have 2 or 3 coordinates.");
                        }

                        x = Single.Parse(tokens[1], CultureInfo.InvariantCulture);
                        y = Single.Parse(tokens[2], CultureInfo.InvariantCulture);

                        state.VertexTextureCoords.Add(new Point(x, y));
                        break;

                    case "vn":   // Vertex normal
                        if (tokens.Length < 4)
                        {
                            throw new IOException("Vertex normals in the OBJ file must have 3 coordinates.");
                        }

                        x = Single.Parse(tokens[1], CultureInfo.InvariantCulture);
                        y = Single.Parse(tokens[2], CultureInfo.InvariantCulture);
                        z = Single.Parse(tokens[3], CultureInfo.InvariantCulture);

                        state.VertexNormals.Add(new Vector3D(x, y, z));
                        break;

                    case "f":   // Face
                        faceVertices = new Vertex[tokens.Length - 1];
                        vt = new int?[tokens.Length - 1];
                        vn = new int?[tokens.Length - 1];

                        // Parse vertex/texture coordinate/normal indices
                        for (int i = 0; i < faceVertices.Length; ++i)
                        {
                            string[] vertexTokens = tokens[i + 1].Split("/".ToCharArray());

                            v = Int32.Parse(vertexTokens[0]);

                            if (vertexTokens.Length > 1 && vertexTokens[1].Length > 0)
                            {
                                vt[i] = Int32.Parse(vertexTokens[1]);
                            }
                            else
                            {
                                Traits.HasTextureCoordinates = false;
                            }

                            if (vertexTokens.Length > 2 && vertexTokens[2].Length > 0)
                            {
                                vn[i] = Int32.Parse(vertexTokens[2]);
                            }
                            else
                            {
                                Traits.HasFaceVertexNormals = false;
                            }

                            faceVertices[i] = Vertices[v.Value - 1];
                        }

                        Face[] addedFaces = Faces.AddTriangles(faceVertices);
                        
                        // Set texture coordinates and normals if any are given
                        // This sets per-face-vertex normals and tex-coords
                        for (int i = 0; i < faceVertices.Length; ++i)
                        {
                            Halfedge faceVertex;
                            if (vt[i].HasValue || vn[i].HasValue)
                            {
                                foreach (Face f in addedFaces)
                                {
                                    faceVertex = f.FindHalfedgeTo(faceVertices[i]);

                                    // Make sure vertex belongs to face if triangularization is on
                                    if (faceVertex != null)
                                    {
                                        if (vt[i].HasValue)
                                        {
                                            faceVertex.Traits.TextureCoordinate = state.VertexTextureCoords[vt[i].Value - 1];
                                            faceVertices[i].Traits.TextureCoordinate = state.VertexTextureCoords[vt[i].Value - 1];
                                        }
                                        if (vn[i].HasValue)
                                        {
                                            faceVertex.Traits.Normal = state.VertexNormals[vn[i].Value - 1];
                                            faceVertices[i].Traits.Normal = state.VertexNormals[vn[i].Value - 1];
                                        }
                                    }
                                }
                            }
                        }
                        break;
                }             
            }
        }
        #endregion

        #region Static methods
        /// <summary>
        /// Returns a <see cref="TriangleMesh"/> object loaded from the specified OBJ file.
        /// </summary>
        /// <param name="fileName">The name of the OBJ file to load.</param>
        /// <returns>The mesh loaded from the OBJ file.</returns>
        public static TriangleMesh FromObjFile(string fileName, bool hasVertexNormals = true, bool hasTextureCoords = false)
        {
            TriangleMesh m = new TriangleMesh();
            m.LoadObjFile(fileName, hasVertexNormals, hasTextureCoords);
            return m;
        }

        /// <summary>
        /// Returns a <see cref="TriangleMesh"/> object loaded from the specified OBJ file stream.
        /// </summary>
        /// <param name="stream">The stream for the OBJ to load.</param>
        /// <returns>The mesh loaded from the OBJ file stream.</returns>
        public static TriangleMesh FromObjStream(Stream stream)
        {
            TriangleMesh m = new TriangleMesh();
            m.LoadObjStream(stream);
            return m;
        }

        /// <summary>
        /// Tangent Space computation for IndexedTriangle meshes
        /// Based on:
        /// http://www.terathon.com/code/tangent.html
        /// </summary>
        private static void ComputeTangents(List<Point3D> positions, List<Point3D> normals, List<Point> textureCoordinates, List<int> triangleIndices,
            out List<Point3D> tangents, out List<Point3D> bitangents)
        {

            var tan1 = new Point3D[positions.Count];
            //var tan2 = new Point3D[positions.Count];
            for (int t = 0; t < triangleIndices.Count; t += 3)
            {
                var i1 = triangleIndices[t];
                var i2 = triangleIndices[t + 1];
                var i3 = triangleIndices[t + 2];

                var v1 = positions[i1];
                var v2 = positions[i2];
                var v3 = positions[i3];

                var w1 = textureCoordinates[i1];
                var w2 = textureCoordinates[i2];
                var w3 = textureCoordinates[i3];

                float x1 = v2.X - v1.X;
                float x2 = v3.X - v1.X;
                float y1 = v2.Y - v1.Y;
                float y2 = v3.Y - v1.Y;
                float z1 = v2.Z - v1.Z;
                float z2 = v3.Z - v1.Z;

                float s1 = w2.X - w1.X;
                float s2 = w3.X - w1.X;
                float t1 = w2.Y - w1.Y;
                float t2 = w3.Y - w1.Y;

                float r = 1.0f / (s1 * t2 - s2 * t1);
                var udir = new Vector3D((t2 * x1 - t1 * x2) * r, (t2 * y1 - t1 * y2) * r, (t2 * z1 - t1 * z2) * r);
                //var vdir = new Vector3D((s1 * x2 - s2 * x1) * r, (s1 * y2 - s2 * y1) * r, (s1 * z2 - s2 * z1) * r);

                tan1[i1] += udir;
                tan1[i2] += udir;
                tan1[i3] += udir;

                //tan2[i1] += vdir;
                //tan2[i2] += vdir;
                //tan2[i3] += vdir;
            }

            tangents = new List<Point3D>(positions.Count);
            bitangents = new List<Point3D>(positions.Count);
            for (int i = 0; i < positions.Count; i++)
            {
                var n = normals[i];
                var t = tan1[i];
                t = (t - n * Vector3D.Dot(n, t));
                t.Normalize();
                var b = Vector3D.Cross(n, t);
                tangents.Add(t);
                bitangents.Add(b);
            }
        }
        #endregion
    }
}
