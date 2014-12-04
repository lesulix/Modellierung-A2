
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using Meshes.Generic;
using SharpDX;

using CSparse.Double;

using System.Runtime.Serialization.Formatters.Binary;
using System.IO;

namespace Meshes.Algorithms
{
    /// <summary>
    /// This class encapsulates static methods needed for Laplacian mesh processing. 
    /// </summary>
    public static class MeshLaplacian
    {
        public enum Type
        {
            Uniform, Cotan, MeanValue,
        }

        /// <summary>
        /// Collection of supported Laplacian types
        /// </summary>
        public static IList<string> TypeCollection = new List<string>()
        {
            Type.Uniform.ToString(),
            Type.Cotan.ToString(),
            Type.MeanValue.ToString(),                
        };

        /// <summary>
        /// Returns a Laplacian Matrix depending on the chosen type.
        /// </summary>
        public static TripletMatrix CreateLaplacian(TriangleMesh mesh, double lambda = 0.0, double eye = 0.0)
        {
            switch (SelectedLaplacian)
            {
                default:
                case Type.Uniform:
                    return CreateUniformLaplacian(mesh, lambda, eye, LaplacianNormalize);
                case Type.Cotan:
                    return CreateCotanLaplacian(mesh, lambda, eye, LaplacianNormalize);
                case Type.MeanValue:
                    return CreateMeanValueLaplacian(mesh, lambda, eye, LaplacianNormalize);
            }
        }

        /// <summary>
        /// Create an extended Laplacian with an additional constraint matrix appended on the bottom.            
        /// </summary>
        public static TripletMatrix CreateExtendedLaplacian(TriangleMesh mesh, double lambda = 1.0)
        {
            switch (SelectedLaplacian)
            {
                default:
                case Type.MeanValue:
                case Type.Uniform:
                    return CreateExtendedUniformLaplacian(mesh, lambda, LaplacianNormalize);
                case Type.Cotan:
                    return CreateExtendedCotanLaplacian(mesh, lambda, LaplacianNormalize);
            }
        }

        /// <summary>
        /// 
        /// </summary>
        public static TripletMatrix CreateAdjacenceMatrix(TriangleMesh mesh)
        {
            var n = mesh.Vertices.Count;
            int nz = mesh.Vertices.Aggregate(0, (c, x) => c += x.VertexCount());
            var A = new TripletMatrix(n, n, nz + n);

            /// build adjacence matrix
            for (int i = 0; i < n; i++)
            {
                var v = mesh.Vertices[i];
                foreach (var vj in v.Vertices)
                {
                    int j = vj.Index;
                    A[i, j] = 1;
                }
            }
            return A;
        }

        /// <summary>
        /// Creates a square uniform n x n Laplacian matrix
        /// eye*[ I ] + lambda*[ L ];
        /// Note that depending on lambda you can use this function to create both:
        /// [I - λL]
        /// [I + λL]
        /// L is such that it has negative values on the diagonal and non-negative values elsewhere
        /// If the normalized flag is true, each diagonal element of L is normalized to -1
        /// </summary>       
        public static TripletMatrix CreateUniformLaplacian(TriangleMesh mesh, double lambda = 0.0, double eye = 0.0, bool normalized = false)
        {
            int n = mesh.Vertices.Count;
            int nz = mesh.Vertices.Aggregate(0, (c, x) => c += x.VertexCount());
            var L = new TripletMatrix(n, n, nz, true);

            /// build adjacence A and uniform L matrix
            for (int i = 0; i < n; i++)
            {
                var v = mesh.Vertices[i];
                var wsum = (double)v.VertexCount();
                var wij = 1.0;

                foreach (var vj in v.Vertices)
                {
                    int j = vj.Index;

                    if (normalized)
                        L[i, j] = lambda * (wij / wsum);
                    else
                        L[i, j] = lambda * (wij);
                }

                if (normalized)
                    L[i, i] = eye - lambda * 1;
                else
                    L[i, i] = eye - lambda * wsum;
            }

            return L;
        }

        /// <summary>
        /// Creates a square cotangent n x n Laplacian matrix
        /// eye*[ I ] + lambda*[ L ];
        /// Note that depending on lambda you can use this function to create both:
        /// [I - λL]
        /// [I + λL]
        /// L is such that it has negative values on the diagonal and non-negative values elsewhere
        /// If the normalized flag is true, each diagonal element of L is normalized to -1
        /// </summary>       
        public static TripletMatrix CreateCotanLaplacian(TriangleMesh mesh, double lambda = 1.0, double eye = 0.0, bool normalized = false)
        {
            var n = mesh.Vertices.Count;
            int nz = mesh.Vertices.Aggregate(0, (c, x) => c += x.VertexCount());
            var L = new TripletMatrix(n, n, nz + n);

            PrecomputeTraits(mesh);

            /// compute the matrix
            for (int i = 0; i < n; i++)
            {
                var vi = mesh.Vertices[i];

                /// cotan weight sum
                double wsum = 0;

                foreach (var hj in vi.Halfedges)
                {
                    ///right                    
                    var cotA = hj.Opposite.Previous.Traits.Cotan;
                    /// left
                    var cotB = hj.Previous.Traits.Cotan;
                    /// sum the weights for one ring
                    wsum += (cotA + cotB) / 2;
                }

                foreach (var hj in vi.Halfedges)
                {
                    /// the other vertex index
                    var j = hj.ToVertex.Index;
                    ///right                    
                    var cotA = hj.Opposite.Previous.Traits.Cotan;
                    /// left
                    var cotB = hj.Previous.Traits.Cotan;
                    /// weight wj normalized by the wsum
                    var wij = (cotA + cotB) / 2;

                    /// set the cotan values
                    if (normalized)
                        L[i, j] = lambda * (wij / wsum);
                    else
                        L[i, j] = lambda * (wij);
                }

                if (normalized)
                    L[i, i] = eye - lambda * 1.0;
                else
                    L[i, i] = eye - lambda * wsum;

            }

            return L;
        }

        /// <summary>
        /// 
        /// This method is not relevant for Assignment 1 yet!
        /// 
        /// Creates a square mean-value n x n Laplacian matrix
        /// eye*[ I ] + lambda*[ L ];
        /// Note that depending on lambda you can use this function to create both:
        /// [I - λL]
        /// [I + λL]
        /// L is such that it has negative values on the diagonal and non-negative values elsewhere
        /// If the normalized flag is true, each diagonal element of L is normalized to -1
        /// </summary>       
        public static TripletMatrix CreateMeanValueLaplacian(TriangleMesh mesh, double lambda = 1.0, double eye = 0.0, bool normalized = false)
        {
            var n = mesh.Vertices.Count;
            int nz = mesh.Vertices.Aggregate(0, (c, x) => c += x.VertexCount());
            var L = new TripletMatrix(n, n, nz + n);

            return L;
        }

        /// <summary>
        /// Creates a uniform n x n laplacian matrix extended by a n x n weight-matrix:
        ///        [ L ]
        /// lambda*[ I ]             
        /// If the normalized flag is true, each diagonal element of L is normalized to -1
        /// </summary>       
        public static TripletMatrix CreateExtendedUniformLaplacian(TriangleMesh mesh, double lambda = 1.0, bool normalized = true)
        {
            int n = mesh.Vertices.Count;
            int nz = mesh.Vertices.Aggregate(0, (c, x) => c += x.VertexCount());
            var L = new TripletMatrix(2 * n, n, nz + 2 * n, true);

            

            return L;
        }

        /// <summary>
        /// Creates a cotangent n x n laplacian matrix extended by a n x n weight-matrix:
        ///        [ L ]
        /// lambda*[ I ]             
        /// If the normalized flag is true, each diagonal element of L is normalized to -1
        /// </summary>          
        public static TripletMatrix CreateExtendedCotanLaplacian(TriangleMesh mesh, double lambda = 1.0, bool normalized = true)
        {
            var n = mesh.Vertices.Count;
            int nz = mesh.Vertices.Aggregate(0, (c, x) => c += x.VertexCount());
            var L = new TripletMatrix(2 * n, n, nz + 2 * n, true);

           

            return L;
        }

        /// <summary>
        /// Extracts the original mesh world-space coordinates as 3 arrays
        /// </summary>
        public static void GetEuclideanCoordinates(TriangleMesh mesh, out double[] px, out double[] py, out double[] pz)
        {
            /// original mesh world-space coordinates
            px = mesh.Vertices.Select(a => (double)a.Traits.Position.X).ToArray();
            py = mesh.Vertices.Select(a => (double)a.Traits.Position.Y).ToArray();
            pz = mesh.Vertices.Select(a => (double)a.Traits.Position.Z).ToArray();
        }

        /// <summary>
        /// Helper function: applies the matrix L to the vectors px, py and pz separately and returns the resutls.
        /// Can compute the differential (=Laplacian) coordinates of the mesh as 3 arrays
        /// </summary>
        public static void ComputeDifferentialCoordinates(SparseMatrix L, double[] px, double[] py, double[] pz,
            out double[] dx, out double[] dy, out double[] dz)
        {
            /// multiply dx = L*px
            L.Ax(px, out dx);
            L.Ax(py, out dy);
            L.Ax(pz, out dz);
        }

        /// <summary>
        /// Update the vertex positions in the mesh
        /// </summary>
        public static void UpdateMesh(TriangleMesh mesh, double[] xx, double[] xy, double[] xz)
        {
            /// update mesh           
            for (int i = 0; i < mesh.Vertices.Count; i++)
            {
                mesh.Vertices[i].Traits.Position = new Vector3((float)xx[i], (float)xy[i], (float)xz[i]);
            }
        }

        /// <summary>
        /// Update the vertex uv-coordinates in the mesh
        /// </summary>
        public static void UpdateMesh(TriangleMesh mesh, double[] u, double[] v)
        {
            /// update mesh           
            for (int i = 0; i < mesh.Vertices.Count; i++)
            {
                mesh.Vertices[i].Traits.TextureCoordinate = new Vector2((float)u[i], (float)v[i]);
            }
        }

        /// <summary>
        /// Update the vertex positions and uv-coordinates in the mesh
        /// </summary>
        public static void UpdateMesh(TriangleMesh mesh, double[] xx, double[] xy, double[] xz, double[] u, double[] v)
        {
            /// update mesh           
            for (int i = 0; i < mesh.Vertices.Count; i++)
            {
                mesh.Vertices[i].Traits.Position = new Vector3((float)xx[i], (float)xy[i], (float)xz[i]);
                mesh.Vertices[i].Traits.TextureCoordinate = new Vector2((float)u[i], (float)v[i]);
            }
        }

        /// <summary>
        /// Precompute Traits for all Mesh-Traits that are needed to the cotan-laplacian
        /// Store each angle-cotan at the (outgoing) halfedge of each vertex.
        /// Store each voronoi-area at the (outgoing) halfedge of 
        /// </summary>
        private static void PrecomputeTraits(TriangleMesh mesh)
        {
            /// compute all cotans
            /// store each cotan at the foot of each halfedge
            foreach (var ti in mesh.Faces)
            {

                /// compute cotans
                foreach (var hi in ti.Halfedges)
                {
                    var ei = hi.ToVertex.Traits.Position - hi.FromVertex.Traits.Position;
                    var hj = hi.Previous;
                    var ej = hj.FromVertex.Traits.Position - hj.ToVertex.Traits.Position;
                    hi.Traits.Cotan = Vector3.Dot(ei, ej) / Vector3.Cross(ei, ej).Length();
                }
            }
        }

        /// <summary>
        /// 
        /// </summary>
        public static Type SelectedLaplacian { get; set; }

        /// <summary>
        /// 
        /// </summary>
        public static bool LaplacianNormalize { get; set; }


        /// <summary>
        /// 
        /// </summary>
        public static void SerializeBinary<T>(string fileName, T data)
        {
            try
            {
                BinaryFormatter formatter = new BinaryFormatter();
                using (Stream stream = new FileStream(fileName, FileMode.Create, FileAccess.Write, FileShare.None))
                {
                    formatter.Serialize(stream, data);
                    stream.Close();
                }
            }
            catch (Exception ex)
            {
                throw new Exception("DeSerialization Error: " + ex.Message);
            }
        }

        /// <summary>
        /// 
        /// </summary>
        public static void DeSerializeBinary<T>(string fileName, out T data)
        {
            try
            {
                BinaryFormatter formatter = new BinaryFormatter();
                using (Stream stream = new FileStream(fileName, FileMode.Open, FileAccess.Read))
                {
                    data = (T)formatter.Deserialize(stream);
                    stream.Close();
                }
            }
            catch (Exception ex)
            {
                throw new Exception("Serialization Error: " + ex.Message);
            }
        }
    }
}
