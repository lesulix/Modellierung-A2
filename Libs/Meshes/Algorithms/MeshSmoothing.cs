using Meshes;
using Meshes.Generic;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using SharpDX;
using CSparse.Double;
using System.Diagnostics;


namespace Meshes.Algorithms
{
    /// <summary>
    /// This class implements various triangle mesh smoothing methods. 
    /// </summary>
    public class MeshSmoothing
    {

        
        public enum Method
        {
            Explicit_Euler, Implicit_Euler, LS_Optimization, TriangleQuality
        }

        /// <summary>
        /// 
        /// </summary>
        public static readonly IList<string> MethodCollection = new List<string>
        {
            Method.Explicit_Euler.ToString(), 
            Method.Implicit_Euler.ToString(), 
            Method.LS_Optimization.ToString(), 
            Method.TriangleQuality.ToString(),
        };


        /// <summary>
        /// Singleton Instance
        /// </summary>
        public static readonly MeshSmoothing Instance = new MeshSmoothing();

        /// <summary>
        /// 
        /// </summary>
        /// <param name="mesh"></param>
        /// <returns></returns>
        public bool Smooth(TriangleMesh mesh)
        {
            switch (this.SelectedMethod)
            {
                default:
                case Method.Explicit_Euler:
                    this.ExplicitEulerSmoothing(mesh, this.Iterations);
                    break;
                case Method.Implicit_Euler:
                    this.ImplicitEulerSmoothing(mesh);
                    break;
                case Method.LS_Optimization:
                    this.GlobalLeastSquaresSmoothing(mesh);
                    break;
                case Method.TriangleQuality:
                    this.TriangleQualityOptimization(mesh);
                    break;

            }

            return true;
        }

        /// <summary>
        /// 
        /// </summary>
        public double Lambda
        {
            get { return this.lambda; }
            set
            {
                this.lambda = value;
                this.hasMatrix = false;
            }
        }

        /// <summary>
        /// 
        /// </summary>
        public int Iterations { get; set; }

        /// <summary>
        /// 
        /// </summary>
        public Method SelectedMethod { get; set; }

        /// <summary>
        /// 
        /// </summary>
        public void Clear()
        {
            this.hasMatrix = false;
        }



        /// <summary>
        /// Singleton, private c'tor
        /// </summary>
        private MeshSmoothing()
        {
            this.Iterations = 1;
        }

        /// <summary>
        /// Performs global least-squares mesh smoothing with the extended matrix
        /// [ L  ] [x'] =       [ 0 ]
        /// [ Wp ]        [ Wp ][ x ]        
        /// </summary>        
        private void GlobalLeastSquaresSmoothing(TriangleMesh mesh)
        {
            /// output vertex positions (after solving/smoothing)
            double[] xx, xy, xz;
            xx = xy = xz = null;

            

            /// update mesh           
            MeshLaplacian.UpdateMesh(mesh, xx, xy, xz);
        }

        /// <summary>
        /// performs the Explicit Euler (Taubin) smoothing
        /// </summary>
        private void ExplicitEulerSmoothing(TriangleMesh mesh, int iterations)
        {
            /// output vertex positions (after solving/smoothing)
            double[] xx, xy, xz;
            xx = xy = xz = null;

            /// get current world positions            
            MeshLaplacian.GetEuclideanCoordinates(mesh, out xx, out xy, out xz);

           

            /// update mesh 
            MeshLaplacian.UpdateMesh(mesh, xx, xy, xz);
        }

        /// <summary>
        /// Performs implicit Euler smoothng (Curvature Flow)
        /// </summary>        
        private void ImplicitEulerSmoothing(TriangleMesh mesh)
        {
            /// get n
            var n = mesh.Vertices.Count;

            /// output vertex positions (after solving/smoothing)
            double[] xx, xy, xz;
            xx = xy = xz = null;

            

            /// update mesh           
            MeshLaplacian.UpdateMesh(mesh, xx, xy, xz);
        }


        /// <summary>
        /// 
        /// </summary>
        /// <param name="mesh"></param>
        /// <returns></returns>
        private static float CalculateVolume(TriangleMesh mesh)
        {
            var faces = mesh.Faces;

            float v = 0.0f;

            foreach (var f in faces)
            {
                var h1 = f.Halfedge;
                var p1 = h1.FromVertex;
                var p2 = h1.Previous.FromVertex;
                var p3 = h1.Next.FromVertex;
                v += CalculateTriangleVolume(p1, p2, p3);
            }

            return v;
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="p1"></param>
        /// <param name="p2"></param>
        /// <param name="p3"></param>
        /// <returns></returns>
        private static float CalculateTriangleVolume(TriangleMesh.Vertex p1, TriangleMesh.Vertex p2, TriangleMesh.Vertex p3)
        {
            return 1.0f / 6.0f * (-p3.Traits.Position.X * p2.Traits.Position.Y * p1.Traits.Position.Z + p2.Traits.Position.X * p3.Traits.Position.Y * p1.Traits.Position.Z + p3.Traits.Position.X * p1.Traits.Position.Y * p2.Traits.Position.Z
                - p1.Traits.Position.X * p3.Traits.Position.Y * p2.Traits.Position.Z + p2.Traits.Position.X * p1.Traits.Position.Y * p1.Traits.Position.Z + p2.Traits.Position.X * p3.Traits.Position.Y * p1.Traits.Position.Z);
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="mesh"></param>
        private void TriangleQualityOptimization(TriangleMesh mesh)
        {
            /// get n
            var n = mesh.Vertices.Count;

            /// output vertex positions (after solving/smoothing)
            double[] xx, xy, xz;
            xx = xy = xz = null;



            /// update mesh           
            MeshLaplacian.UpdateMesh(mesh, xx, xy, xz);
        }


        private QR qr;
        private SparseMatrix L;
        private bool hasMatrix = false;
        private double lambda = 1.0;
        private const int order = 3;
    }
}
