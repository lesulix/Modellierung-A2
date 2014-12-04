//#define MATLAB

using Meshes.Generic;
using SharpDX;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Drawing;
using CSparse.Double;
using MathNet.Numerics.LinearAlgebra.Double;
using System.Threading.Tasks;
using MeshVertex = Meshes.Generic.Mesh<Meshes.NullTraits, Meshes.FaceTraits, Meshes.HalfedgeTraits, Meshes.VertexTraits>.Vertex;

#if MATLAB
using MatlabWrap;
#endif


namespace Meshes.Algorithms
{
    public enum SolverLibrary
    {
        CSparseDotNet,
        CXSparseDotNet,
        LSQRDotNet,
    }

    /// <summary>
    /// This class implements various triangle mesh parametrization methods. 
    /// </summary>
    public class MeshParameterization
    {
        public const SolverLibrary solver = SolverLibrary.CSparseDotNet;
    
        interface IBoundary
        {
            void GetUvCoordinates(double t, Action<double, double> applyUV);
        }

        class CircularBoundary : IBoundary
        {
            public void GetUvCoordinates(double t, Action<double, double> applyUV)
            {
                var radians = 2 * t * Math.PI;
                applyUV(Math.Sin(radians)*0.5d + 0.5d, Math.Cos(radians)*0.5d + 0.5d);
            }
        }

        class RectangleBoundary : IBoundary
        {
            internal struct QuadSide
            {
                public double IntervalStart { get; private set; }

                private readonly Func<double, double> _uIntervalConverter;
                private readonly Func<double, double> _vIntervalConverter;

                public QuadSide(Func<double, double> uIntervalConverter, Func<double, double> vIntervalConverter, double intervalStart)
                    : this()
                {
                    IntervalStart = intervalStart;
                    _uIntervalConverter = uIntervalConverter;
                    _vIntervalConverter = vIntervalConverter;
                }

                public void Convert(double t, Action<double, double> applyUV)
                {
                    applyUV(_uIntervalConverter(t), _vIntervalConverter(t));
                }
            }

            static readonly QuadSide Bottom = new QuadSide(t => t, t => 0d, 0d);
            static readonly QuadSide Top = new QuadSide(t => 1d - t, t => 1d, 0.5d);
            static readonly QuadSide Left = new QuadSide(t => 0d, t => 1d - t, 0.75d);
            static readonly QuadSide Right = new QuadSide(t => 1d, t => t, 0.25d);

            public static readonly List<QuadSide> QuadIntervals = new List<QuadSide> { Left, Top, Right, Bottom };  

            public void GetUvCoordinates(double t, Action<double, double> applyUV)
            {
                var interval = QuadIntervals.First(i => t >= i.IntervalStart);
                interval.Convert((t - interval.IntervalStart) * 4d, applyUV);
            }
        }

        public enum Method
        {
            Barycentric, LSCM, DCP, LinearABF
        }

        public enum BoundaryType
        {
            Rectangle, RectangleAdaptive, Circle, CircleAdaptive
        }

        /// <summary>
        /// 
        /// </summary>
        public static readonly IList<string> MethodCollection = new List<string>
        {
            Method.Barycentric.ToString(),
            Method.LSCM.ToString(),
            Method.DCP.ToString(),
            Method.LinearABF.ToString(),
        };

        public static readonly IList<string> BoundaryTypeCollection = new List<string>
		{
			BoundaryType.Rectangle.ToString(),
			BoundaryType.RectangleAdaptive.ToString(),
			BoundaryType.Circle.ToString(),
			BoundaryType.CircleAdaptive.ToString(),
		};

        public static BoundaryType BoundaryTypeFromString(string boundaryType)
        {
            if (boundaryType == BoundaryType.Rectangle.ToString()) return BoundaryType.Rectangle;
            else if (boundaryType == BoundaryType.RectangleAdaptive.ToString()) return BoundaryType.RectangleAdaptive;
            else if (boundaryType == BoundaryType.Circle.ToString()) return BoundaryType.Circle;
            else return BoundaryType.CircleAdaptive;
        }

        /// <summary>
        /// 
        /// </summary>
        public Method SelectedMethod
        {
            get;
            set;
        }

        public BoundaryType SelectedBoundaryType
        {
            get;
            set;
        }

        public int P1Index { get; set; }
        public int P2Index { get; set; }
        public Vector2 P1UV { get; set; }
        public Vector2 P2UV { get; set; }


        /// <summary>
        /// the one an only instance of a singleton class
        /// </summary>
        public static readonly MeshParameterization Instance = new MeshParameterization();

        /// <summary>
        /// private constuctor of a singleton
        /// </summary>
        private MeshParameterization()
        {
            this.SelectedMethod = Method.Barycentric;
            this.P1Index = 153;
            this.P2Index = 19;
            this.P1UV = new Vector2(0.5f, 0);
            this.P2UV = new Vector2(0.5f, 1);

#if MATLAB
            /// initalizes a MALAB session connected to this C# program
            /// if you want to see it in your MATLAB window,
            /// (1) first, start MABLAB *before* starting this program
            /// (2) call in MATLAB: enableservice('AutomationServer',true); (two times?)
            /// (3) start this program, then you can push matrices to MATLAB, see examples below
            Matlab.InitMATLAB(true);
#endif
        }



        /// <summary>
        /// Performs parameterization with the chosen method on the given mesh
        /// The mesh is assumed to be non-closed with proper boundary
        /// </summary>
        /// <param name="meshIn">input mesh, after solving its texture coordinates in vertex traits will be adjusted</param>
        /// <param name="meshOut">an flattened output mesh with only X,Y coordinates set, Z is set to 0</param>
        public void PerformParameterization(TriangleMesh meshin, out TriangleMesh meshout)
        {
            switch (this.SelectedMethod)
            {
                default:
                case Method.Barycentric:
                    BarycentricMapping(meshin, out meshout);
                    break;
                case Method.LSCM:
                    LSCM(meshin, out meshout);
                    break;
                case Method.DCP:
                    DCP(meshin, out meshout);
                    break;
                case Method.LinearABF:
                    LinearABF(meshin, out meshout);
                    break;                                    
            }
        }

        /// <summary>
        /// Barycentric Parameterization
        /// Covers barycentric methods which need a fully defined boundary
        /// A particular method can be chosen by creating an appropriate Laplacian matrix
        /// See also (Floater 2003)
        /// </summary>
        /// <param name="meshin">input mesh, after solving its texture coordinates in vertex traits will be adjusted</param>
        /// <param name="meshout">an flattened output mesh with only X,Y coordinates set, Z is set to 0</param>
        private void BarycentricMapping(TriangleMesh meshin, out TriangleMesh meshout)
        {
            /// init an mesh that serves for output of the 2d parametrized mesh
            meshout = meshin.Copy();
            //meshOut = meshIn;            

            /// get lenghts
            var vertexCount = meshout.Vertices.Count;
            var boundaryVertices = meshout.Vertices.Where(x => x.OnBoundary).ToList();

            /// right hand side (RHS)
            var bu = new double[vertexCount];
            var bv = new double[vertexCount];
            var b0 = new double[vertexCount];

            FixBoundaryToShape(boundaryVertices, bu, bv);

            /// TODO_A2 Task 1
            /// implement linear Barycentric Parameterization
            ///     c.  using uniform and harmonic weights (5 points) 
            ///     d.  using mean value weights [2] (10 points)  

            var laplacian = MeshLaplacian.CreateBoundedUniformLaplacian(meshin, -1d, 0d, true);

            var qrSolver = QR.Create(laplacian.Compress());
            var success = qrSolver.Solve(bu) && qrSolver.Solve(bv);

            /// update mesh positions
            MeshLaplacian.UpdateMesh(meshout, bu, bv, b0, bu, bv);
            MeshLaplacian.UpdateMesh(meshin, bu, bv);
        }

        private void FixBoundaryToShape(List<MeshVertex> boundaryVertices, double[] bu, double[] bv)
        {
            var sortedBoundary = new List<Mesh<NullTraits, FaceTraits, HalfedgeTraits, VertexTraits>.Vertex>();

            var boundary =
                RectangleBoundaryIsActive ? 
                    (IBoundary) new RectangleBoundary() : 
                    (IBoundary) new CircularBoundary();

            var edgeLengthFunc =
                AdaptiveBoundaryIsActive
                    ? (Func<Vector3, Vector3, double>) ((lhs, rhs) => (lhs - rhs).Length())
                    : (Func<Vector3, Vector3, double>) ((lhs, rhs) => 1d);

            var currentBoundaryVertex = boundaryVertices.FirstOrDefault();
            var prevBoundaryVertex = (MeshVertex) null;
            var edgeSum = 0d;

            while (currentBoundaryVertex != null)
            {
                sortedBoundary.Add(currentBoundaryVertex);

                if (prevBoundaryVertex != null)
                    edgeSum += edgeLengthFunc(prevBoundaryVertex.Traits.Position, currentBoundaryVertex.Traits.Position);

                prevBoundaryVertex = currentBoundaryVertex;
                currentBoundaryVertex =
                    currentBoundaryVertex.Vertices.FirstOrDefault(v => v.OnBoundary && !sortedBoundary.Contains(v));
                
            }
            edgeSum += edgeLengthFunc(sortedBoundary.Last().Traits.Position, sortedBoundary.First().Traits.Position);

            var paramT = 0d;
            prevBoundaryVertex = sortedBoundary.Last();

            foreach (var entry in sortedBoundary)
            {
                boundary.GetUvCoordinates(paramT, (u, v) =>
                {
                    bu[entry.Index] = u;
                    bv[entry.Index] = v;
                });

                paramT += edgeLengthFunc(prevBoundaryVertex.Traits.Position, entry.Traits.Position) / edgeSum;
                prevBoundaryVertex = entry;
            }
        }

        private bool AdaptiveBoundaryIsActive
        {
            get
            {
                return SelectedBoundaryType == BoundaryType.RectangleAdaptive ||
                       SelectedBoundaryType == BoundaryType.CircleAdaptive;
            }
        }

        private bool RectangleBoundaryIsActive
        {
            get
            {
                return SelectedBoundaryType == BoundaryType.Rectangle ||
                       SelectedBoundaryType == BoundaryType.RectangleAdaptive;
            }
        }

        /// <summary>
        /// The Least-Squares Conformal Mapping method, see (Levy et al. 2002)
        /// Performs linear mapping with free boundary
        /// </summary>
        /// <param name="meshIn">input mesh, after solving its texture coordinates in vertex traits will be adjusted</param>
        /// <param name="meshOut">an flattened output mesh with only X,Y coordinates set, Z is set to 0</param>
        private void LSCM(TriangleMesh meshin, out TriangleMesh meshout)
        {
            /// copy mesh for output
            meshout = meshin.Copy();

            /// get fixed points
            int fv0 = this.P1Index;
            int fv1 = this.P2Index;

            /// provide uv's for fixed 2 points            
            var b = new double[]
            {
                this.P1UV.X, this.P1UV.Y, // u1,v1
                this.P2UV.X, this.P2UV.Y, // u2,v2
            };

            /// get counts
            int n = meshout.Vertices.Count;
            int m = meshout.Faces.Count;

            /// output uv-coordinates
            var bu = new double[n];
            var bv = new double[n];
            var b0 = new double[n];

            var A1 = new TripletMatrix(2 * m, 2 * n - 4, 6 * 2 * m);
            var A2 = new TripletMatrix(2 * m, 4, 4 * 2 * m);

            /// TODO_A2 Task 4
            /// implement Least Squares Conformal Maps (LCSM)
            /// use this function to compute the matrix M per triangle      

            /// update mesh positions and uv's
            MeshLaplacian.UpdateMesh(meshout, bu, bv, b0, bu, bv);
            MeshLaplacian.UpdateMesh(meshin, bu, bv);
        }

        /// <summary>
        /// The Direct Conformal Parameterization (DCP) method, see (Desbrun et al. 2002)
        /// </summary>
        /// <param name="meshin"></param>
        /// <param name="meshout"></param>
        private void DCP(TriangleMesh meshin, out TriangleMesh meshout)
        {
            /// copy the mesh
            meshout = meshin.Copy();

            /// counters
            int n = meshout.Vertices.Count;
            int m = meshout.Faces.Count;

            /// output uv-coordinates
            var bu = new double[n];
            var bv = new double[n];
            var b0 = new double[n];

            /// TODO_A2 Task 5
            /// implement Direct Conformal Parameterization (DCP)           

            /// update mesh positions and uv's
            MeshLaplacian.UpdateMesh(meshout, bu, bv, b0, bu, bv);
            MeshLaplacian.UpdateMesh(meshin, bu, bv);
        }

        /// <summary>
        /// Linear Angle Based Parameterization (LinABF), see (Zayer et al. 2007)
        /// </summary>
        /// <param name="meshin"></param>
        /// <param name="meshout"></param>
        private void LinearABF(TriangleMesh meshin, out TriangleMesh meshout)
        {
            /// copy the mesh
            meshout = meshin.Copy();

            /// counters
            int n = meshout.Vertices.Count;
            int m = meshout.Faces.Count;

            /// output uv-coordinates
            var bu = new double[n];
            var bv = new double[n];
            var b0 = new double[n];

            /// TODO_A2 Task 6
            /// implement Linear Angle Based Smoothing (LinABF)            

            /// update mesh positions and uv's
            MeshLaplacian.UpdateMesh(meshout, bu, bv, b0, bu, bv);
            MeshLaplacian.UpdateMesh(meshin, bu, bv);
        }


        /// <summary>
        /// Create a geometry bitmap image from a parameterization (without cutting)
        /// See (Gu et al. 2002)
        /// </summary>
        /// <param name="meshin"></param>
        /// <returns></returns>
        public Bitmap GenerateGeometryImage(TriangleMesh meshin)
        {
            var gimg = new Bitmap(512, 512);

            /// TODO_A2 Task 3
            /// implement geometry image sampling using a parameterized mesh
            /// Images will be saved in the Apps/MeshViewerDX/Data folder
            /// see (Gu et al. 2002)
            int lastSlash = meshin.FileName.LastIndexOf('\\');
            lastSlash = Math.Max(lastSlash, meshin.FileName.LastIndexOf('/'));
            string file = (lastSlash == -1) ? meshin.FileName : meshin.FileName.Substring(lastSlash + 1);
            string path = string.Format("./../../Data/gimg_{0}.png", file);
            gimg.Save(path, System.Drawing.Imaging.ImageFormat.Png);
            return gimg;
        }


        /// <summary>
        /// Compute matrix M_t for each triangle
        /// (see slides, lecture 7-2, #9)
        /// </summary>
        private static double[,] ComputeMatrixM(TriangleMesh.Vertex[] vertices)
        {
#if DEBUG
            Debug.Assert(vertices.Length == 3);
#endif
            double[,] M;

            /// TODO_A2 Task 4
            /// implement Least Squares Conformal Maps (LCSM)
            /// use this function to compute the matrix M per triangle
            /// 
            M = null;

            return M;
        }

        /// <summary>
        /// Creates a 2x3 matrix from two 1x3 row vectors
        /// </summary>
        /// <param name="r0">1x3 row vector</param>
        /// <param name="r1">1x3 row vector</param>
        /// <returns>2x3 rectangular matrix</returns>
        private static double[,] MatrixFromRowVectors(float[] r0, float[] r1)
        {
#if DEBUG
            Debug.Assert(r0.Length == 3);
            Debug.Assert(r1.Length == 3);
#endif
            var m = new double[2, 3]
            {
                {r0[0], r0[1], r0[2]},
                {r1[0], r1[1], r1[2]},
            };
            return m;
        }

        /// <summary>
        /// Creates a 2x3 matrix from two 1x3 row vectors
        /// </summary>
        /// <param name="r0">1x3 row vector</param>
        /// <param name="r1">1x3 row vector</param>
        /// <returns>2x3 rectangular matrix</returns>
        private static double[,] MatrixFromRowVectors(double[] r0, double[] r1)
        {
#if DEBUG
            Debug.Assert(r0.Length == 3);
            Debug.Assert(r1.Length == 3);
#endif
            var m = new double[2, 3]
            {
                {r0[0], r0[1], r0[2]},
                {r1[0], r1[1], r1[2]},
            };
            return m;
        }

        /// <summary>
        /// Multiply a 2x3 matrix with a 3x1 vector.
        /// </summary>
        /// <param name="m">2x3 matrix</param>
        /// <param name="v">3x1 vector</param>
        /// <returns>2x1 vector</returns>
        private static double[] MatrixMultiply(double[,] m, float[] v)
        {
#if DEBUG
            Debug.Assert(v.Length == 3);
            Debug.Assert(m.Length == 6);
#endif
            double[] value = new double[2]
            {
                m[0, 0] * v[0] + m[0, 1] * v[1] + m[0, 2] * v[2],
                m[1, 0] * v[0] + m[1, 1] * v[1] + m[1, 2] * v[2],
            };
            return value;
        }

        /// <summary>
        /// Multiply a 2x3 matrix with a 3x1 vector.
        /// </summary>
        /// <param name="m">2x3 matrix</param>
        /// <param name="v">3x1 vector</param>
        /// <returns>2x1 vector</returns>
        private static double[] MatrixMultiply(double[,] m, double[] v)
        {
#if DEBUG
            Debug.Assert(v.Length == 3);
            Debug.Assert(m.Length == 6);
#endif
            double[] value = new double[2]
            {
                m[0, 0] * v[0] + m[0, 1] * v[1] + m[0, 2] * v[2],
                m[1, 0] * v[0] + m[1, 1] * v[1] + m[1, 2] * v[2],
            };
            return value;
        }
    }
}
