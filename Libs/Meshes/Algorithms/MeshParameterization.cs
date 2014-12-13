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
using Matrix = SharpDX.Matrix;
using MeshVertex = Meshes.Generic.Mesh<Meshes.NullTraits, Meshes.FaceTraits, Meshes.HalfedgeTraits, Meshes.VertexTraits>.Vertex;
using MeshFace = Meshes.Generic.Mesh<Meshes.NullTraits, Meshes.FaceTraits, Meshes.HalfedgeTraits, Meshes.VertexTraits>.Face;
using MeshHalfEdge = Meshes.Generic.Mesh<Meshes.NullTraits, Meshes.FaceTraits, Meshes.HalfedgeTraits, Meshes.VertexTraits>.Halfedge;
using SparseMatrix = CSparse.Double.SparseMatrix;

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
            static readonly QuadSide Right = new QuadSide(t => 1d, t => t, 0.25d);
            static readonly QuadSide Top = new QuadSide(t => 1d - t, t => 1d, 0.5d);
            static readonly QuadSide Left = new QuadSide(t => 0d, t => 1d - t, 0.75d);

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

            // TODO : For geometry images, L mapped edges require splitting. Adaptive length parameterization should be sufficient for crack prediction however
            FixBoundaryToShape(boundaryVertices, bu, bv);

            var laplacian = MeshLaplacian.SelectedLaplacian == MeshLaplacian.Type.Harmonic ?
                MeshLaplacian.CreateBoundedHarmonicLaplacian(meshin, 1d, 0d, true) :
                MeshLaplacian.SelectedLaplacian == MeshLaplacian.Type.MeanValue ?
                    MeshLaplacian.CreateBoundedMeanLaplacian(meshin, 1d, 0d, true) :
                    MeshLaplacian.CreateBoundedUniformLaplacian(meshin, 1d, 0d, true);

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

            foreach (var face in meshin.Faces)
            {
                var v1_global = face.Vertices.ElementAt(0).Traits.Position;
                var v2_global = face.Vertices.ElementAt(1).Traits.Position;
                var v3_global = face.Vertices.ElementAt(2).Traits.Position;

                var xDir = v2_global - v1_global;
                var skewedZDir = v3_global - v1_global;
                var yDir = Vector3.Cross(xDir, skewedZDir);

                xDir.Normalize();
                yDir.Normalize();

                var zDir = Vector3.Cross(yDir, xDir);

                var transform = new Matrix(new[]
                    {
                        xDir.X, xDir.Y, xDir.Z, 0,
                        yDir.X, yDir.Y, yDir.Z, 0,
                        zDir.X, zDir.Y, zDir.Z, 0,
                        0, 0, 0, 1,
                    });
                transform.Transpose();

                var v1 = Vector3.Transform(v1_global, transform);
                var v2 = Vector3.Transform(v2_global, transform);
                var v3 = Vector3.Transform(v3_global, transform);
            
                var areaTriangle =
                    ((double) v1.X * v2.Z - (double) v1.Z * v2.X) +
                    ((double) v2.X * v3.Z - (double) v2.Z * v3.X) +
                    ((double) v3.X * v1.Z - (double) v3.Z * v1.X);

                var mImaginary = new Vector3(v3.Z - v2.Z, v1.Z - v3.Z, v2.Z - v1.Z) * (float) (1d / areaTriangle);
                var mReal = new Vector3(v3.X - v2.X, v1.X - v3.X, v2.X - v1.X) * (float) (1d / areaTriangle);
               
                var subIndex = 0;
                
                // Expected x layout : u1 v1 u2 v2...ui vi ui+2 vi+2...uj vj uj+2 vj+2...un vn
                foreach (var vertex in face.Vertices)
                {
                    if (vertex.Index == P1Index || vertex.Index == P2Index)
                    {
                        var entryIndex = vertex.Index == P1Index ? 0 : 1;

                        // -R*MT * u (dx,dy)
                        A2.Entry(2 * face.Index, 2 * entryIndex, -mReal[subIndex]);
                        A2.Entry(2 * face.Index + 1, 2 * entryIndex, -mImaginary[subIndex]);

                        // MT * v (dx,dy)
                        A2.Entry(2 * face.Index, 2 * entryIndex + 1, mImaginary[subIndex]);
                        A2.Entry(2 * face.Index + 1, 2 * entryIndex + 1, -mReal[subIndex]);
                    }
                    else
                    {
                        var entryIndex = AdaptedIndexFor(vertex.Index);
                        
                        // -R*MT * u (dx,dy)
                        A1.Entry(2 * face.Index, 2 * entryIndex, mReal[subIndex]);
                        A1.Entry(2 * face.Index + 1, 2 * entryIndex, mImaginary[subIndex]);

                        // MT * v (dx,dy)
                        A1.Entry(2 * face.Index, 2 * entryIndex + 1, -mImaginary[subIndex]);
                        A1.Entry(2 * face.Index + 1, 2 * entryIndex + 1, mReal[subIndex]);
                    }

                    subIndex++;
                }
            }

            double[] bPrime;
            A2.Compress().Ax(b, out bPrime);

            var solver = QR.Create(A1.Compress());
            solver.Solve(bPrime);

            for (var vertIndex = 0; vertIndex < n; vertIndex++)
            {
                if (vertIndex == P1Index)
                {
                    bu[vertIndex] = P1UV[0];
                    bv[vertIndex] = P1UV[1];
                } else if (vertIndex == P2Index)
                {
                    bu[vertIndex] = P2UV[0];
                    bv[vertIndex] = P2UV[1];
                }
                else
                {
                    var adaptedIndex = AdaptedIndexFor(vertIndex);
                    bu[vertIndex] = bPrime[2 * adaptedIndex];
                    bv[vertIndex] = bPrime[2 * adaptedIndex + 1];
                }
            }

            /// update mesh positions and uv's
            MeshLaplacian.UpdateMesh(meshout, bu, bv, b0, bu, bv);
            MeshLaplacian.UpdateMesh(meshin, bu, bv);
        }

        private int AdaptedIndexFor(int vertexIndex)
        {
            return vertexIndex - (vertexIndex > P1Index ? 1 : 0) - (vertexIndex > P2Index ? 1 : 0);
        }

        /// <summary>
        /// The Direct Conformal Parameterization (DCP) method, see (Desbrun et al. 2002)
        /// </summary>
        /// <param name="meshin"></param>
        /// <param name="meshout"></param>
        private void DCP(TriangleMesh meshin, out TriangleMesh meshout)
        {
            MeshLaplacian.PrecomputeTraits(meshin);

            /// copy the mesh
            meshout = meshin.Copy();

            /// counters
            var vertexCount = meshout.Vertices.Count;

            /// output uv-coordinates
            var bu = new double[vertexCount];
            var bv = new double[vertexCount];
            var b0 = new double[vertexCount];

            // A * x = b
            var x = new double[vertexCount * 2 + 4];

            var M_A = new TripletMatrix(2 * (vertexCount + 2), 2 * vertexCount, 4 * (vertexCount + 1) * vertexCount);
            var M_X = new TripletMatrix(2 * (vertexCount + 2), 2 * vertexCount, 4 * (vertexCount + 1) * vertexCount);

            foreach (var vertex in meshin.Vertices.Where(v => !v.OnBoundary))
            {
                var areaWeightSum = 0d;
                var angleWeightSum = 0d;

                foreach (var halfEdge in vertex.Halfedges)
                {
                    // cot(alpha) + cot(beta)
                    var areaWeight = halfEdge.Previous.Traits.Cotan + halfEdge.Opposite.Previous.Traits.Cotan;
                    // cot(gamma) + cot(delta)
                    var angleWeight = (halfEdge.Next.Traits.Cotan + halfEdge.Opposite.Traits.Cotan);
                    angleWeight /= (halfEdge.FromVertex.Traits.Position - halfEdge.ToVertex.Traits.Position).LengthSquared();

                    M_A.Entry(vertex.Index * 2, halfEdge.ToVertex.Index * 2, areaWeight);
                    M_A.Entry(vertex.Index * 2 + 1, halfEdge.ToVertex.Index * 2 + 1, areaWeight);
                    M_X.Entry(vertex.Index * 2, halfEdge.ToVertex.Index * 2, angleWeight);
                    M_X.Entry(vertex.Index * 2 + 1, halfEdge.ToVertex.Index * 2 + 1, angleWeight);

                    areaWeightSum += areaWeight;
                    angleWeightSum += angleWeight;
                }

                M_A.Entry(vertex.Index * 2, vertex.Index * 2, -areaWeightSum);
                M_A.Entry(vertex.Index * 2 + 1, vertex.Index * 2 + 1, -areaWeightSum);
                M_X.Entry(vertex.Index * 2, vertex.Index * 2, -angleWeightSum);
                M_X.Entry(vertex.Index * 2 + 1, vertex.Index * 2 + 1, -angleWeightSum);
            }



            // Free boundary
            foreach (var vertex in meshin.Vertices.Where(v => v.OnBoundary))
            {
                var weightSum = 0d;
                
                foreach (var halfEdge in vertex.Halfedges.Where(he => !he.Edge.OnBoundary))
                {
                    // cot(alpha) + cot(beta)
                    var borderWeight = halfEdge.Previous.Traits.Cotan + halfEdge.Opposite.Previous.Traits.Cotan;

                    M_A.Entry(vertex.Index * 2, halfEdge.ToVertex.Index * 2, -borderWeight);
                    M_A.Entry(vertex.Index * 2 + 1, halfEdge.ToVertex.Index * 2 + 1, -borderWeight);
                    M_X.Entry(vertex.Index * 2, halfEdge.ToVertex.Index * 2, -borderWeight);
                    M_X.Entry(vertex.Index * 2 + 1, halfEdge.ToVertex.Index * 2 + 1, -borderWeight);

                    weightSum += borderWeight;
                }

                foreach (var halfEdge in vertex.Halfedges.Where(he => he.Edge.OnBoundary))
                {
                    if (halfEdge.OnBoundary)
                    {
                        // cot(alpha) + cot(beta)
                        var borderWeight = halfEdge.Previous.Traits.Cotan;

                        M_A.Entry(vertex.Index * 2, halfEdge.ToVertex.Index * 2, -borderWeight);
                        M_A.Entry(vertex.Index * 2, halfEdge.ToVertex.Index * 2 + 1, -1);
                        M_A.Entry(vertex.Index * 2 + 1, halfEdge.ToVertex.Index * 2 + 1, -borderWeight);
                        M_A.Entry(vertex.Index * 2 + 1, halfEdge.ToVertex.Index * 2, 1);

                        M_X.Entry(vertex.Index * 2, halfEdge.ToVertex.Index * 2, -borderWeight);
                        M_X.Entry(vertex.Index * 2, halfEdge.ToVertex.Index * 2 + 1, -1);
                        M_X.Entry(vertex.Index * 2 + 1, halfEdge.ToVertex.Index * 2 + 1, -borderWeight);
                        M_X.Entry(vertex.Index * 2 + 1, halfEdge.ToVertex.Index * 2 + 1, 1);

                        weightSum += borderWeight;
                    }
                    else
                    {
                        // cot(alpha) + cot(beta)
                        var borderWeight = halfEdge.Opposite.Previous.Traits.Cotan;

                        M_A.Entry(vertex.Index * 2, halfEdge.ToVertex.Index * 2, -borderWeight);
                        M_A.Entry(vertex.Index * 2, halfEdge.ToVertex.Index * 2 + 1, 1);
                        M_A.Entry(vertex.Index * 2 + 1, halfEdge.ToVertex.Index * 2 + 1, -borderWeight);
                        M_A.Entry(vertex.Index * 2 + 1, halfEdge.ToVertex.Index * 2, -1);

                        M_X.Entry(vertex.Index * 2, halfEdge.ToVertex.Index * 2, -borderWeight);
                        M_X.Entry(vertex.Index * 2, halfEdge.ToVertex.Index * 2 + 1, 1);
                        M_X.Entry(vertex.Index * 2 + 1, halfEdge.ToVertex.Index * 2 + 1, -borderWeight);
                        M_X.Entry(vertex.Index * 2 + 1, halfEdge.ToVertex.Index * 2 + 1, -1);

                        weightSum += borderWeight;
                    }
                }

                M_A.Entry(vertex.Index * 2, vertex.Index * 2, weightSum);
                M_A.Entry(vertex.Index * 2 + 1, vertex.Index * 2 + 1, weightSum);
                M_X.Entry(vertex.Index * 2, vertex.Index * 2, weightSum);
                M_X.Entry(vertex.Index * 2 + 1, vertex.Index * 2 + 1, weightSum);
            }

            // Fixed vertices

            // M^n
            M_A.Entry(2 * vertexCount, 2 * P1Index, 1);
            M_A.Entry(2 * vertexCount + 1, 2 * P1Index + 1, 1);
            M_X.Entry(2 * vertexCount, 2 * P1Index, 1);
            M_X.Entry(2 * vertexCount + 1, 2 * P1Index + 1, 1);

            M_A.Entry(2 * vertexCount + 2, 2 * P2Index, 1);
            M_A.Entry(2 * vertexCount + 3, 2 * P2Index + 1, 1);
            M_X.Entry(2 * vertexCount + 2, 2 * P2Index, 1);
            M_X.Entry(2 * vertexCount + 3, 2 * P2Index + 1, 1);

            // M^n transp
            M_A.Entry(2 * P1Index, 2 * vertexCount, 1);
            M_A.Entry(2 * P1Index + 1, 2 * vertexCount + 1, 1);
            M_X.Entry(2 * P1Index, 2 * vertexCount, 1);
            M_X.Entry(2 * P1Index + 1, 2 * vertexCount + 1, 1);

            M_A.Entry(2 * P2Index, 2 * vertexCount + 2, 1);
            M_A.Entry(2 * P2Index + 1, 2 * vertexCount + 3, 1);
            M_X.Entry(2 * P2Index, 2 * vertexCount + 2, 1);
            M_X.Entry(2 * P2Index + 1, 2 * vertexCount + 3, 1);

            // b^n
            x[2 * vertexCount] = P1UV.X;
            x[2 * vertexCount + 1] = P1UV.Y;
            x[2 * vertexCount + 2] = P2UV.X;
            x[2 * vertexCount + 3] = P2UV.Y;

            var matrix = SparseMatrix.Add(M_A.Compress(), M_X.Compress(), 0d, 1d);
            var solver = QR.Create(matrix);
            solver.Solve(x);

            for (var vertexIndex = 0; vertexIndex < vertexCount; vertexIndex++)
            {
                bu[vertexIndex] = x[2 * vertexIndex];
                bv[vertexIndex] = x[2 * vertexIndex + 1];
            }

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

            var minimum = meshin.Traits.BoundingBox.Minimum;
            var extents = meshin.Traits.BoundingBox.Maximum - meshin.Traits.BoundingBox.Minimum;

            // Use naive rasterizing approach, don't want to implement scanline algorithm : https://fgiesen.wordpress.com/2013/02/08/triangle-rasterization-in-practice/
            foreach (var face in meshin.Faces)
            {
                var a = face.Vertices.ElementAt(0).Traits;
                var b = face.Vertices.ElementAt(1).Traits;
                var c = face.Vertices.ElementAt(2).Traits;

                var triangleBounds = BoundingBox.FromPoints(new[]
                {
                    new Vector3(a.TextureCoordinate, 0),
                    new Vector3(b.TextureCoordinate, 0),
                    new Vector3(c.TextureCoordinate, 0),
                });

                triangleBounds.Maximum.X = (float)Math.Round(triangleBounds.Maximum.X * 512d);
                triangleBounds.Maximum.Y = (float)Math.Round(triangleBounds.Maximum.Y * 512d);
                triangleBounds.Minimum.X = (float)Math.Round(triangleBounds.Minimum.X * 512d);
                triangleBounds.Minimum.Y = (float)Math.Round(triangleBounds.Minimum.Y * 512d);

                for (var ptX = (int) triangleBounds.Minimum.X; ptX < triangleBounds.Maximum.X; ptX++)
                {
                    for (var ptY = (int) triangleBounds.Minimum.Y; ptY < triangleBounds.Maximum.Y; ptY++)
                    {
                        var current = new Vector2((float) (ptX/512d + 1d/1024d), (float) (ptY/512d + 1d/1024d));

                        var det = GetBarycentricCoordinate(a.TextureCoordinate, b.TextureCoordinate, c.TextureCoordinate);
                        var wA = GetBarycentricCoordinate(b.TextureCoordinate, c.TextureCoordinate, current) / det;
                        var wB = GetBarycentricCoordinate(c.TextureCoordinate, a.TextureCoordinate, current) / det;
                        var wC = GetBarycentricCoordinate(a.TextureCoordinate, b.TextureCoordinate, current) / det;

                        if (wA < 0 || wB < 0 || wC < 0)
                            continue;
                       
                        var position = a.Position*wA + b.Position*wB + c.Position*wC - minimum;
                        gimg.SetPixel(ptX, ptY, System.Drawing.Color.FromArgb((int)((position.X / extents.X) * 255), (int)((position.Y / extents.Y) * 255), (int)((position.Z / extents.Z) * 255)));
                    }
                }
            }

            int lastSlash = meshin.FileName.LastIndexOf('\\');
            lastSlash = Math.Max(lastSlash, meshin.FileName.LastIndexOf('/'));
            string file = (lastSlash == -1) ? meshin.FileName : meshin.FileName.Substring(lastSlash + 1);
            string path = string.Format("./../../Data/gimg_{0}.png", file);
            gimg.Save(path, System.Drawing.Imaging.ImageFormat.Png);
            return gimg;
        }

        float GetBarycentricCoordinate(Vector2 v0, Vector2 v1, Vector2 pt)
        {
            return (v0.Y - v1.Y) * (pt.X - v1.X) + (v1.X - v0.X) * (pt.Y - v1.Y);
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

            var v1_global = vertices.ElementAt(0).Traits.Position;
            var v2_global = vertices.ElementAt(1).Traits.Position;
            var v3_global = vertices.ElementAt(2).Traits.Position;

            var xDir = v2_global - v1_global;
            var skewedZDir = v3_global - v1_global;
            var yDir = Vector3.Cross(xDir, skewedZDir);
            var zDir = Vector3.Cross(yDir, xDir);

            xDir.Normalize();
            yDir.Normalize();

            var transform = new Matrix(new[]
                {
                    xDir.X, xDir.Y, xDir.Z, -v1_global.X,
                    yDir.X, yDir.Y, yDir.Z, -v1_global.Y,
                    zDir.X, zDir.Y, zDir.Z, -v1_global.Z,
                    0, 0, 0, 1,
                });

            var v1 = Vector3.Transform(v1_global, transform);
            var v2 = Vector3.Transform(v2_global, transform);
            var v3 = Vector3.Transform(v3_global, transform);
            
            var areaTriangle =
                v1.X * v2.Y - v1.Y * v2.X +
                v2.X * v3.Y - v2.Y * v3.X +
                v3.X * v1.Y - v3.Y * v1.X;

            var rowVec1 = new Vector3(v2.Y - v3.Y, v3.Y - v1.Y, v1.Y - v2.Y) * 1f / areaTriangle;
            var rowVec2 = new Vector3(v3.Y - v2.Y, v1.Y - v3.Y, v2.Y - v1.Y) * 1f / areaTriangle;

            M = MatrixFromRowVectors(rowVec1.ToArray(), rowVec2.ToArray());

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
