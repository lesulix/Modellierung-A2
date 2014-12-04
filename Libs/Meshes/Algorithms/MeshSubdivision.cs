using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using Meshes;
using Meshes.Generic;
using SharpDX;

namespace Meshes.Algorithms
{
    /// <summary>
    /// This class implements various methods for subdivsion of triangle methods. 
    /// </summary>
    public class MeshSubdivision
    {
        public enum Method
        {
            Loop, Butterfly, Sqrt3,
        }

        /// <summary>
        /// Collection of supported subdivision methods
        /// </summary>
        public static IList<string> MethodCollection = new List<string>()
        {
            Method.Loop.ToString(),
            Method.Butterfly.ToString(),
            Method.Sqrt3.ToString(),                
        };

        /// <summary>
        /// Private constructor of the singleton
        /// </summary>
        private MeshSubdivision()
        {
            this.Steps = 1;
            this.DisplacementFactor = 1.0f;
            this.DetailMesh = null;
        }

        /// <summary>
        /// number of subdivsion steps to perform
        /// </summary>
        public int Steps { get; set; }

        /// <summary>
        /// Factor for displacement mapping
        /// </summary>
        public float DisplacementFactor { get; set; }

        /// <summary>
        /// Detailed Mesh for calculating displacement
        /// </summary>
        public TriangleMesh DetailMesh { get; set; }

        /// <summary>
        /// Filename for the displacement map
        /// </summary>
        public string DisplacementFileName { get; set; }

        /// <summary>
        /// Singleton Instance
        /// </summary>
        public static readonly MeshSubdivision Instance = new MeshSubdivision();

        /// <summary>
        /// Selected Subdivision Method
        /// </summary>
        public Method SelectedMethod { get; set; }

        /// <summary>
        /// Current subdivison level
        /// </summary>
        public int SubdivisionLevel { get; set; }

        /// <summary>
        /// Main Subdivision method
        /// </summary>
        /// <param name="inputMesh"></param>
        /// <returns></returns>
        public TriangleMesh Subdivision(TriangleMesh inputMesh)
        {
            switch (SelectedMethod)
            {
                case Method.Loop:
                    return LoopSubdivision(inputMesh);
                case Method.Butterfly:
                    return ButterflySubdivision(inputMesh);
                case Method.Sqrt3:
                    return Sqrt3Subdivision(inputMesh);
                default:
                    return null;
            }
        }

        /// <summary>
        /// Perform Loop-Subdivision scheme
        /// </summary>
        /// <param name="inputMesh"></param>
        /// <returns></returns>
        private TriangleMesh LoopSubdivision(TriangleMesh inputMesh)
        {
            TriangleMesh subdividedMesh = new TriangleMesh();
            subdividedMesh.FileName = inputMesh.FileName;

            return subdividedMesh;
        }

        /// <summary>
        /// Perform Butterfly-Subdivision scheme
        /// </summary>
        /// <param name="inputMesh"></param>
        /// <returns></returns>
        private TriangleMesh ButterflySubdivision(TriangleMesh inputMesh)
        {
            TriangleMesh subdividedMesh = new TriangleMesh();
            subdividedMesh.FileName = inputMesh.FileName;


            return subdividedMesh;
        }


        /// <summary>
        /// Perform Sqrt3-Subdivision scheme
        /// </summary>
        /// <param name="inputMesh"></param>
        /// <returns></returns>
        private TriangleMesh Sqrt3Subdivision(TriangleMesh inputMesh)
        {
            TriangleMesh subdividedMesh = new TriangleMesh();
            subdividedMesh.FileName = inputMesh.FileName;

            return subdividedMesh;
        }

        /// <summary>
        /// Implements exact evaluation of Loop subdivision (cf. Jos Stam 1998)
        /// </summary>
        /// <param name="inputMesh"></param>
        /// <returns></returns>
        private TriangleMesh LoopExactEvaluation(TriangleMesh inputMesh)
        {
            TriangleMesh subdividedMesh = new TriangleMesh();
            subdividedMesh.FileName = inputMesh.FileName;

            return subdividedMesh;
        }


        /// <summary>
        /// Calculate the face normal for the given face
        /// </summary>
        /// <param name="face"></param>
        public static void CalculateFaceNormal(TriangleMesh.Face face)
        {
            var face_normal = Vector3.Zero;
            int c = 0;
            foreach (var v in face.Vertices)
            {
                face_normal += v.Traits.Normal;
                c++;
            }
            face_normal /= c;
            face.Traits.Normal = face_normal;
        }

        /// <summary>
        /// Generates and saves a displacement map
        /// </summary>
        /// <param name="subdividedMesh"></param>
        /// <param name="detailMesh"></param>
        /// <param name="file_name"></param>
        public static void GenerateDisplacementMap(TriangleMesh subdividedMesh)
        {
            var kdtree = new MeshKdTree(Instance.DetailMesh);

        }

        /// <summary>
        /// Applies the displacement map of the given file multiplied with the given factor
        /// </summary>
        /// <param name="mesh"></param>
        /// <param name="file_name"></param>
        /// <param name="displacement_factor"></param>
        public static void ApplyDisplacement(TriangleMesh mesh)
        {

        }


    }
}