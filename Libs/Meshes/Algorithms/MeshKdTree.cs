using SharpDX;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace Meshes.Algorithms
{
    public class MeshKdTree
    {
        /// <summary>
        /// Encapsulates potential hits of the ray. 
        /// </summary>
        public class Hit
        {
            public TriangleMesh.Face Triangle;
            public Vector3 Point;
        }

        /// <summary>
        /// Generate a simple boundig box based kd-tree for the mesh
        /// </summary>        
        public MeshKdTree(TriangleMesh mesh)
            : this(GetBoxes(mesh.Faces), 0)
        {
        }

        /// <summary>
        /// Checks the kd-tree for intersections with the ray. 
        /// Returns a list of triangles which have potentially intersections with the ray. 
        /// Note: it is not sure that all trianges in the list have a realy intersection with Ray.
        /// </summary>
        /// <param name="ray">The ray to intersect with the triangle mesh</param>
        /// <returns>List of triangles which have potentially intersections with the ray</returns>
        public IList<TriangleMesh.Face> Intersect(ref Ray ray)
        {
            var triangles = new List<TriangleMesh.Face>();
            this.Intersect(ref ray, ref triangles);
            return triangles;
        }

        /// <summary>
        /// 
        /// </summary>
        private MeshKdTree(IEnumerable<Node> nodes, int dim)
        {            
            this.Count = nodes.Count();
            this.Nodes = nodes;
            this.BBox = BoundingBox.FromPoints(nodes.AsParallel().SelectMany(x => x.BBox.GetCorners()).ToArray());

            if (this.Count > DivisionThreshold)
            {
                nodes = nodes.OrderBy(x => x.BBox.Minimum[dim]);

                var left = nodes.Take(Count / 2).ToList();
                var right = nodes.Skip(Count / 2).Take(Count).ToList();

                dim = ++dim % 3;
                this.Left = new MeshKdTree(left, dim);
                this.Right = new MeshKdTree(right, dim);
            }
            else
            {
                this.IsTerminal = true;
            }
        }

        /// <summary>
        /// 
        /// </summary>
        private void Intersect(ref Ray ray, ref List<TriangleMesh.Face> triangles)
        {
            if (this.IsTerminal)
            {
                // terminal node
                foreach (var t in this.Nodes)
                {
                    triangles.Add(t.Triangle);
                }
            }
            else
            {
                // non-terminal node
                if (this.Left.BBox.Intersects(ref ray))
                {
                    this.Left.Intersect(ref ray, ref triangles);
                }

                if (this.Right.BBox.Intersects(ref ray))
                {
                    this.Right.Intersect(ref ray, ref triangles);
                }
            }
        }

        /// <summary>
        /// 
        /// </summary>
        private void Intersect(ref Ray ray, ref List<Hit> hits)
        {

            if (this.Count == 1)
            {
                // terminal node
                foreach (var t in this.Nodes)
                {
                    var vs = t.Triangle.Vertices.ToList();
                    Vector3 hit;
                    if (ray.Intersects(ref vs[0].Traits.Position, ref vs[1].Traits.Position, ref vs[2].Traits.Position, out hit))
                    {
                        hits.Add(new Hit()
                        {
                            Triangle = t.Triangle,
                            Point = hit,
                        });
                    }
                }
            }
            else
            {
                // non-terminal node
                if (this.Left.BBox.Intersects(ref ray))
                {
                    this.Left.Intersect(ref ray, ref hits);
                }

                if (this.Right.BBox.Intersects(ref ray))
                {
                    this.Right.Intersect(ref ray, ref hits);
                }
            }
        }
      
        /// <summary>
        /// Creates nodes that wrapp each face with a bouonding box
        /// </summary>
        private static IEnumerable<Node> GetBoxes(IEnumerable<TriangleMesh.Face> faces)
        {            
            foreach (var tri in faces.AsParallel())
            {
                var node = new Node()
                {
                    Triangle = tri,
                    BBox = BoundingBox.FromPoints(tri.Vertices.Select(x => x.Traits.Position).ToArray()),
                };
                yield return node;
            }            
        }

        private const int DivisionThreshold = 10;
        private bool IsTerminal = false;
        private int Count;
        private BoundingBox BBox;
        private MeshKdTree Left, Right;
        private IEnumerable<Node> Nodes;
        
        private class Node
        {
            public BoundingBox BBox;
            public TriangleMesh.Face Triangle;
        }
    }
}
