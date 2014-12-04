using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using HelixToolkit.SharpDX;
using HelixToolkit.SharpDX.Wpf;
using SharpDX;

namespace Meshes.Utilities
{
    public static class Mesh3DUtilities
    {
        public enum Mesh3DShading
        {
            Flat,Smooth,
        }

        /// <summary>
        /// Converts the geometry to a <see cref="MeshGeometry3D"/>.
        /// </summary>
        /// <param name="freeze">
        /// freeze the mesh if set to <c>true</c>.
        /// </param>
        /// <returns>
        /// A mesh geometry.
        /// </returns>
        public static MeshGeometry3D ToMeshGeometry3D(this TriangleMesh mesh3d, Mesh3DShading shading = Mesh3DShading.Smooth, bool flipTriangles = false)
        {
            Vector3[] positions;
            Vector3[] normals;
            Vector3[] tangents = null;
            Vector3[] bitangents = null;
            Vector2[] texcoords;
            int[] indices;

            switch (shading)
            {
                default:
                case Mesh3DShading.Flat:
                {
                    positions = new Vector3[(3 * mesh3d.Faces.Count)];
                    normals = new Vector3[(3 * mesh3d.Faces.Count)];
                    texcoords = new Vector2[(3 * mesh3d.Faces.Count)];
                    indices = new int[(3 * mesh3d.Faces.Count)];
                    int idx = 0;
                    int ii = 0;

                    foreach (var f in mesh3d.Faces)
                    {
                        var triangle = f.Vertices;
                        if (flipTriangles)
                            triangle = triangle.Reverse();

                        foreach (var v in triangle)
                        {
                            positions[ii] = (v.Traits.Position);
                            normals[ii] = (f.Traits.Normal);
                            texcoords[ii] = (v.Traits.TextureCoordinate);
                            indices[ii] = (idx++);
                            ii++;
                        }
                    }
                    break;                
                }
                    
                case Mesh3DShading.Smooth:
                {
                    //positions = new Vector3[(mesh3d.Vertices.Count)];
                    //normals = new Vector3[(mesh3d.Vertices.Count)];
                    //texcoords = new Vector2[(mesh3d.Vertices.Count)];
                    indices = new int[(3 * mesh3d.Faces.Count)];
                                            
                    //for (int i = 0; i < mesh3d.Vertices.Count; i++)
                    //{
                    //    var v = mesh3d.Vertices[i].Traits;
                    //    positions[i] = (v.Position);
                    //    normals[i] = (v.Normal);
                    //    texcoords[i] = (v.TextureCoordinate);                        
                    //};
                    positions = mesh3d.Vertices.Select(x => x.Traits.Position).ToArray();
                    normals = mesh3d.Vertices.Select(x => x.Traits.Normal).ToArray();
                    texcoords = mesh3d.Vertices.Select(x => x.Traits.TextureCoordinate).ToArray();

                    if (mesh3d.Traits.HasTangents)
                    {
                        tangents = mesh3d.Vertices.Select(x => x.Traits.Tangent).ToArray();
                        bitangents = mesh3d.Vertices.Select(x => x.Traits.BiTangent).ToArray();
                    }                    
                    //indices = mesh3d.Vertices.Select(x => x.Index).ToArray();

                    int ii = 0;
                    for (int i = 0; i < mesh3d.Faces.Count; i++)
                    {
                        var triangle = mesh3d.Faces[i].Vertices.Select(x => x.Index);
                        if (flipTriangles)
                            triangle = triangle.Reverse();
                        foreach (var idx in triangle)
                            indices[ii++] = (idx);
                    }
                    break;
                }                
            }

            var g = new MeshGeometry3D
            {
                Positions = positions,
                Indices = indices,
                Normals = normals,
                Tangents = tangents,
                BiTangents = bitangents,
                TextureCoordinates = texcoords,
            };

            return g;
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="mesh3d"></param>
        /// <param name="shading"></param>
        /// <param name="flipTriangles"></param>
        /// <param name="freeze"></param>
        /// <returns></returns>
        public static LineGeometry3D ToLineGeometry3D(this TriangleMesh mesh3d)
        {
            var indices = new int[2 * mesh3d.Edges.Count];

            int ii = 0;
            foreach (var e in mesh3d.Edges)
            {
                indices[ii++] = (e.Vertex0.Index);
                indices[ii++] = (e.Vertex1.Index);
            }

            return new LineGeometry3D
            {
                Positions = mesh3d.Vertices.Select(x => x.Traits.Position).ToArray(),
                Indices = indices.ToArray(),
            };
        }

        /// <summary>
        /// Creates a new renderable MeshGeomerty3D out of a given TriangleMesh3D
        /// </summary>
        /// <param name="mgeo3d"></param>
        /// <param name="positionsBuffer"></param>
        public static MeshGeometry3D UpdateMeshGeometry3D(this TriangleMesh mesh3d, MeshGeometry3D mgeo3d, bool flatShading, bool flipTriangles = false, bool updateNormals = false)
        {
            var positions = mgeo3d.Positions;
            var normals = mgeo3d.Normals;

            mgeo3d.Positions = null;

            if (updateNormals)
            {
                mesh3d.ComputeNormals();
                mgeo3d.Normals = null;
            }

            if (flatShading)
            {
                int idx = 0;
                foreach (var f in mesh3d.Faces)
                {
                    var triangle = f.Vertices;
                    if (flipTriangles)
                        triangle = triangle.Reverse();

                    foreach (var v in triangle)
                    {                        
                        positions[idx] = (v.Traits.Position);
                        if (updateNormals)
                            normals[idx] = f.Traits.Normal;

                        idx++;
                    }
                }
            }
            else
            {                
                for (int i = 0; i < mesh3d.Vertices.Count; i++)
                {
                    var v = mesh3d.Vertices[i].Traits;                    
                    positions[i] = (v.Position);

                    if (updateNormals)
                        normals[i] = v.Normal;
                };
            }

            mgeo3d.Positions = positions;
            if (updateNormals)            
                mgeo3d.Normals = normals;
            
            return mgeo3d;
        }


        #region TriangleMesh3D Creation

#if OBJREADER
        /// <summary>        
        /// </summary>
        /// <param name="filename"></param>
        /// <returns></returns>
        public static TriangleMesh3D LoadMesh(string filename)
        {
            if (System.IO.Path.GetExtension(filename).ToLower() != ".obj")
                throw new ArgumentException("Currently only .obj files supported");

            var reader = new ObjReader();
            var mesh = new TriangleMesh3D();
            reader.Read(filename);
            foreach (var group in reader.Groups)
            {
                var mb = group.MeshBuilder;
                CreateMesh(mesh, mb.Positions, mb.TriangleIndices, mb.Normals, mb.TextureCoordinates);
                return mesh;
            }

            return mesh;
        } 
#endif

        /// <summary>
        /// 
        /// </summary>
        /// <param name="mg"></param>
        /// <returns></returns>
        public static TriangleMesh CreateMesh(MeshGeometry3D mg)
        {
            var mesh = new TriangleMesh();
            CreateMesh(mesh, mg.Positions, mg.Indices, mg.Normals, mg.TextureCoordinates);
            return mesh;
        }


        /// <summary>
        /// 
        /// </summary>
        private static void CreateMesh(TriangleMesh mesh, IList<Vector3> positions, IList<int> triangleIndices, IList<Vector3> normals = null, IList<Vector2> texcoords = null)
        {
            if (normals != null)
            {
                if (normals.Count != positions.Count)
                    throw new InvalidOperationException("Normals do not fit the point set.");
            }
            if (texcoords != null)
            {
                if (texcoords.Count != positions.Count)
                    throw new InvalidOperationException("Texcoords do not fit the point set.");
            }

            int startVertex = mesh.Vertices.Count;

            for (int i = 0; i < positions.Count; i++)
            {
                var v = new VertexTraits(positions[i]);
                if (normals != null)
                    v.Normal = normals[i];
                if (texcoords != null)
                    v.TextureCoordinate = texcoords[i];

                mesh.Vertices.Add(v);
            }

            for (int i = 0; i < triangleIndices.Count; i += 3)
            {
                var tri = new[]
                {
                    mesh.Vertices[startVertex+triangleIndices[i]],
                    mesh.Vertices[startVertex+triangleIndices[i+1]],
                    mesh.Vertices[startVertex+triangleIndices[i+2]],
                };
                mesh.Faces.Add(tri);                
            }
        } 
        #endregion
    }
}
