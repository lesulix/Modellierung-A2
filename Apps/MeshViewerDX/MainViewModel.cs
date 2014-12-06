namespace MeshViewerDX
{
    using System;
    using System.IO;
    using System.Linq;
    using System.Windows;
    using System.Windows.Input;
    using Media3D = System.Windows.Media.Media3D;
    using Point3D = System.Windows.Media.Media3D.Point3D;
    using Transform3D = System.Windows.Media.Media3D.Transform3D;
    using Vector3D = System.Windows.Media.Media3D.Vector3D;
    using HelixToolkit.SharpDX;
    using HelixToolkit.SharpDX.Wpf;
    using Meshes;
    using Meshes.Utilities;
    using Meshes.Algorithms;
    using SharpDX;
    using System.Threading.Tasks;
    using System.Collections.Generic;
    using System.Windows.Media;
    using Color = SharpDX.Color;
    using System.Windows.Media.Imaging;
    using System.Windows.Controls;
    using System.Windows.Shapes;
    using System.Collections.ObjectModel;
    using System.Globalization;


    public class MainViewModel : BaseViewModel
    {
        #region Properties

        public MeshGeometry3D MeshModel { get; private set; }

        public LineGeometry3D MeshEdges { get; private set; }

        public LineGeometry3D Box { get; private set; }

        public LineGeometry3D BorderEdges { get; private set; }

        public LineGeometry3D Grid { get; private set; }

        public BitmapImage DefaultTextureImage { get; private set; }

        public BitmapImage NormalMapImage { get; private set; }

        public DrawingImage MeshDrawingImage { get; private set; }

        public IList<Point> BorderPoints { get; private set; }

        public PhongMaterial MeshModelMaterial { get; private set; }

        public PhongMaterial GreenMaterial { get; private set; }

        public PhongMaterial BlueMaterial { get; private set; }

        public Color GridColor { get; private set; }

        public Transform3D MeshModelTransform { get; private set; }

        public Transform3D GridTransform { get; private set; }

        public Vector3 DirectionalLight1Direction { get; private set; }

        public Vector3 DirectionalLight2Direction { get; private set; }

        public Vector3 DirectionalLight3Direction { get; private set; }

        public Color4 DirectionalLightColor { get; private set; }

        public Color4 AmbientLightColor { get; private set; }

        public IList<string> RenderTechniques { get { return renderTechniques; } }

        public IList<string> TextureImages { get { return textureImages; } }

        public ICommand OpenCmd { get; private set; }
        public ICommand SmoothCmd { get; private set; }
        public ICommand ResetCmd { get; private set; }
        public ICommand ParameterizeCmd { get; private set; }
        public ICommand SubdivisionCmd { get; private set; }
        public ICommand LoadDetailModelCmd { get; private set; }
        public ICommand LoadDisplacementCmd { get; private set; }
        public ICommand SaveDisplacementCmd { get; private set; }
        public ICommand GenerateDisplacementCmd { get; private set; }
        public ICommand ApplyDisplacementCmd { get; private set; }
        public ICommand OpenNormalMapCmd { get; private set; }
        public ICommand CreateGeometryImageCmd { get; private set; }

        public TriangleMesh Mesh { get; private set; }

        public double MinCurv { get; private set; }

        public double MaxCurv { get; private set; }

        public bool FlatShading
        {
            get { return this.flatShading; }
            set
            {
                this.flatShading = value;
                if (this.Mesh != null)
                {
                    this.UpdateModel(this.Mesh, this.flatShading ? Mesh3DUtilities.Mesh3DShading.Flat : Mesh3DUtilities.Mesh3DShading.Smooth);
                }
            }
        }

        public MeshSmoothing SmoothingSolver
        {
            get { return MeshSmoothing.Instance; }
        }

        public MeshParameterization ParameterizationSolver
        {
            get { return MeshParameterization.Instance; }
        }

        public MeshSubdivision SubdivisionSolver
        {
            get { return MeshSubdivision.Instance; }
        }

        public bool LaplacianNormalize
        {
            get { return MeshLaplacian.LaplacianNormalize; }
            set { MeshLaplacian.LaplacianNormalize = value; }
        }

        public string Filename { get; private set; }


        public string Student { get; private set; }


        public bool IsExplicit
        {
            get { return this.SmoothingSolver.SelectedMethod == MeshSmoothing.Method.Explicit_Euler; }
        }

        public IList<string> LaplacianTypes
        {
            get { return MeshLaplacian.TypeCollection; }
        }

        public IList<string> SmoothingMethods
        {
            get { return MeshSmoothing.MethodCollection; }
        }

        public IList<string> BoundaryTypes
        {
            get { return MeshParameterization.BoundaryTypeCollection; }
        }

        public IList<string> SubdivisionMethods
        {
            get { return MeshSubdivision.MethodCollection; }
        }

        public IList<string> ParameterizationMethods
        {
            get { return MeshParameterization.MethodCollection; }
        }

        public string SelectedLaplacian
        {
            get { return MeshLaplacian.SelectedLaplacian.ToString(); }
            set
            {
                if (value == MeshLaplacian.Type.Cotan.ToString())
                {
                    MeshLaplacian.SelectedLaplacian = MeshLaplacian.Type.Cotan;
                }
                else if (value == MeshLaplacian.Type.Harmonic.ToString())
                {
                    MeshLaplacian.SelectedLaplacian = MeshLaplacian.Type.Harmonic;
                }
                else if (value == MeshLaplacian.Type.MeanValue.ToString())
                {
                    MeshLaplacian.SelectedLaplacian = MeshLaplacian.Type.MeanValue;
                }
                else
                {
                    MeshLaplacian.SelectedLaplacian = MeshLaplacian.Type.Uniform;
                }

                this.SmoothingSolver.Clear();
            }
        }

        public string SelectedParameterizationMethod
        {
            get { return MeshParameterization.Instance.SelectedMethod.ToString(); }
            set
            {
                if (value == MeshParameterization.Method.LSCM.ToString())
                {
                    this.ParameterizationSolver.SelectedMethod = MeshParameterization.Method.LSCM;
                }
                else if (value == MeshParameterization.Method.DCP.ToString())
                {
                    this.ParameterizationSolver.SelectedMethod = MeshParameterization.Method.DCP;
                }
                else if (value == MeshParameterization.Method.LinearABF.ToString())
                {
                    this.ParameterizationSolver.SelectedMethod = MeshParameterization.Method.LinearABF;
                }
                else
                {
                    this.ParameterizationSolver.SelectedMethod = MeshParameterization.Method.Barycentric;
                }
            }
        }

        public string SelectedBoundaryType
        {
            get { return MeshParameterization.Instance.SelectedBoundaryType.ToString(); }
            set { MeshParameterization.Instance.SelectedBoundaryType = MeshParameterization.BoundaryTypeFromString(value); }
        }

       
        
        public string SelectedSmoothingMethod
        {
            get { return this.SmoothingSolver.SelectedMethod.ToString(); }
            set
            {
                if (value == MeshSmoothing.Method.Explicit_Euler.ToString())
                {
                    this.SmoothingSolver.SelectedMethod = MeshSmoothing.Method.Explicit_Euler;
                }
                else if (value == MeshSmoothing.Method.Implicit_Euler.ToString())
                {
                    this.SmoothingSolver.SelectedMethod = MeshSmoothing.Method.Implicit_Euler;
                }
                else if (value == MeshSmoothing.Method.TriangleQuality.ToString())
                {
                    this.SmoothingSolver.SelectedMethod = MeshSmoothing.Method.TriangleQuality;
                }
                else
                {
                    this.SmoothingSolver.SelectedMethod = MeshSmoothing.Method.LS_Optimization;
                }
                this.SmoothingSolver.Clear();
            }
        }

        public string SelectedSubdivisionMethod
        {
            get { return this.SubdivisionSolver.SelectedMethod.ToString(); }
            set
            {
                if (value == MeshSubdivision.Method.Loop.ToString())
                {
                    this.SubdivisionSolver.SelectedMethod = MeshSubdivision.Method.Loop;
                }
                else if (value == MeshSubdivision.Method.Butterfly.ToString())
                {
                    this.SubdivisionSolver.SelectedMethod = MeshSubdivision.Method.Butterfly;
                }
                else if (value == MeshSubdivision.Method.Sqrt3.ToString())
                {
                    this.SubdivisionSolver.SelectedMethod = MeshSubdivision.Method.Sqrt3;
                }
                else
                {
                    throw new NotImplementedException();
                }
            }
        }

        public string SelectedTextureImage
        {
            get { return this.selectedImage; }
            set
            {
                this.selectedImage = value;
                this.DefaultTextureImage = new BitmapImage(new Uri(@"./Data/" + this.selectedImage, UriKind.RelativeOrAbsolute));
                this.CreateDrawingImage(this.Mesh2D, this.showMesh2D, this.showImage, this.showIndices);
                this.MeshModelMaterial.DiffuseMap = this.DefaultTextureImage;
                this.UpdateModel(this.Mesh, this.flatShading ? Mesh3DUtilities.Mesh3DShading.Flat : Mesh3DUtilities.Mesh3DShading.Smooth);
            }
        }

        public TriangleMesh Mesh2D
        {
            get;
            private set;
        }

        public bool ShowMesh2D
        {
            get { return showMesh2D; }
            set { showMesh2D = value; CreateDrawingImage(this.Mesh2D, this.ShowMesh2D, this.showImage, this.ShowIndices); }
        }

        public bool ShowImage
        {
            get { return showImage; }
            set { showImage = value; CreateDrawingImage(this.Mesh2D, this.ShowMesh2D, this.showImage, this.ShowIndices); }
        }

        public bool ShowIndices
        {
            get { return this.showIndices; }
            set { this.showIndices = value; CreateDrawingImage(this.Mesh2D, this.ShowMesh2D, this.showImage, this.ShowIndices); }
        }

        public bool NormalMapping
        {
            get { return this.normalMapping; }
            set
            {
                this.normalMapping = value;
                if (this.normalMapping)
                {
                    var mat = PhongMaterials.DefaultVRML;
                    mat.NormalMap = this.NormalMapImage;
                    this.MeshModelMaterial = mat;
                }
                else
                {
                    var mat = PhongMaterials.Orange;
                    this.MeshModelMaterial = mat;
                }

            }
        }

        private bool showMesh2D, showImage, showIndices, normalMapping;
        private string selectedImage;
        private bool flatShading;

        #endregion

        #region Functions

        public MainViewModel()
        {
            this.Student = "0347138-domanski-michal-A2";

            // camera setup
            this.Camera = new PerspectiveCamera { Position = new Point3D(1, 1, 1.5), LookDirection = new Vector3D(-1, -1, -1.5), UpDirection = new Vector3D(0, 1, 0) };

            // setup lighting            
            this.AmbientLightColor = new Color4(0.3f, 0.3f, 0.3f, 1f);
            this.DirectionalLightColor = Color.WhiteSmoke;
            this.DirectionalLight1Direction = -new Vector3(+2.0f, +1.0f, +1.5f);
            this.DirectionalLight2Direction = -new Vector3(-2.0f, -1.0f, +1.5f);
            this.DirectionalLight3Direction = -new Vector3(+0.0f, +2.0f, -1.0f);

            // floor plane grid
            this.Grid = LineBuilder.GenerateGrid();
            this.GridColor = SharpDX.Color.Black;
            this.GridTransform = new Media3D.TranslateTransform3D(-5, -1, -5);

            // model trafos
            this.MeshModelTransform = new Media3D.TranslateTransform3D(0, 0, 0);

            // commands
            this.OpenCmd = new RelayCommand((x) => OnOpenClick());
            this.SmoothCmd = new RelayCommand((x) => OnSmoothingClick());
            this.ResetCmd = new RelayCommand((x) => OnResetModel());
            this.ParameterizeCmd = new RelayCommand((x) => OnParametrizeClick());
            this.SubdivisionCmd = new RelayCommand((x) => OnSubdivisionClick());
            this.LoadDetailModelCmd = new RelayCommand((x) => OnLoadDetailModelClick());
            this.SaveDisplacementCmd = new RelayCommand((x) => OnSaveDisplacementClick());
            this.LoadDisplacementCmd = new RelayCommand((x) => OnLoadDisplacementClick());
            this.GenerateDisplacementCmd = new RelayCommand((x) => OnGenerateDisplacementClick());
            this.ApplyDisplacementCmd = new RelayCommand((x) => OnApplyDisplacementClick());
            this.OpenNormalMapCmd = new RelayCommand((x) => OnOpenNormalMapClick());
            this.CreateGeometryImageCmd = new RelayCommand((x) => OnCreateGeometryImageClick());

            // start model
            //this.LoadModel(@"./Data/foot_decim.obj");
            this.LoadModel(@"./Data/nefertiti.obj");
            //this.LoadModel(@"./Data/ico_sphere_3.obj");            

            // default texture
            this.SelectedTextureImage = "grid.gif";

            // model materials
            this.MeshModelMaterial = PhongMaterials.Orange;

            // render technique for curvature 
            this.RenderTechnique = Techniques.RenderBlinn;

            this.FlatShading = false;
            this.ShowMesh2D = true;
            this.ShowImage = true;
        }

        private void OnOpenClick()
        {
            try
            {
                var d = new Microsoft.Win32.OpenFileDialog()
                {
                    Filter = "mesh files|*.obj",
                };
                if (d.ShowDialog().Value)
                {
                    if (File.Exists(d.FileName))
                    {
                        this.LoadModel(d.FileName);
                        this.SubdivisionSolver.SubdivisionLevel = 0;
                    }
                }
            }
            catch (Exception ex)
            {
                MessageBox.Show(ex.Message, "File open error!", MessageBoxButton.OK, MessageBoxImage.Error);
            }
        }

        private void OnSmoothingClick()
        {
            if (this.Mesh != null)
            {
                Mouse.OverrideCursor = Cursors.Wait;

                if (this.SmoothingSolver.Smooth(this.Mesh))
                {
                    this.Mesh.ComputeAllTraits();
                    this.UpdateModel(this.Mesh, this.flatShading ? Mesh3DUtilities.Mesh3DShading.Flat : Mesh3DUtilities.Mesh3DShading.Smooth, false);
                }
                Mouse.OverrideCursor = Cursors.Arrow;
            }
        }

        private void OnCreateGeometryImageClick()
        {
            if (this.Mesh != null)
            {
                Mouse.OverrideCursor = Cursors.Wait;
                MeshParameterization.Instance.GenerateGeometryImage(this.Mesh);
                Mouse.OverrideCursor = Cursors.Arrow;
            }
        }

        private void OnParametrizeClick()
        {
            if (this.Mesh != null)
            {
                Mouse.OverrideCursor = Cursors.Wait;

                var mesh = this.Mesh;
                MeshParameterization.Instance.PerformParameterization(mesh, out mesh);
                this.Mesh2D = mesh;
                this.Mesh2D.ComputeAllTraits();
                this.UpdateModel(this.Mesh, this.flatShading ? Mesh3DUtilities.Mesh3DShading.Flat : Mesh3DUtilities.Mesh3DShading.Smooth);
                this.MeshModelMaterial = PhongMaterials.DefaultVRML;
                this.MeshModelMaterial.DiffuseMap = DefaultTextureImage;

                this.CreateDrawingImage(this.Mesh2D, this.ShowMesh2D, this.ShowImage, this.ShowIndices);

                /// reset render rechnique
                this.RenderTechnique = null;
                this.RenderTechnique = Techniques.RenderBlinn;
                Mouse.OverrideCursor = Cursors.Arrow;
            }
        }

        private void OnSubdivisionClick()
        {            
            if (this.Mesh != null)
            {
                Mouse.OverrideCursor = Cursors.Wait;

                
                var mesh = this.Mesh;

                for (int i = 0; i < this.SubdivisionSolver.Steps; i++)
                {
                    // subdivide...
                    mesh = this.SubdivisionSolver.Subdivision(mesh);
                }

                this.Mesh = mesh;
                this.Mesh.ComputeAllTraits();
                this.UpdateModel(mesh, this.flatShading ? Mesh3DUtilities.Mesh3DShading.Flat : Mesh3DUtilities.Mesh3DShading.Smooth, false);

                this.SubdivisionSolver.SubdivisionLevel += this.SubdivisionSolver.Steps;

                Mouse.OverrideCursor = Cursors.Arrow;
            }            
        }

        private void OnResetModel()
        {
            if (!string.IsNullOrEmpty(this.Filename))
            {
                this.LoadModel(this.Filename);
                this.SubdivisionSolver.SubdivisionLevel = 0;
            }
        }

        private void OnLoadDetailModelClick()
        {
            try
            {
                var d = new Microsoft.Win32.OpenFileDialog()
                {
                    Filter = "mesh files|*.obj",
                };
                if (d.ShowDialog().Value)
                {
                    if (File.Exists(d.FileName))
                    {
                        var m = TriangleMesh.FromObjFile(d.FileName, true, true);
                        m.VerifyTopology();
                        m.ComputeAllTraits();

                        SubdivisionSolver.DetailMesh = m;
                    }
                }
            }
            catch (Exception ex)
            {
                MessageBox.Show(ex.Message, "File open error during load detail model!", MessageBoxButton.OK, MessageBoxImage.Error);
            }
        }

        private void OnLoadDisplacementClick()
        {
            try
            {
                var d = new Microsoft.Win32.OpenFileDialog()
                {
                    Filter = "binary files| *.bin",
                };
                if (d.ShowDialog().Value)
                {
                    SubdivisionSolver.DisplacementFileName = d.FileName;
                }
            }
            catch (Exception ex)
            {
                MessageBox.Show(ex.Message, "File open error during Load Displacement!", MessageBoxButton.OK, MessageBoxImage.Error);
            }
        }

        private void OnSaveDisplacementClick()
        {
            try
            {
                var d = new Microsoft.Win32.SaveFileDialog()
                {
                    Filter = "binary files| *.bin",
                };
                if (d.ShowDialog().Value)
                {
                    SubdivisionSolver.DisplacementFileName = d.FileName;
                }
            }
            catch (Exception ex)
            {
                MessageBox.Show(ex.Message, "File save error during Load Displacement!", MessageBoxButton.OK, MessageBoxImage.Error);
            }
        }

        private void OnGenerateDisplacementClick()
        {
            Mouse.OverrideCursor = Cursors.Wait;
            
            if (SubdivisionSolver.DisplacementFileName == null || SubdivisionSolver.DisplacementFileName.Length < 1)
            {
                this.OnSaveDisplacementClick();
                //MessageBox.Show("No Displacement File Name inserted!", "No Displacement File Name inserted!", MessageBoxButton.OK, MessageBoxImage.Error);                
                if (SubdivisionSolver.DisplacementFileName == null || SubdivisionSolver.DisplacementFileName.Length < 1)
                {
                    MessageBox.Show("No Displacement File Name inserted!", "No Displacement File Name inserted!", MessageBoxButton.OK, MessageBoxImage.Error);
                    return;
                }
            }
            if (SubdivisionSolver.DetailMesh == null)
            {
                MessageBox.Show("No detailed mesh loaded!", "No detailed mesh loaded!", MessageBoxButton.OK, MessageBoxImage.Error);
                return;
            }
            MeshSubdivision.GenerateDisplacementMap(this.Mesh);
            
            Mouse.OverrideCursor = Cursors.Arrow;
        }

        private void OnApplyDisplacementClick()
        {
            Mouse.OverrideCursor = Cursors.Wait;

            if (SubdivisionSolver.DisplacementFileName == null || SubdivisionSolver.DisplacementFileName.Length < 1)
            {
                MessageBox.Show("No Displacement File Name inserted!", "No Displacement File Name inserted!", MessageBoxButton.OK, MessageBoxImage.Error);
                return;
            }
            if (!File.Exists(SubdivisionSolver.DisplacementFileName))
            {
                MessageBox.Show("Displacement File not found!", "Displacement File not found!", MessageBoxButton.OK, MessageBoxImage.Error);
                return;
            }
            MeshSubdivision.ApplyDisplacement(this.Mesh);
            this.UpdateModel(this.Mesh, this.flatShading ? Mesh3DUtilities.Mesh3DShading.Flat : Mesh3DUtilities.Mesh3DShading.Smooth, false);

            Mouse.OverrideCursor = Cursors.Arrow;
        }

        private void OnOpenNormalMapClick()
        {
            try
            {
                var d = new Microsoft.Win32.OpenFileDialog()
                {
                    Filter = "image files|*.png;*.jpg;*.bmp;*.gif",
                };
                if (d.ShowDialog().Value)
                {
                    if (File.Exists(d.FileName))
                    {
                        this.NormalMapImage = new BitmapImage(new Uri(d.FileName, UriKind.RelativeOrAbsolute));
                    }
                }
            }
            catch (Exception ex)
            {
                MessageBox.Show(ex.Message, "File open error!", MessageBoxButton.OK, MessageBoxImage.Error);
            }            
        }

        private void LoadModel(string filename)
        {
            this.Filename = filename;
            this.Mesh = TriangleMesh.FromObjFile(filename, true, true);
            this.Mesh.VerifyTopology();
            this.Mesh.ComputeAllTraits();

            this.ComputeMinMaxCurvature(this.Mesh);
            this.UpdateModel(this.Mesh, this.FlatShading ? Mesh3DUtilities.Mesh3DShading.Flat : Mesh3DUtilities.Mesh3DShading.Smooth);
            this.SmoothingSolver.Clear();

            // model materials
            this.MeshModelMaterial = PhongMaterials.Orange;

            // render technique for curvature 
            this.RenderTechnique = Techniques.RenderBlinn;
        }        

        private void CreateDrawingImage(TriangleMesh mesh, bool addMesh = true, bool addImage = true, bool showNumbers = true)
        {
            if (mesh != null)
            {
                /// drawing group
                var drawingGroup = new DrawingGroup();

                /// Obtain a DrawingContext from 
                /// the DrawingGroup.
                using (DrawingContext dc = drawingGroup.Open())
                {

                    if (addImage)
                    {
                        dc.PushOpacity(0.5);
                        dc.DrawImage((ImageSource)this.DefaultTextureImage, new Rect(0, 0, 256, 256));
                        dc.Pop();
                    }

                    if (addMesh)
                    {
                        var pen = new Pen(Brushes.DarkRed, 0.2);
                        foreach (var e in mesh.Edges)
                        {
                            var p0 = 256 * e.Vertex0.Traits.Position;
                            var p1 = 256 * e.Vertex1.Traits.Position;
                            var lineGeometry = new LineGeometry(new Point(p0.X, p0.Y), new Point(p1.X, p1.Y));
                            dc.DrawLine(pen, new Point(p0.X, p0.Y), new Point(p1.X, p1.Y));
                        }
                    }
                    if (showNumbers)
                    {
                        foreach (var v in mesh.Vertices)//.Where(x => x.OnBoundary))
                        {
                            var p = new Point(256 * v.Traits.Position.X, 256 * v.Traits.Position.Y);
                            var text = new FormattedText(v.Index.ToString(), CultureInfo.CurrentCulture, FlowDirection.LeftToRight, new Typeface("Tahoma"), 2, Brushes.Red);
                            dc.DrawText(text, p);
                        }
                    }
                }

                /// freese the group
                drawingGroup.Freeze();

                /// drawing image
                this.MeshDrawingImage = new DrawingImage(drawingGroup);
            }
        }

        private static GeometryGroup CreatePathGeometry(TriangleMesh mesh)
        {
            var geometryGroup = new GeometryGroup();
            foreach (var e in mesh.Edges)
            {
                var p0 = 256 * e.Vertex0.Traits.Position;
                var p1 = 256 * e.Vertex1.Traits.Position;
                var lineGeometry = new LineGeometry(new Point(p0.X, p0.Y), new Point(p1.X, p1.Y));
                lineGeometry.Freeze();
                geometryGroup.Children.Add(lineGeometry);
            }
            foreach (var v in mesh.Vertices)//.Where(x => x.OnBoundary))
            {
                var p = new Point(256 * v.Traits.Position.X, 256 * v.Traits.Position.Y);
                //var elli = new EllipseGeometry(p, 2, 2);
                //geometryGroup.Children.Add(elli);

                /// text geometry & drawing
                var text = new FormattedText(v.Index.ToString(), CultureInfo.CurrentCulture, FlowDirection.LeftToRight, new Typeface("Tahoma"), 2, Brushes.Black);
                var textGeometry = text.BuildGeometry(p);
                textGeometry.Freeze();
                geometryGroup.Children.Add(textGeometry);
            }
            geometryGroup.Freeze();
            return geometryGroup;
        }
        
        private void ComputeMinMaxCurvature(TriangleMesh mesh)
        {
            // add curvature color coding                                                
            MinCurv = mesh.Vertices.Min(xx => (xx.Traits.MinCurvature + xx.Traits.MaxCurvature) / 2);
            MaxCurv = mesh.Vertices.Max(xx => (xx.Traits.MinCurvature + xx.Traits.MaxCurvature) / 2) - MinCurv;
        }

        private void UpdateModel(TriangleMesh mesh, Mesh3DUtilities.Mesh3DShading meshShading, bool transform = true)
        {
            if (mesh == null) return;
            if (mesh.Vertices.Count == 0) return;

            var box = mesh.Traits.BoundingBox;
            var scale = 2f / (box.Maximum - box.Minimum).Length();
            var center = (box.Maximum - box.Minimum) / 2f;

            // bbox
            this.Box = LineBuilder.GenerateBoundingBox(box);

            // edges
            this.MeshEdges = mesh.ToLineGeometry3D();

            // border edges                        
            var borderEdges = mesh.Halfedges.Where(x => x.OnBoundary).SelectMany(x => new[] { x.FromVertex.Index, x.ToVertex.Index }).ToArray();
            var borderVerts = mesh.Vertices.Select(x => x.Traits.Position).ToArray();

            if (borderEdges.Length > 0)
            {
                this.BorderEdges = new LineGeometry3D()
                {
                    Positions = borderVerts,
                    Indices = borderEdges,
                };
            }
            else
            {
                this.BorderEdges = null;
            }

            // get geometry              
            var model = mesh.ToMeshGeometry3D(meshShading, false);

            if (meshShading == Mesh3DUtilities.Mesh3DShading.Smooth)
            {
                var colors = new Color4[mesh.Vertices.Count];
                foreach (var v in mesh.Vertices)
                {
                    int idx = (int)(254 * (((v.Traits.MinCurvature + v.Traits.MaxCurvature) / 2 - MinCurv) / (MaxCurv)));
                    idx = idx < 0 ? 0 : idx;
                    idx = idx > 254 ? 254 : idx;
                    var c = JetMap255[idx];
                    //colors[v.Index] = c.ToColor4();
                    colors[v.Index] = (Color4)Color.WhiteSmoke;
                }
                model.Colors = colors;
            }
            else if (meshShading == Mesh3DUtilities.Mesh3DShading.Flat)
            {
                int ii = 0;
                var colors = new Color4[3 * mesh.Faces.Count];
                foreach (var f in mesh.Faces)
                {
                    foreach (var v in f.Vertices)
                    {
                        int idx = (int)(254 * (((v.Traits.MinCurvature + v.Traits.MaxCurvature) / 2 - MinCurv) / (MaxCurv)));
                        idx = idx < 0 ? 0 : idx;
                        idx = idx > 254 ? 254 : idx;
                        var c = JetMap255[idx];
                        //colors[ii++] = c.ToColor4();
                        colors[ii++] = (Color4)Color.WhiteSmoke;
                    }
                }
                model.Colors = colors;
            }
            else
            {
                model.Colors = mesh.Vertices.Select(x => x.Traits.Normal.ToColor4()).ToArray();
            }

            for (int i = 0; i < mesh.Vertices.Count; i++)
            {
                var v = mesh.Vertices[i];
                if (v.Halfedge == null)
                {
                    model.Colors[i] = (Color4)Color.Red;
                }
            }

            var nullVerts = mesh.Vertices.Where(x => x.Halfedge == null).ToArray().Length;
            if (nullVerts > 0)
            {
                Console.WriteLine("Mesh warning: found {0} overhang vertices!", nullVerts);
            }

            if (transform)
            {
                Media3D.Transform3D trafo = new Media3D.TranslateTransform3D(-box.Minimum.ToVector3D());
                trafo = trafo.AppendTransform(new Media3D.TranslateTransform3D(-center.ToVector3D()));
                trafo = trafo.AppendTransform(new Media3D.ScaleTransform3D(scale, scale, scale));
                this.MeshModelTransform = trafo;
            }


            //model.Colors = model.TextureCoordinates.Select(x => x.ToColor4()).ToArray();
            this.MeshModel = model;

            this.Title = System.IO.Path.GetFileName(mesh.FileName);
            this.SubTitle = string.Format("V:{0} | T:{1} | E:{2}", mesh.Vertices.Count, mesh.Faces.Count, mesh.Edges.Count);
        }

        #endregion

        #region Static Lists and Color Map

        private static readonly IList<string> renderTechniques = new List<string>
        { 
            Techniques.RenderBlinn,    
            Techniques.RenderColors,            
            Techniques.RenderPositions,
            Techniques.RenderNormals,
            Techniques.RenderTangents, 
            Techniques.RenderTexCoords,
            Techniques.RenderWires,                                 
        };

        private static readonly IList<string> textureImages = new List<string>()
        {
            "grid.gif",
            "checker.gif",
            "checker_large.gif",
            "checker_medium.gif",
            "checker_small.gif",
        };

        public static readonly Color3[] JetMap255 = new Color3[]
        {
             new Color3(     0f ,        0f ,   0.5156f),
             new Color3(     0f ,        0f ,   0.5313f),
             new Color3(     0f ,        0f ,   0.5469f),
             new Color3(     0f ,        0f ,   0.5625f),
             new Color3(     0f ,        0f ,   0.5781f),
             new Color3(     0f ,        0f ,   0.5938f),
             new Color3(     0f ,        0f ,   0.6094f),
             new Color3(     0f ,        0f ,   0.6250f),
             new Color3(     0f ,        0f ,   0.6406f),
             new Color3(     0f ,        0f ,   0.6563f),
             new Color3(     0f ,        0f ,   0.6719f),
             new Color3(     0f ,        0f ,   0.6875f),
             new Color3(     0f ,        0f ,   0.7031f),
             new Color3(     0f ,        0f ,   0.7188f),
             new Color3(     0f ,        0f ,   0.7344f),
             new Color3(     0f ,        0f ,   0.7500f),
             new Color3(     0f ,        0f ,   0.7656f),
             new Color3(     0f ,        0f ,   0.7813f),
             new Color3(     0f ,        0f ,   0.7969f),
             new Color3(     0f ,        0f ,   0.8125f),
             new Color3(     0f ,        0f ,   0.8281f),
             new Color3(     0f ,        0f ,   0.8438f),
             new Color3(     0f ,        0f ,   0.8594f),
             new Color3(     0f ,        0f ,   0.8750f),
             new Color3(     0f ,        0f ,   0.8906f),
             new Color3(     0f ,        0f ,   0.9063f),
             new Color3(     0f ,        0f ,   0.9219f),
             new Color3(     0f ,        0f ,   0.9375f),
             new Color3(     0f ,        0f ,   0.9531f),
             new Color3(     0f ,        0f ,   0.9688f),
             new Color3(     0f ,        0f ,   0.9844f),
             new Color3(     0f ,        0f ,   1.0000f),
             new Color3(     0f ,   0.0156f ,   1.0000f),
             new Color3(     0f ,   0.0313f ,   1.0000f),
             new Color3(     0f ,   0.0469f ,   1.0000f),
             new Color3(     0f ,   0.0625f ,   1.0000f),
             new Color3(     0f ,   0.0781f ,   1.0000f),
             new Color3(     0f ,   0.0938f ,   1.0000f),
             new Color3(     0f ,   0.1094f ,   1.0000f),
             new Color3(     0f ,   0.1250f ,   1.0000f),
             new Color3(     0f ,   0.1406f ,   1.0000f),
             new Color3(     0f ,   0.1563f ,   1.0000f),
             new Color3(     0f ,   0.1719f ,   1.0000f),
             new Color3(     0f ,   0.1875f ,   1.0000f),
             new Color3(     0f ,   0.2031f ,   1.0000f),
             new Color3(     0f ,   0.2188f ,   1.0000f),
             new Color3(     0f ,   0.2344f ,   1.0000f),
             new Color3(     0f ,   0.2500f ,   1.0000f),
             new Color3(     0f ,   0.2656f ,   1.0000f),
             new Color3(     0f ,   0.2813f ,   1.0000f),
             new Color3(     0f ,   0.2969f ,   1.0000f),
             new Color3(     0f ,   0.3125f ,   1.0000f),
             new Color3(     0f ,   0.3281f ,   1.0000f),
             new Color3(     0f ,   0.3438f ,   1.0000f),
             new Color3(     0f ,   0.3594f ,   1.0000f),
             new Color3(     0f ,   0.3750f ,   1.0000f),
             new Color3(     0f ,   0.3906f ,   1.0000f),
             new Color3(     0f ,   0.4063f ,   1.0000f),
             new Color3(     0f ,   0.4219f ,   1.0000f),
             new Color3(     0f ,   0.4375f ,   1.0000f),
             new Color3(     0f ,   0.4531f ,   1.0000f),
             new Color3(     0f ,   0.4688f ,   1.0000f),
             new Color3(     0f ,   0.4844f ,   1.0000f),
             new Color3(     0f ,   0.5000f ,   1.0000f),
             new Color3(     0f ,   0.5156f ,   1.0000f),
             new Color3(     0f ,   0.5313f ,   1.0000f),
             new Color3(     0f ,   0.5469f ,   1.0000f),
             new Color3(     0f ,   0.5625f ,   1.0000f),
             new Color3(     0f ,   0.5781f ,   1.0000f),
             new Color3(     0f ,   0.5938f ,   1.0000f),
             new Color3(     0f ,   0.6094f ,   1.0000f),
             new Color3(     0f ,   0.6250f ,   1.0000f),
             new Color3(     0f ,   0.6406f ,   1.0000f),
             new Color3(     0f ,   0.6563f ,   1.0000f),
             new Color3(     0f ,   0.6719f ,   1.0000f),
             new Color3(     0f ,   0.6875f ,   1.0000f),
             new Color3(     0f ,   0.7031f ,   1.0000f),
             new Color3(     0f ,   0.7188f ,   1.0000f),
             new Color3(     0f ,   0.7344f ,   1.0000f),
             new Color3(     0f ,   0.7500f ,   1.0000f),
             new Color3(     0f ,   0.7656f ,   1.0000f),
             new Color3(     0f ,   0.7813f ,   1.0000f),
             new Color3(     0f ,   0.7969f ,   1.0000f),
             new Color3(     0f ,   0.8125f ,   1.0000f),
             new Color3(     0f ,   0.8281f ,   1.0000f),
             new Color3(     0f ,   0.8438f ,   1.0000f),
             new Color3(     0f ,   0.8594f ,   1.0000f),
             new Color3(     0f ,   0.8750f ,   1.0000f),
             new Color3(     0f ,   0.8906f ,   1.0000f),
             new Color3(     0f ,   0.9063f ,   1.0000f),
             new Color3(     0f ,   0.9219f ,   1.0000f),
             new Color3(     0f ,   0.9375f ,   1.0000f),
             new Color3(     0f ,   0.9531f ,   1.0000f),
             new Color3(     0f ,   0.9688f ,   1.0000f),
             new Color3(     0f ,   0.9844f ,   1.0000f),
             new Color3(     0f ,   1.0000f ,   1.0000f),
             new Color3(0.0156f ,   1.0000f ,   0.9844f),
             new Color3(0.0313f ,   1.0000f ,   0.9688f),
             new Color3(0.0469f ,   1.0000f ,   0.9531f),
             new Color3(0.0625f ,   1.0000f ,   0.9375f),
             new Color3(0.0781f ,   1.0000f ,   0.9219f),
             new Color3(0.0938f ,   1.0000f ,   0.9063f),
             new Color3(0.1094f ,   1.0000f ,   0.8906f),
             new Color3(0.1250f ,   1.0000f ,   0.8750f),
             new Color3(0.1406f ,   1.0000f ,   0.8594f),
             new Color3(0.1563f ,   1.0000f ,   0.8438f),
             new Color3(0.1719f ,   1.0000f ,   0.8281f),
             new Color3(0.1875f ,   1.0000f ,   0.8125f),
             new Color3(0.2031f ,   1.0000f ,   0.7969f),
             new Color3(0.2188f ,   1.0000f ,   0.7813f),
             new Color3(0.2344f ,   1.0000f ,   0.7656f),
             new Color3(0.2500f ,   1.0000f ,   0.7500f),
             new Color3(0.2656f ,   1.0000f ,   0.7344f),
             new Color3(0.2813f ,   1.0000f ,   0.7188f),
             new Color3(0.2969f ,   1.0000f ,   0.7031f),
             new Color3(0.3125f ,   1.0000f ,   0.6875f),
             new Color3(0.3281f ,   1.0000f ,   0.6719f),
             new Color3(0.3438f ,   1.0000f ,   0.6563f),
             new Color3(0.3594f ,   1.0000f ,   0.6406f),
             new Color3(0.3750f ,   1.0000f ,   0.6250f),
             new Color3(0.3906f ,   1.0000f ,   0.6094f),
             new Color3(0.4063f ,   1.0000f ,   0.5938f),
             new Color3(0.4219f ,   1.0000f ,   0.5781f),
             new Color3(0.4375f ,   1.0000f ,   0.5625f),
             new Color3(0.4531f ,   1.0000f ,   0.5469f),
             new Color3(0.4688f ,   1.0000f ,   0.5313f),
             new Color3(0.4844f ,   1.0000f ,   0.5156f),
             new Color3(0.5000f ,   1.0000f ,   0.5000f),
             new Color3(0.5156f ,   1.0000f ,   0.4844f),
             new Color3(0.5313f ,   1.0000f ,   0.4688f),
             new Color3(0.5469f ,   1.0000f ,   0.4531f),
             new Color3(0.5625f ,   1.0000f ,   0.4375f),
             new Color3(0.5781f ,   1.0000f ,   0.4219f),
             new Color3(0.5938f ,   1.0000f ,   0.4063f),
             new Color3(0.6094f ,   1.0000f ,   0.3906f),
             new Color3(0.6250f ,   1.0000f ,   0.3750f),
             new Color3(0.6406f ,   1.0000f ,   0.3594f),
             new Color3(0.6563f ,   1.0000f ,   0.3438f),
             new Color3(0.6719f ,   1.0000f ,   0.3281f),
             new Color3(0.6875f ,   1.0000f ,   0.3125f),
             new Color3(0.7031f ,   1.0000f ,   0.2969f),
             new Color3(0.7188f ,   1.0000f ,   0.2813f),
             new Color3(0.7344f ,   1.0000f ,   0.2656f),
             new Color3(0.7500f ,   1.0000f ,   0.2500f),
             new Color3(0.7656f ,   1.0000f ,   0.2344f),
             new Color3(0.7813f ,   1.0000f ,   0.2188f),
             new Color3(0.7969f ,   1.0000f ,   0.2031f),
             new Color3(0.8125f ,   1.0000f ,   0.1875f),
             new Color3(0.8281f ,   1.0000f ,   0.1719f),
             new Color3(0.8438f ,   1.0000f ,   0.1563f),
             new Color3(0.8594f ,   1.0000f ,   0.1406f),
             new Color3(0.8750f ,   1.0000f ,   0.1250f),
             new Color3(0.8906f ,   1.0000f ,   0.1094f),
             new Color3(0.9063f ,   1.0000f ,   0.0938f),
             new Color3(0.9219f ,   1.0000f ,   0.0781f),
             new Color3(0.9375f ,   1.0000f ,   0.0625f),
             new Color3(0.9531f ,   1.0000f ,   0.0469f),
             new Color3(0.9688f ,   1.0000f ,   0.0313f),
             new Color3(0.9844f ,   1.0000f ,   0.0156f),
             new Color3(1.0000f ,   1.0000f ,        0f),
             new Color3(1.0000f ,   0.9844f ,        0f),
             new Color3(1.0000f ,   0.9688f ,        0f),
             new Color3(1.0000f ,   0.9531f ,        0f),
             new Color3(1.0000f ,   0.9375f ,        0f),
             new Color3(1.0000f ,   0.9219f ,        0f),
             new Color3(1.0000f ,   0.9063f ,        0f),
             new Color3(1.0000f ,   0.8906f ,        0f),
             new Color3(1.0000f ,   0.8750f ,        0f),
             new Color3(1.0000f ,   0.8594f ,        0f),
             new Color3(1.0000f ,   0.8438f ,        0f),
             new Color3(1.0000f ,   0.8281f ,        0f),
             new Color3(1.0000f ,   0.8125f ,        0f),
             new Color3(1.0000f ,   0.7969f ,        0f),
             new Color3(1.0000f ,   0.7813f ,        0f),
             new Color3(1.0000f ,   0.7656f ,        0f),
             new Color3(1.0000f ,   0.7500f ,        0f),
             new Color3(1.0000f ,   0.7344f ,        0f),
             new Color3(1.0000f ,   0.7188f ,        0f),
             new Color3(1.0000f ,   0.7031f ,        0f),
             new Color3(1.0000f ,   0.6875f ,        0f),
             new Color3(1.0000f ,   0.6719f ,        0f),
             new Color3(1.0000f ,   0.6563f ,        0f),
             new Color3(1.0000f ,   0.6406f ,        0f),
             new Color3(1.0000f ,   0.6250f ,        0f),
             new Color3(1.0000f ,   0.6094f ,        0f),
             new Color3(1.0000f ,   0.5938f ,        0f),
             new Color3(1.0000f ,   0.5781f ,        0f),
             new Color3(1.0000f ,   0.5625f ,        0f),
             new Color3(1.0000f ,   0.5469f ,        0f),
             new Color3(1.0000f ,   0.5313f ,        0f),
             new Color3(1.0000f ,   0.5156f ,        0f),
             new Color3(1.0000f ,   0.5000f ,        0f),
             new Color3(1.0000f ,   0.4844f ,        0f),
             new Color3(1.0000f ,   0.4688f ,        0f),
             new Color3(1.0000f ,   0.4531f ,        0f),
             new Color3(1.0000f ,   0.4375f ,        0f),
             new Color3(1.0000f ,   0.4219f ,        0f),
             new Color3(1.0000f ,   0.4063f ,        0f),
             new Color3(1.0000f ,   0.3906f ,        0f),
             new Color3(1.0000f ,   0.3750f ,        0f),
             new Color3(1.0000f ,   0.3594f ,        0f),
             new Color3(1.0000f ,   0.3438f ,        0f),
             new Color3(1.0000f ,   0.3281f ,        0f),
             new Color3(1.0000f ,   0.3125f ,        0f),
             new Color3(1.0000f ,   0.2969f ,        0f),
             new Color3(1.0000f ,   0.2813f ,        0f),
             new Color3(1.0000f ,   0.2656f ,        0f),
             new Color3(1.0000f ,   0.2500f ,        0f),
             new Color3(1.0000f ,   0.2344f ,        0f),
             new Color3(1.0000f ,   0.2188f ,        0f),
             new Color3(1.0000f ,   0.2031f ,        0f),
             new Color3(1.0000f ,   0.1875f ,        0f),
             new Color3(1.0000f ,   0.1719f ,        0f),
             new Color3(1.0000f ,   0.1563f ,        0f),
             new Color3(1.0000f ,   0.1406f ,        0f),
             new Color3(1.0000f ,   0.1250f ,        0f),
             new Color3(1.0000f ,   0.1094f ,        0f),
             new Color3(1.0000f ,   0.0938f ,        0f),
             new Color3(1.0000f ,   0.0781f ,        0f),
             new Color3(1.0000f ,   0.0625f ,        0f),
             new Color3(1.0000f ,   0.0469f ,        0f),
             new Color3(1.0000f ,   0.0313f ,        0f),
             new Color3(1.0000f ,   0.0156f ,        0f),
             new Color3(1.0000f ,        0f ,        0f),
             new Color3(0.9844f ,        0f ,        0f),
             new Color3(0.9688f ,        0f ,        0f),
             new Color3(0.9531f ,        0f ,        0f),
             new Color3(0.9375f ,        0f ,        0f),
             new Color3(0.9219f ,        0f ,        0f),
             new Color3(0.9063f ,        0f ,        0f),
             new Color3(0.8906f ,        0f ,        0f),
             new Color3(0.8750f ,        0f ,        0f),
             new Color3(0.8594f ,        0f ,        0f),
             new Color3(0.8438f ,        0f ,        0f),
             new Color3(0.8281f ,        0f ,        0f),
             new Color3(0.8125f ,        0f ,        0f),
             new Color3(0.7969f ,        0f ,        0f),
             new Color3(0.7813f ,        0f ,        0f),
             new Color3(0.7656f ,        0f ,        0f),
             new Color3(0.7500f ,        0f ,        0f),
             new Color3(0.7344f ,        0f ,        0f),
             new Color3(0.7188f ,        0f ,        0f),
             new Color3(0.7031f ,        0f ,        0f),
             new Color3(0.6875f ,        0f ,        0f),
             new Color3(0.6719f ,        0f ,        0f),
             new Color3(0.6563f ,        0f ,        0f),
             new Color3(0.6406f ,        0f ,        0f),
             new Color3(0.6250f ,        0f ,        0f),
             new Color3(0.6094f ,        0f ,        0f),
             new Color3(0.5938f ,        0f ,        0f),
             new Color3(0.5781f ,        0f ,        0f),
             new Color3(0.5625f ,        0f ,        0f),
             new Color3(0.5469f ,        0f ,        0f),
             new Color3(0.5313f ,        0f ,        0f),
             new Color3(0.5156f ,        0f ,        0f),
        };
        #endregion
    }
}
