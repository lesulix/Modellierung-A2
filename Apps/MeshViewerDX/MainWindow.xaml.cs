using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Controls.Primitives;
using System.Windows.Data;
using System.Windows.Documents;
using System.Windows.Input;
using System.Windows.Markup;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Navigation;
using System.Windows.Shapes;

namespace MeshViewerDX
{
    /// <summary>
    /// Interaction logic for MainWindow.xaml
    /// </summary>
    public partial class MainWindow : Window
    {
        public MainWindow()
        {
            InitializeComponent();            
            this.DataContext = new MainViewModel();


            var canvasTrafo = new MatrixTransform();
            var mouseDownPos = new Point();
            
            this.canvas.RenderTransform = canvasTrafo;
            this.canvas.MouseMove += (o, e) =>
            {
                /// move image on the canvas
                if (e.RightButton == MouseButtonState.Pressed)
                {
                    var p = e.GetPosition(this.canvas) - mouseDownPos;
                    var m = canvasTrafo.Matrix;
                    m.TranslatePrepend(p.X, p.Y);
                    canvasTrafo.Matrix = m;
                }
            };

            this.canvas.MouseWheel += (o, e) =>
            {
                var p = e.GetPosition(this.canvas);
                var delta = 1.0 + (0.001 * e.Delta);
                var m = canvasTrafo.Matrix;
                m.ScaleAtPrepend(delta, delta, p.X, p.Y);
                canvasTrafo.Matrix = m;
            };

            this.canvas.MouseDown += (o, e) =>
            {
                mouseDownPos = e.GetPosition(this.canvas);
            };
        }


    }


    public class BoolToVisibilityConverter : IValueConverter
    {
        public BoolToVisibilityConverter()
        {
        }

        public object Convert(object value, Type targetType, object parameter, System.Globalization.CultureInfo culture)
        {
            bool val = System.Convert.ToBoolean(value);
            return val ? Visibility.Visible : Visibility.Collapsed;
        }

        public object ConvertBack(object value, Type targetType, object parameter, System.Globalization.CultureInfo culture)
        {
            return ((Visibility)(value)) == Visibility.Visible ? true : false;
        }
    }

    public class MoveThumb : Thumb
    {
        public MoveThumb()
        {
            DragDelta += new DragDeltaEventHandler(this.MoveThumb_DragDelta);
        }

        private void MoveThumb_DragDelta(object sender, DragDeltaEventArgs e)
        {
            var designerItem = this.DataContext as Control;
            var presenter = VisualTreeHelper.GetParent(designerItem) as ContentPresenter;
            var selector = VisualTreeHelper.GetParent(presenter) as FrameworkElement;
            var parent = VisualTreeHelper.GetParent(selector) as FrameworkElement;

            if (presenter != null)
            {
                double left = Canvas.GetLeft(presenter);
                double top = Canvas.GetTop(presenter);

                Canvas.SetLeft(presenter, left + e.HorizontalChange);
                Canvas.SetTop(presenter, top + e.VerticalChange);
            }
        }
    }
}
