using SharpDX;
using System;
using Box3 = SharpDX.BoundingBox;

namespace Meshes
{
    public static class Extensions
    {
        /// <summary>
        /// Extends the box to contain the supplied box.
        /// </summary>
        public static Box3 ExtendBy(this Box3 thisBox, Box3 box)
        {
            if (box.Minimum.X < thisBox.Minimum.X) thisBox.Minimum.X = box.Minimum.X;
            if (box.Maximum.X > thisBox.Maximum.X) thisBox.Maximum.X = box.Maximum.X;
            if (box.Minimum.Y < thisBox.Minimum.Y) thisBox.Minimum.Y = box.Minimum.Y;
            if (box.Maximum.Y > thisBox.Maximum.Y) thisBox.Maximum.Y = box.Maximum.Y;
            if (box.Minimum.Z < thisBox.Minimum.Z) thisBox.Minimum.Z = box.Minimum.Z;
            if (box.Maximum.Z > thisBox.Maximum.Z) thisBox.Maximum.Z = box.Maximum.Z;
            return thisBox;
        }

        /// <summary>
        /// Extends the box to contain the supplied value.
        /// </summary>
        public static Box3 ExtendBy(this Box3 thisBox, Vector3 point)
        {
            if (point.X < thisBox.Minimum.X) thisBox.Minimum.X = point.X;
            if (point.X > thisBox.Maximum.X) thisBox.Maximum.X = point.X;
            if (point.Y < thisBox.Minimum.Y) thisBox.Minimum.Y = point.Y;
            if (point.Y > thisBox.Maximum.Y) thisBox.Maximum.Y = point.Y;
            if (point.Z < thisBox.Minimum.Z) thisBox.Minimum.Z = point.Z;
            if (point.Z > thisBox.Maximum.Z) thisBox.Maximum.Z = point.Z;
            return thisBox;
        }
        /// <summary>
        /// Extends the box to contain the supplied value.
        /// </summary>
        public static Box3 ExtendXBy(this Box3 thisBox, float x)
        {
            if (x < thisBox.Minimum.X) thisBox.Minimum.X = x;
            if (x > thisBox.Maximum.X) thisBox.Maximum.X = x;
            return thisBox;
        }
        /// <summary>
        /// Extends the box to contain the supplied value.
        /// </summary>
        public static Box3 ExtendYBy(this Box3 thisBox, float y)
        {
            if (y < thisBox.Minimum.Y) thisBox.Minimum.Y = y;
            if (y > thisBox.Maximum.Y) thisBox.Maximum.Y = y;
            return thisBox;
        }
        /// <summary>
        /// Extends the box to contain the supplied value.
        /// </summary>
        public static Box3 ExtendZBy(this Box3 thisBox, float z)
        {
            if (z < thisBox.Minimum.Z) thisBox.Minimum.Z = z;
            if (z > thisBox.Maximum.Z) thisBox.Maximum.Z = z;
            return thisBox;
        }
    }
}
