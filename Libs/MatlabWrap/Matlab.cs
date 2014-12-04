#define MATLAB_

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.IO;

namespace MatlabWrap
{
    public static class Matlab
    {
        public static string MatlabPath { get; private set; }
        public static string BinDir = "";

#if MATLAB
        private static MLApp.MLApp MATLAB = null;
#endif


        public static void InitMATLAB(bool clear)
        {
#if MATLAB
            //#if DEPLOY
            //            var path = Directory.GetCurrentDirectory();
            //            MatlabPath = path + "\\..\\..\\Matlab";
            //            Console.WriteLine("Matlab path: " + MatlabPath);
            //#else
            //            var path = Directory.GetCurrentDirectory();
            //            MatlabPath = path + "\\..\\..\\..\\..\\Matlab";
            //            Console.WriteLine("Matlab path: " + MatlabPath);
            //#endif
            MATLAB = new MLApp.MLApp();
            if (clear)
                MATLAB.Execute("clear;");
            // MATLAB.Execute("cd " + MatlabPath);
            //  MATLAB.Execute("cd Clustering");
            //MATLAB.Execute("EnableAutomation");   
#endif
        }

        public static void SetPath(string path)
        {
#if MATLAB
            Matlab.MATLAB.Execute(@"cd " + path);
#endif
        }

        public static string Execute(string s)
        {
#if MATLAB
            return Matlab.MATLAB.Execute(s);
#else
            return null;
#endif
        }

        public static void PutWorkspaceData(string name, string workspace, object o)
        {
#if MATLAB
            Matlab.MATLAB.PutWorkspaceData(name, workspace, o);
#endif
        }

        public static void GetWorkspaceData(string name, string workspace, out object o)
        {
            o = null;
#if MATLAB
            Matlab.MATLAB.GetWorkspaceData(name, workspace, out o);
#endif
        }

        public static double GetDoubleFromWorkspace(string doubleName)
        {

            object O = null;
#if MATLAB
            MATLAB.GetWorkspaceData(doubleName, "base", out O);
#endif
            return (double)O;

        }

        public static double[] GetDouble1FromWorkspace(string arrayName)
        {

            object O = null;
#if MATLAB
            MATLAB.GetWorkspaceData(arrayName, "base", out O);
#endif

            var array0 = O as double[,];
            if (array0 == null)
            {
                var value0 = (double)O;
                return new double[] { value0 };
            }

            var m = array0.GetLength(0);
            var array1 = new double[m];
            for (int i = 0; i < m; i++)
                array1[i] = array0[i, 0];

            return array1;
        }

        public static double[,] GetDouble2FromWorkspace(string arrayName)
        {

            object O = null;
#if MATLAB
            MATLAB.GetWorkspaceData(arrayName, "base", out O);
#endif

            var array0 = (double[,])O;
            return array0;
        }

        public static double[, ,] GetDouble3FromWorkspace(string arrayName)
        {
            object O = null;
#if MATLAB
            MATLAB.GetWorkspaceData(arrayName, "base", out O);
#endif
            var array0 = (double[, ,])O;
            return array0;
        }


        public static float GetFloatFromWorkspace(string doubleName)
        {
            object O = null;
#if MATLAB
            MATLAB.GetWorkspaceData(doubleName, "base", out O);
#endif
            return (float)O;
        }

        public static float[] GetFloat1FromWorkspace(string arrayName)
        {

            object O = null;

#if MATLAB
            MATLAB.GetWorkspaceData(arrayName, "base", out O);
#endif

            var array0 = (float[,])O;
            var m = array0.GetLength(0);
            var array1 = new float[m];
            for (int i = 0; i < m; i++)
                array1[i] = array0[i, 0];

            return array1;
        }

        public static float[,] GetFloat2FromWorkspace(string arrayName)
        {
            object O = null;

#if MATLAB
            MATLAB.GetWorkspaceData(arrayName, "base", out O);
#endif

            var array0 = (float[,])O;
            return array0;
        }

        public static float[, ,] GetFloat3FromWorkspace(string arrayName)
        {
            object O = null;

#if MATLAB
            MATLAB.GetWorkspaceData(arrayName, "base", out O);
#endif

            var array0 = (float[, ,])O;
            return array0;
        }


        public static byte[,] GetByte2FromWorkspace(string arrayName)
        {
            object O = null;

#if MATLAB
            MATLAB.GetWorkspaceData(arrayName, "base", out O);
#endif

            var array0 = (byte[,])O;
            return array0;
        }

        public static byte[, ,] GetByte3FromWorkspace(string arrayName)
        {
            object O = null;

#if MATLAB
            MATLAB.GetWorkspaceData(arrayName, "base", out O);
#endif

            var array0 = (byte[, ,])O;
            return array0;
        }
    }
}
