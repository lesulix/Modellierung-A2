#region License
// Copyright (c) 2006 Alexander Kolliopoulos
//
// This software is provided 'as-is', without any express or implied
// warranty. In no event will the authors be held liable for any damages
// arising from the use of this software.
//
// Permission is granted to anyone to use this software for any purpose,
// including commercial applications, and to alter it and redistribute it
// freely, subject to the following restrictions:
//
// 1. The origin of this software must not be misrepresented; you must not
//    claim that you wrote the original software. If you use this software
//    in a product, an acknowledgment in the product documentation would
//    be appreciated but is not required.
//
// 2. Altered source versions must be plainly marked as such, and must not
//    be misrepresented as being the original software.
//
// 3. This notice may not be removed or altered from any source distribution.
#endregion

using System;
using System.Runtime.Serialization;

namespace Meshes.Generic
{
    #region MeshException
    /// <summary>
    /// The base class for mesh-related exceptions.
    /// </summary>
    [Serializable]
    public class MeshException : Exception
    {
        /// <summary>
        /// Initializes a new instance of this class.
        /// </summary>
        public MeshException() : base() { }

        /// <summary>
        /// Initializes a new instance of this class with a specified error message.
        /// </summary>
        /// <param name="message">A string that describes the error.</param>
        public MeshException(string message) : base(message) { }

        /// <summary>
        /// Initializes a new instance of this class with a specified error message and a reference to the inner exception that is the cause of this exception.
        /// </summary>
        /// <param name="message">A string that describes the error.</param>
        /// <param name="inner">The exception that is the cause of the current exception.</param>
		public MeshException(string message, Exception inner) : base(message, inner) { }

        /// <summary>
        /// Initializes a new instance of this class with serialized data.
        /// </summary>
        /// <param name="info">The object that holds the serialized object data.</param>
        /// <param name="context">The contextual information about the source or destination.</param>
        protected MeshException(SerializationInfo info, StreamingContext context) : base(info, context) {}
    }
    #endregion

    #region BadTopologyException
    /// <summary>
    /// The exception that is thrown when non-manifold topology is detected.
    /// </summary>
    [Serializable]
    public class BadTopologyException : MeshException
    {
        /// <summary>
        /// Initializes a new instance of this class.
        /// </summary>
        public BadTopologyException() : base() { }

        /// <summary>
        /// Initializes a new instance of this class with a specified error message.
        /// </summary>
        /// <param name="message">A string that describes the error.</param>
        public BadTopologyException(string message) : base(message) { }

        /// <summary>
        /// Initializes a new instance of this class with a specified error message and a reference to the inner exception that is the cause of this exception.
        /// </summary>
        /// <param name="message">A string that describes the error.</param>
        /// <param name="inner">The exception that is the cause of the current exception.</param>
		public BadTopologyException(string message, Exception inner) : base(message, inner) { }

        /// <summary>
        /// Initializes a new instance of this class with serialized data.
        /// </summary>
        /// <param name="info">The object that holds the serialized object data.</param>
        /// <param name="context">The contextual information about the source or destination.</param>
        protected BadTopologyException(SerializationInfo info, StreamingContext context) : base(info, context) {}
    }
    #endregion

    #region MismatchedMeshException
    /// <summary>
    /// The exception that is thrown when an attempt is made to use a mesh element with an element or method for a different mesh.
    /// </summary>
    [Serializable]
    public class MismatchedMeshException : MeshException
    {
        /// <summary>
        /// Initializes a new instance of this class.
        /// </summary>
        public MismatchedMeshException() : base() { }

        /// <summary>
        /// Initializes a new instance of this class with a specified error message.
        /// </summary>
        /// <param name="message">A string that describes the error.</param>
        public MismatchedMeshException(string message) : base(message) { }

        /// <summary>
        /// Initializes a new instance of this class with a specified error message and a reference to the inner exception that is the cause of this exception.
        /// </summary>
        /// <param name="message">A string that describes the error.</param>
        /// <param name="inner">The exception that is the cause of the current exception.</param>
        public MismatchedMeshException(string message, Exception inner) : base(message, inner) { }

        /// <summary>
        /// Initializes a new instance of this class with serialized data.
        /// </summary>
        /// <param name="info">The object that holds the serialized object data.</param>
        /// <param name="context">The contextual information about the source or destination.</param>
        protected MismatchedMeshException(SerializationInfo info, StreamingContext context) : base(info, context) {}
    }
    #endregion
}
