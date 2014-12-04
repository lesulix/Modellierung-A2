using System;
using System.Collections.Generic;
using System.Linq;

namespace Meshes.Algorithms
{
    public static class EnumerableExtensions
    {
        public static void Apply<T>(this IEnumerable<T> collection, Action<T> func)
        {
            foreach (var currentElement in collection)
            {
                func(currentElement);
            }
        }

        public static void Combine<T>(this T[] lhs, T[] rhs, Func<T, T, T> combinator)
        {
            foreach (var index in Enumerable.Range(0, lhs.Count()))
                lhs[index] = combinator(lhs[index], rhs[index]);
        }

        public static void Apply<T>(this IEnumerable<T> collection, Action<T, int> func)
        {
            var currentIndex = 0;
            foreach (var currentElement in collection)
            {
                func(currentElement, currentIndex++);
            }
        }
    }
}