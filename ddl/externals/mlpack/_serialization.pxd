"""
serialization.pxd: serialization functions for mlpack classes.

This simply makes the utility serialization functions from serialization.hpp
available from Python.
"""
cimport cython

from libcpp.string cimport string

cdef extern from "_serialization.hpp" namespace "mlpack::bindings::python" nogil:
  string SerializeOut[T](T* t, string name) nogil
  void SerializeIn[T](T* t, string str, string name) nogil
