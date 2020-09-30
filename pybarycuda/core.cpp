#include "Python.h"
#include "barylib.h"

static PyObject* barycuda_core_point_in_simplex(PyObject* self,
                                                PyObject* args) {
  void* py_pts;
  int n, dim;
  void* py_verts;
  int tn, vn;

  // Parse Python arguments. tn and vn are used only as bounds for y# 
  if (!PyArg_ParseTuple(args, "y#iiy#", &py_pts, &tn, 
                        &n, &dim, &py_verts, &vn))
    return NULL;

  vec3f* pts = static_cast<vec3f*>(py_pts);
  vec3f* verts = static_cast<vec3f*>(py_verts);

  bool* res = bary::point_in_simplex(pts, n, dim, verts);

  // Create a list of ints which we later cast to boolean 
  PyObject* out = PyList_New(n);
  for (int i = 0; i < n; ++i) {
    PyList_SET_ITEM(out, i, Py_BuildValue("i", res[i]));
  }
  return out;
}

static PyObject* barycuda_core_bary_simplex(PyObject* self, PyObject* args) {
  void* py_pts;
  void* py_verts;
  int n, dim;
  int tn, vn;

  // Parse Python arguments. tn and vn are used only as bounds for y# 
  if (!PyArg_ParseTuple(args, "y#iiy#", &py_pts, &tn, 
                        &n, &dim, &py_verts, &vn))
    return NULL;

  vec3f* pts = static_cast<vec3f*>(py_pts);
  vec3f* verts = static_cast<vec3f*>(py_verts);

  float** res = bary::bary_simplex(pts, n, dim, verts);

  // Construct a 2D Python list
  PyObject* out = PyList_New(n);
  for (int i = 0; i < n; ++i) {
    PyObject* sub = PyList_New(dim);
    for (int j = 0; j < dim; ++j) {
      PyList_SET_ITEM(sub, j, Py_BuildValue("f", res[i][j]));
    }
    PyList_SET_ITEM(out, i, sub);
  }
  return out;
}

static PyMethodDef barycuda_core_methods[] = {
    {"point_in_simplex", (PyCFunction)barycuda_core_point_in_simplex,
     METH_VARARGS, "Check if points are in a simplex"},
    {"bary_simplex", (PyCFunction)barycuda_core_bary_simplex, METH_VARARGS,
     "Return barycentric coordinates of points relative to a simplex"},
    {NULL, NULL, 0, NULL}};

static struct PyModuleDef module_def = {
    PyModuleDef_HEAD_INIT, "pybarycuda.core",
    "A tiny CUDA library for fast barycentric operations.", -1,
    barycuda_core_methods};

PyMODINIT_FUNC PyInit_core() { return PyModule_Create(&module_def); }
