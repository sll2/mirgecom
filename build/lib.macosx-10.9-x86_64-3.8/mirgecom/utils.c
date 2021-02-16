#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <stdint.h>

static PyObject* as_buffer(PyObject *self, PyObject *args)
{
    unsigned long mem_ptr = 0;
    int size = 0;
    int offset = 0;

    if (!PyArg_ParseTuple(args, "kii", &mem_ptr, &size, &offset))
    {
        return NULL;
    }

    return PyMemoryView_FromMemory((char *) (mem_ptr + offset), size, PyBUF_WRITE);
}

static PyMethodDef CaclMethods[] = {

    {"as_buffer", as_buffer, METH_VARARGS,
     "Return device pointer as Python object."},
    {NULL, NULL, 0, NULL}

};

static struct PyModuleDef caclmodule = {
    PyModuleDef_HEAD_INIT,
    "cacl",
    "Python interface to put OpenCL device pointer into Python buffer.",
    -1,
    CaclMethods
};

PyMODINIT_FUNC PyInit_cacl(void)
{
    return PyModule_Create(&caclmodule);
}
