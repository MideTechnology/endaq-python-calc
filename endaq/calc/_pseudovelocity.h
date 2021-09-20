/* This module contains an accelerated method for calculating a bank of peak
   pseudovelocity impulses
*/

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <stdio.h>

#include <numpy/arrayobject.h>
#include <numpy/ndarraytypes.h>
