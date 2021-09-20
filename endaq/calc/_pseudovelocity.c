/* This module contains an accelerated method for calculating a bank of peak
   pseudovelocity impulses
*/

#include <omp.h>
#include <intrin.h>

#include "_pseudovelocity.h"


// https://ccrma.stanford.edu/~jos/filters/Transposed_Direct_Forms.html
// direct form II transposed
// scipy SOS should be [b0, b1, b2, 1, a1, a2]
// note: b2 should be 0 so we'll just assume
void psBank(double filters[][6], double x[], double z[][2], int filterWidth, int zLen) {

    #pragma omp parallel shared(filters, x, z)
    {
        register int i, j, k;
        register __m256d s00, s11, b00, b11, a11, a22, xxn, zzMin, zzMax, yyn;
        double b0[4], b1[4], a1[4], a2[4];

        #pragma omp for
        for (j = 0; j < filterWidth; j += 4) {

            s00 = _mm256_setzero_pd();
            s11 = _mm256_setzero_pd();
            b00 = _mm256_setzero_pd();
            b11 = _mm256_setzero_pd();
            a11 = _mm256_setzero_pd();
            a22 = _mm256_setzero_pd();
            b00 = _mm256_setzero_pd();
            b11 = _mm256_setzero_pd();
            a11 = _mm256_setzero_pd();
            a22 = _mm256_setzero_pd();
            xxn = _mm256_setzero_pd();
            zzMin = _mm256_setzero_pd();
            zzMax = _mm256_setzero_pd();
            yyn = _mm256_setzero_pd();

            for (k = 0; k < 4; k++) {
                if ((k + j) == filterWidth) {
                    b0[k] = 0;
                    b1[k] = 0;
                    a1[k] = 0;
                    a2[k] = 0;
                } else {
                    b0[k] = filters[j + k][0];
                    b1[k] = filters[j + k][1];
                    a1[k] = filters[j + k][4];
                    a2[k] = filters[j + k][5];
                }
            }

            b00 = _mm256_setr_pd(b0[0], b0[1], b0[2], b0[3]);
            b11 = _mm256_setr_pd(b1[0], b1[1], b1[2], b1[3]);
            a11 = _mm256_setr_pd(a1[0], a1[1], a1[2], a1[3]);
            a22 = _mm256_setr_pd(a2[0], a2[1], a2[2], a2[3]);

            for (i = 0; i < zLen; i++) {
            // for every frequency bin
                xxn = _mm256_set1_pd(x[i]);

                // get y[n]
                yyn = _mm256_fmadd_pd(xxn, b00, s00);

                // update s
                s00 = _mm256_fmsub_pd(xxn, b11, _mm256_fmadd_pd(yyn, a11, s11));
                s11 = _mm256_mul_pd(yyn, a22);

                // compare y[n] to mins and maxes
                zzMax = _mm256_max_pd(zzMax, yyn);
                zzMin = _mm256_min_pd(zzMin, yyn);

            }

            // first element
            z[j + 0][0] = zzMax.m256d_f64[0];
            z[j + 0][1] = zzMin.m256d_f64[0];

            // second element
            if ( (j + 1) < filterWidth) {
                z[j + 1][0] = zzMax.m256d_f64[1];
                z[j + 1][1] = zzMin.m256d_f64[1];
            }

            // third element
            if ( (j + 2) < filterWidth) {
                z[j + 2][0] = zzMax.m256d_f64[2];
                z[j + 2][1] = zzMin.m256d_f64[2];
            }

            // fourth element
            if ( (j + 3) < filterWidth) {
                z[j + 3][0] = zzMax.m256d_f64[3];
                z[j + 3][1] = zzMin.m256d_f64[3];
            }

        }

    }

}


static PyObject* pyPseudovelocityBank(PyObject *self, PyObject *args) {

    PyObject *pyFilters=NULL, *pyX=NULL, *pyZ=NULL;
    PyObject *npFilters=NULL, *npX=NULL, *npZ=NULL;

    // parse the args from the arg tuple
    // the type "OO" should, I believe, give us two non-specified objects
    if (!PyArg_ParseTuple(args, "OOO", &pyFilters, &pyX, &pyZ)) {
        return NULL;
    }

    npFilters = PyArray_FROM_OTF(pyFilters, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    npX = PyArray_FROM_OTF(pyX, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    npZ = PyArray_FROM_OTF(pyZ, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);

    // check that filters is (nX6), x is one-dimensional
    if (PyArray_NDIM(npFilters) != 2) {
        PyErr_SetString(PyExc_TypeError, "Filter bank was not a 2D array");
        return NULL;
    } else if (PyArray_DIMS(npFilters)[1] != 6) {
        PyErr_SetString(PyExc_TypeError, "Filter bank was not an nX6 array");
        return NULL;
    } else if (PyArray_NDIM(npX) != 1) {
        PyErr_SetString(PyExc_TypeError, "x was not a 1D array");
        return NULL;
    } else if (PyArray_NDIM(npZ) != 2) {
        PyErr_SetString(PyExc_TypeError, "z was not a 2D array");
        return NULL;
    } else if (PyArray_DIMS(npZ)[1] != 2) {
        PyErr_SetString(PyExc_TypeError, "Second dimension of z is not 2");
        return NULL;
    } else if (PyArray_DIMS(npZ)[0] != PyArray_DIMS(npFilters)[0]) {
        PyErr_SetString(PyExc_TypeError, "First dimension of z did not match the width of the filterbank");
        return NULL;
    }

    // grab data from arrays
    double *filters = PyArray_DATA(npFilters);
    double *x = PyArray_DATA(npX);
    double *z = PyArray_DATA(npZ);

    // run the thing
    psBank(filters, x, z, PyArray_DIMS(npFilters)[0], PyArray_DIMS(npX)[0]);

    return pyZ;

}

/*  PyMethodDef defines each of the methods used in
 *
 */
static PyMethodDef PsMethods[] = {
    {"_ps_bank",  pyPseudovelocityBank, METH_VARARGS,
     "Calculate a full bank of pseudovelocity peaks"},
    {NULL, NULL, 0, NULL}        // Snentinel, marks no more methods
};

PyDoc_STRVAR(mod_doc,
    "This module contains methods to accelerate pseudovelocity math, which isn't"
    "well optimizable in numpy.\n"
    "`_ps_bank` specificaly takes a bank of Second Order Section filters and, "
    "rather than compute them in series like a normal SOS filter, instead calculates "
    "them all in parallel and accumulates information"
    );


/*  PyModuleDef defines metadata about this module
 *
 */
static struct PyModuleDef ps_module = {
    PyModuleDef_HEAD_INIT,
    "_ps_bank",   // name of module
    mod_doc,      // module documentation
    -1,           // size of per-interpreter state of the module,
                  //or -1 if the module keeps state in global variables.
    PsMethods
};

PyMODINIT_FUNC PyInit_pseudovelocity(void) {
    import_array();
    return PyModule_Create(&ps_module);
}
