//
// Created by jglrxavpok on 05/09/2020.
//

#ifndef CUDAUTILS_H
#define CUDAUTILS_H

#include <thrust/device_ptr.h>
#include <cuda.h>
#include <iostream>

#define checkCudaErrors(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}
#endif // CUDAUTILS_H