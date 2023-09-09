#include <CoreGraphics/CoreGraphics.h>
#include <MetalPerformanceShaders/MetalPerformanceShaders.h>
#include <Metal/Metal.h>
#include "krn_mtl_buffer_fill.h"
#include "krn_mtl_buffer_relu_fwd.h"
#include "krn_mtl_buffer_relu_bwd.h"
#include "krn_mtl_buffer_mul.h"
#include "krn_mtl_buffer_dropout.h"
#include "krn_mtl_buffer_softmax.h"
#include "krn_mtl_buffer_softmax_tril.h"
#include "krn_mtl_buffer_softmax_tril_bwd.h"

void* createFillKernel(void *deviceID, const char *kernelSource);
void* createReLUFwdKernel(void *deviceID, const char *kernelSource);
void* createReLUBwdKernel(void *deviceID, const char *kernelSource);
void* createMulKernel(void *deviceID, const char *kernelSource);
void* createDropoutKernel(void *deviceID, const char *kernelSource);

void* createSoftmaxBufferTrilKernel(void *deviceID, const char *kernelSource);
void* createSoftmaxBufferTrilBwdKernel(void *deviceID, const char *kernelSource);