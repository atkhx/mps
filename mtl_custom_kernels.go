package mps

/*
#cgo CFLAGS: -x objective-c
#cgo LDFLAGS: -framework Metal -framework MetalPerformanceShaders -framework CoreGraphics -framework Foundation
#include "mtl_custom_kernels.h"
*/
import "C"
import (
	_ "embed"
	"unsafe"
)

//go:embed kernel/krn_mtl_buffer_fill.metal
var kernelFill string

//go:embed kernel/krn_mtl_buffer_relu_fwd.metal
var kernelReLU string

//go:embed kernel/krn_mtl_buffer_relu_bwd.metal
var kernelReLUBwd string

//go:embed kernel/krn_mtl_buffer_mul.metal
var kernelMul string

//go:embed kernel/krn_mtl_buffer_dropout.metal
var kernelDropout string

//go:embed kernel/krn_mtl_buffer_softmax.metal
var kernelSoftmax string

//go:embed kernel/krn_mtl_buffer_softmax_tril.metal
var kernelSoftmaxTril string

//go:embed kernel/krn_mtl_buffer_softmax_tril_bwd.metal
var kernelSoftmaxTrilBwd string

type MTLCustomKernels struct {
	kernelIDs map[string]unsafe.Pointer
}

func (d *MTLDevice) CreateCustomKernels() *MTLCustomKernels {
	return &MTLCustomKernels{
		kernelIDs: map[string]unsafe.Pointer{
			"fill": func() unsafe.Pointer {
				cKernelString := C.CString(kernelFill)
				defer C.free(unsafe.Pointer(cKernelString))
				return C.createFillKernel(d.deviceID, cKernelString)
			}(),
			"relu_fwd": func() unsafe.Pointer {
				cKernelString := C.CString(kernelReLU)
				defer C.free(unsafe.Pointer(cKernelString))
				return C.createReLUFwdKernel(d.deviceID, cKernelString)
			}(),
			"relu_bwd": func() unsafe.Pointer {
				cKernelString := C.CString(kernelReLUBwd)
				defer C.free(unsafe.Pointer(cKernelString))
				return C.createReLUBwdKernel(d.deviceID, cKernelString)
			}(),
			"mul": func() unsafe.Pointer {
				cKernelString := C.CString(kernelMul)
				defer C.free(unsafe.Pointer(cKernelString))
				return C.createMulKernel(d.deviceID, cKernelString)
			}(),
			"dropout": func() unsafe.Pointer {
				cKernelString := C.CString(kernelDropout)
				defer C.free(unsafe.Pointer(cKernelString))
				return C.createDropoutKernel(d.deviceID, cKernelString)
			}(),
			"softmax_tril": func() unsafe.Pointer {
				cKernelString := C.CString(kernelSoftmaxTril)
				defer C.free(unsafe.Pointer(cKernelString))
				return C.createSoftmaxBufferTrilKernel(d.deviceID, cKernelString)
			}(),
			"softmax_tril_bwd": func() unsafe.Pointer {
				cKernelString := C.CString(kernelSoftmaxTrilBwd)
				defer C.free(unsafe.Pointer(cKernelString))
				return C.createSoftmaxBufferTrilBwdKernel(d.deviceID, cKernelString)
			}(),
		},
	}
}

func (k *MTLCustomKernels) GetKernelID(name string) unsafe.Pointer {
	return k.kernelIDs[name]
}
