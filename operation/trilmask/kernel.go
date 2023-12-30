package trilmask

/*
#cgo CFLAGS: -x objective-c
#cgo LDFLAGS: -framework Metal -framework MetalPerformanceShaders -framework CoreGraphics -framework Foundation

#include "kernel.h"

void* trilMaskKernelCreate(void *device, const char *kernelSource) {
    return [[TrilMaskKernelImpl alloc] initWithDevice:(id<MTLDevice>)device
		kernelSource:[NSString stringWithUTF8String:kernelSource]];
}

void trilMaskForward(
    void *kernel,
    void *commandBuffer,
    void *inputData,
    void *outputData,
	float mask,
	uint colsCount,
	uint rowsCount
) {

    [(__bridge TrilMaskKernelImpl*)kernel forward:(id<MTLCommandBuffer>)commandBuffer
        inputData:(id<MTLBuffer>)inputData
        outputData:(id<MTLBuffer>)outputData
		mask:(float)mask
        colsCount:(uint)colsCount
        rowsCount:(uint)rowsCount
	];
}

void trilMaskBackward(
    void *kernel,
    void *commandBuffer,
    void *inputGrad,
    void *outputGrad,
	uint colsCount,
	uint rowsCount
) {

    [(__bridge TrilMaskKernelImpl*)kernel backward:(id<MTLCommandBuffer>)commandBuffer
        inputGrad:(id<MTLBuffer>)inputGrad
        outputGrad:(id<MTLBuffer>)outputGrad
        colsCount:(uint)colsCount
        rowsCount:(uint)rowsCount
	];
}

*/
import "C"
import (
	_ "embed"
	"math"
	"unsafe"
)

//go:embed kernel.metal
var metalFunctions string

func New(deviceID unsafe.Pointer) *Kernel {
	cKernelString := C.CString(metalFunctions)
	defer C.free(unsafe.Pointer(cKernelString))
	return &Kernel{
		deviceID: deviceID,
		kernelID: C.trilMaskKernelCreate(deviceID, cKernelString),
	}
}

type Kernel struct {
	deviceID unsafe.Pointer
	kernelID unsafe.Pointer
}

func (k *Kernel) Forward(
	commandBufferID unsafe.Pointer,
	inputData unsafe.Pointer,
	outputData unsafe.Pointer,
	colsCount int,
	rowsCount int,
) {
	C.trilMaskForward(
		k.kernelID,
		commandBufferID,
		inputData,
		outputData,
		C.float(float32(math.Inf(-1))),
		C.uint(colsCount),
		C.uint(rowsCount),
	)
}

func (k *Kernel) Backward(
	commandBufferID unsafe.Pointer,
	inputGrad unsafe.Pointer,
	outputGrad unsafe.Pointer,
	colsCount int,
	rowsCount int,
) {
	C.trilMaskBackward(
		k.kernelID,
		commandBufferID,
		inputGrad,
		outputGrad,
		C.uint(colsCount),
		C.uint(rowsCount),
	)
}
