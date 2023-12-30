package trilmask2

/*
#cgo CFLAGS: -x objective-c
#cgo LDFLAGS: -framework Metal -framework MetalPerformanceShaders -framework CoreGraphics -framework Foundation

#include "kernel.h"

void* trilMask2KernelCreate(void *device, const char *kernelSource) {
    return [[TrilMask2KernelImpl alloc] initWithDevice:(id<MTLDevice>)device
		kernelSource:[NSString stringWithUTF8String:kernelSource]];
}

void trilMask2Forward(
    void *kernel,
    void *commandBuffer,
    void *inputData,
    void *outputData,
	float mask,
	uint colsCount,
	uint rowsCount
) {

    [(__bridge TrilMask2KernelImpl*)kernel forward:(id<MTLCommandBuffer>)commandBuffer
        inputData:(id<MTLBuffer>)inputData
        outputData:(id<MTLBuffer>)outputData
		mask:(float)mask
        colsCount:(uint)colsCount
        rowsCount:(uint)rowsCount
	];
}

void trilMask2Backward(
    void *kernel,
    void *commandBuffer,
    void *inputGrad,
	uint colsCount,
	uint rowsCount
) {

    [(__bridge TrilMask2KernelImpl*)kernel backward:(id<MTLCommandBuffer>)commandBuffer
        inputGrad:(id<MTLBuffer>)inputGrad
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
		kernelID: C.trilMask2KernelCreate(deviceID, cKernelString),
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
	C.trilMask2Forward(
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
	colsCount int,
	rowsCount int,
) {
	C.trilMask2Backward(
		k.kernelID,
		commandBufferID,
		inputGrad,
		C.uint(colsCount),
		C.uint(rowsCount),
	)
}
