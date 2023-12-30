package divcols

/*
#cgo CFLAGS: -x objective-c
#cgo LDFLAGS: -framework Metal -framework MetalPerformanceShaders -framework CoreGraphics -framework Foundation

#include "kernel.h"

void* divColsKernelCreate(void *device, const char *kernelSource) {
    return [[DivColsKernelImpl alloc] initWithDevice:(id<MTLDevice>)device
		kernelSource:[NSString stringWithUTF8String:kernelSource]];
}

void divColsForward(
    void *kernel,
    void *commandBuffer,
    void *inputData,
    void *weightsData,
    void *outputData,
	uint rowWidth,
	uint colHeight
) {

    [(__bridge DivColsKernelImpl*)kernel forward:(id<MTLCommandBuffer>)commandBuffer
        inputData:(id<MTLBuffer>)inputData
        weightsData:(id<MTLBuffer>)weightsData
        outputData:(id<MTLBuffer>)outputData
		rowWidth:(uint)rowWidth
        colHeight:(uint)colHeight
	];
}

void divColsBackward(
    void *kernel,
    void *commandBuffer,
    void *inputData,
    void *inputGrad,
	void *weightsData,
	void *weightsGrad,
    void *outputData,
    void *outputGrad,
	uint rowWidth,
	uint colHeight
) {
    [(__bridge DivColsKernelImpl*)kernel backward:(id<MTLCommandBuffer>)commandBuffer
        inputData:(id<MTLBuffer>)inputData
        inputGrad:(id<MTLBuffer>)inputGrad
        weightsData:(id<MTLBuffer>)weightsData
        weightsGrad:(id<MTLBuffer>)weightsGrad
        outputData:(id<MTLBuffer>)outputData
        outputGrad:(id<MTLBuffer>)outputGrad
		rowWidth:(uint)rowWidth
		colHeight:(uint)colHeight
	];
}

*/
import "C"
import (
	_ "embed"
	"unsafe"
)

//go:embed kernel.metal
var metalFunctions string

func New(deviceID unsafe.Pointer) *Kernel {
	cKernelString := C.CString(metalFunctions)
	defer C.free(unsafe.Pointer(cKernelString))
	return &Kernel{
		deviceID: deviceID,
		kernelID: C.divColsKernelCreate(deviceID, cKernelString),
	}
}

type Kernel struct {
	deviceID unsafe.Pointer
	kernelID unsafe.Pointer
}

func (k *Kernel) Forward(
	commandBufferID unsafe.Pointer,
	inputData unsafe.Pointer,
	weightsData unsafe.Pointer,
	outputData unsafe.Pointer,
	rowWidth int,
	colHeight int,
) {
	C.divColsForward(
		k.kernelID,
		commandBufferID,
		inputData,
		weightsData,
		outputData,
		C.uint(rowWidth),
		C.uint(colHeight),
	)
}

func (k *Kernel) Backward(
	commandBufferID unsafe.Pointer,
	inputData unsafe.Pointer,
	inputGrad unsafe.Pointer,
	weightsData unsafe.Pointer,
	weightsGrad unsafe.Pointer,
	outputData unsafe.Pointer,
	outputGrad unsafe.Pointer,
	rowWidth int,
	colHeight int,
) {
	C.divColsBackward(
		k.kernelID,
		commandBufferID,
		inputData,
		inputGrad,
		weightsData,
		weightsGrad,
		outputData,
		outputGrad,
		C.uint(rowWidth),
		C.uint(colHeight),
	)
}
