package roperows

/*
#cgo CFLAGS: -x objective-c
#cgo LDFLAGS: -framework Metal -framework MetalPerformanceShaders -framework CoreGraphics -framework Foundation

#include "kernel.h"

void* ropeRowsKernelCreate(void *device, const char *kernelSource) {
    return [[ropeRowsKernelImpl alloc] initWithDevice:(id<MTLDevice>)device
		kernelSource:[NSString stringWithUTF8String:kernelSource]];
}

void ropeRowsForward(
    void *kernel,
    void *commandBuffer,
    void *inputData,
    void *outputData,
    uint featuresCount,
    uint headSize,
    uint contextLength

) {

    [(__bridge ropeRowsKernelImpl*)kernel forward:(id<MTLCommandBuffer>)commandBuffer
        inputData:(id<MTLBuffer>)inputData
        outputData:(id<MTLBuffer>)outputData
		featuresCount:(uint)featuresCount
        headSize:(uint)headSize
        contextLength:(uint)contextLength
	];
}

void ropeRowsBackward(
    void *kernel,
    void *commandBuffer,
    void *inputGrad,
    void *outputGrad,
    uint featuresCount,
    uint headSize,
    uint contextLength
) {
    [(__bridge ropeRowsKernelImpl*)kernel backward:(id<MTLCommandBuffer>)commandBuffer
        inputGrad:(id<MTLBuffer>)inputGrad
        outputGrad:(id<MTLBuffer>)outputGrad
		featuresCount:(uint)featuresCount
        headSize:(uint)headSize
        contextLength:(uint)contextLength
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
		kernelID: C.ropeRowsKernelCreate(deviceID, cKernelString),
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
	featuresCount int,
	headSize int,
	contextLength int,
) {
	C.ropeRowsForward(
		k.kernelID,
		commandBufferID,
		inputData,
		outputData,
		C.uint(featuresCount),
		C.uint(headSize),
		C.uint(contextLength),
	)
}

func (k *Kernel) Backward(
	commandBufferID unsafe.Pointer,
	inputGrad unsafe.Pointer,
	outputGrad unsafe.Pointer,
	featuresCount int,
	headSize int,
	contextLength int,
) {
	C.ropeRowsBackward(
		k.kernelID,
		commandBufferID,
		inputGrad,
		outputGrad,
		C.uint(featuresCount),
		C.uint(headSize),
		C.uint(contextLength),
	)
}
