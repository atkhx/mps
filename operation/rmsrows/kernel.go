package rmsrows

/*
#cgo CFLAGS: -x objective-c
#cgo LDFLAGS: -framework Metal -framework MetalPerformanceShaders -framework CoreGraphics -framework Foundation

#include "kernel.h"

void* rmsRowsKernelCreate(void *device, const char *kernelSource) {
    return [[RmsRowsKernelImpl alloc] initWithDevice:(id<MTLDevice>)device
		kernelSource:[NSString stringWithUTF8String:kernelSource]];
}

void forward(
    void *kernel,
    void *commandBuffer,
    void *inputData,
    void *outputData,
    uint chunkSize
) {
    [(__bridge RmsRowsKernelImpl*)kernel forward:(id<MTLCommandBuffer>)commandBuffer
        inputData:(id<MTLBuffer>)inputData
        outputData:(id<MTLBuffer>)outputData
        chunkSize:chunkSize];
}

void backward(
    void *kernel,
    void *commandBuffer,
    void *inputData,
    void *outputData,
    void *inputGrads,
    void *outputGrads,
    uint chunkSize
) {
    [(__bridge RmsRowsKernelImpl*)kernel backward:(id<MTLCommandBuffer>)commandBuffer
        inputData:(id<MTLBuffer>)inputData
        outputData:(id<MTLBuffer>)outputData
        inputGrads:(id<MTLBuffer>)inputGrads
        outputGrads:(id<MTLBuffer>)outputGrads
        chunkSize:chunkSize];
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
		kernelID: C.rmsRowsKernelCreate(deviceID, cKernelString),
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
	chunkSize int,
) {
	C.forward(
		k.kernelID,
		commandBufferID,
		inputData,
		outputData,
		C.uint(chunkSize),
	)
}

func (k *Kernel) Backward(
	commandBufferID unsafe.Pointer,
	inputData unsafe.Pointer,
	outputData unsafe.Pointer,
	inputGrads unsafe.Pointer,
	outputGrads unsafe.Pointer,
	chunkSize int,
) {
	C.backward(
		k.kernelID,
		commandBufferID,
		inputData,
		outputData,
		inputGrads,
		outputGrads,
		C.uint(chunkSize),
	)
}
