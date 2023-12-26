package rmsnorm

/*
#cgo CFLAGS: -x objective-c
#cgo LDFLAGS: -framework Metal -framework MetalPerformanceShaders -framework CoreGraphics -framework Foundation

#include "kernel.h"

void* rmsNormKernelCreate(void *device, const char *kernelSource) {
    return [[RMSNormKernelImpl alloc] initWithDevice:(id<MTLDevice>)device
		kernelSource:[NSString stringWithUTF8String:kernelSource]];
}

void rmsNormForward(
    void *kernel,
    void *commandBuffer,
    void *inputData,
    void *outputData,
    void *aggData,
    uint chunkSize
) {
    [(__bridge RMSNormKernelImpl*)kernel forward:(id<MTLCommandBuffer>)commandBuffer
        inputData:(id<MTLBuffer>)inputData
        outputData:(id<MTLBuffer>)outputData
        aggData:(id<MTLBuffer>)aggData
        chunkSize:chunkSize];
}

void rmsNormBackward(
    void *kernel,
    void *commandBuffer,
    void *inputData,
    void *inputGrad,
    void *outputData,
    void *outputGrad,
    void *aggData,
    void *aggGrad,
    uint chunkSize
) {
    [(__bridge RMSNormKernelImpl*)kernel backward:(id<MTLCommandBuffer>)commandBuffer
        inputData:(id<MTLBuffer>)inputData
        inputGrad:(id<MTLBuffer>)inputGrad
        outputData:(id<MTLBuffer>)outputData
        outputGrad:(id<MTLBuffer>)outputGrad
        aggData:(id<MTLBuffer>)aggData
        aggGrad:(id<MTLBuffer>)aggGrad
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
		kernelID: C.rmsNormKernelCreate(deviceID, cKernelString),
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
	aggData unsafe.Pointer,
	chunkSize int,
) {
	C.rmsNormForward(
		k.kernelID,
		commandBufferID,
		inputData,
		outputData,
		aggData,
		C.uint(chunkSize),
	)
}

func (k *Kernel) Backward(
	commandBufferID unsafe.Pointer,
	inputData unsafe.Pointer,
	inputGrad unsafe.Pointer,
	outputData unsafe.Pointer,
	outputGrad unsafe.Pointer,
	aggData unsafe.Pointer,
	aggGrad unsafe.Pointer,
	chunkSize int,
) {
	C.rmsNormBackward(
		k.kernelID,
		commandBufferID,
		inputData,
		inputGrad,
		outputData,
		outputGrad,
		aggData,
		aggGrad,
		C.uint(chunkSize),
	)
}
