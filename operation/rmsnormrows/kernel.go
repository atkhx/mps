package rmsnormrows

/*
#cgo CFLAGS: -x objective-c
#cgo LDFLAGS: -framework Metal -framework MetalPerformanceShaders -framework CoreGraphics -framework Foundation

#include "kernel.h"

void* RmsNormRowsKernelCreate(void *device, const char *kernelSource) {
    return [[RmsNormRowsKernelImpl alloc] initWithDevice:(id<MTLDevice>)device
		kernelSource:[NSString stringWithUTF8String:kernelSource]];
}

void rmsRowsForward(
    void *kernel,
    void *commandBuffer,
    void *inputData,
    void *outputData,
    void *rmsData,
    uint chunkSize
) {
    [(__bridge RmsNormRowsKernelImpl*)kernel forward:(id<MTLCommandBuffer>)commandBuffer
        inputData:(id<MTLBuffer>)inputData
        outputData:(id<MTLBuffer>)outputData
        rmsData:(id<MTLBuffer>)rmsData
        chunkSize:chunkSize];
}

void rmsRowsBackward(
    void *kernel,
    void *commandBuffer,
    void *inputData,
    void *inputGrad,
    void *outputData,
    void *outputGrad,
    void *rmsData,
    void *rmsGrad,
    uint chunkSize
) {
    [(__bridge RmsNormRowsKernelImpl*)kernel backward:(id<MTLCommandBuffer>)commandBuffer
        inputData:(id<MTLBuffer>)inputData
        inputGrad:(id<MTLBuffer>)inputGrad
        outputData:(id<MTLBuffer>)outputData
        outputGrad:(id<MTLBuffer>)outputGrad
        rmsData:(id<MTLBuffer>)rmsData
        rmsGrad:(id<MTLBuffer>)rmsGrad
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
		kernelID: C.RmsNormRowsKernelCreate(deviceID, cKernelString),
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
	rmsData unsafe.Pointer,
	chunkSize int,
) {
	C.rmsRowsForward(
		k.kernelID,
		commandBufferID,
		inputData,
		outputData,
		rmsData,
		C.uint(chunkSize),
	)
}

func (k *Kernel) Backward(
	commandBufferID unsafe.Pointer,
	inputData unsafe.Pointer,
	inputGrad unsafe.Pointer,
	outputData unsafe.Pointer,
	outputGrad unsafe.Pointer,
	rmsData unsafe.Pointer,
	rmsGrad unsafe.Pointer,
	chunkSize int,
) {
	C.rmsRowsBackward(
		k.kernelID,
		commandBufferID,
		inputData,
		inputGrad,
		outputData,
		outputGrad,
		rmsData,
		rmsGrad,
		C.uint(chunkSize),
	)
}
