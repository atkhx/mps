package mps

/*
#cgo CFLAGS: -x objective-c
#cgo LDFLAGS: -framework Metal -framework MetalPerformanceShaders -framework CoreGraphics -framework Foundation
#include "mtl_command_buffer.h"
#include "mps_matrix_multiply.h"
*/
import "C"
import (
	"context"
	_ "embed"
	"sync"
	"unsafe"
)

func ContextWithCommandBuffer(ctx context.Context, buffer *MTLCommandBuffer) context.Context {
	return context.WithValue(ctx, "MTLCommandBuffer", buffer)
}

func CommandBufferFromContext(ctx context.Context) *MTLCommandBuffer {
	return ctx.Value("MTLCommandBuffer").(*MTLCommandBuffer)
}

//go:embed kernel/krn_mtl_buffer_relu_fwd.metal
var kernelReLU string

//go:embed kernel/krn_mtl_buffer_relu_bwd.metal
var kernelReLUBwd string

//go:embed kernel/krn_mtl_buffer_fill.metal
var kernelFill string

//go:embed kernel/krn_mtl_buffer_mul.metal
var kernelMul string

//go:embed kernel/krn_mtl_buffer_dropout.metal
var kernelDropout string

func (queue *MTLCommandQueue) CreateCommandBuffer() *MTLCommandBuffer {
	switch {
	case queue.buffer != nil && !queue.buffer.completed:
		return queue.buffer
	case queue.buffer != nil && queue.buffer.completed:
		queue.buffer.Release()
	}

	buffer := &MTLCommandBuffer{
		id:       C.createCommandBuffer(queue.queueID),
		deviceID: queue.deviceID,
	}

	queue.buffer = buffer
	return queue.buffer
}

type MTLCommandBuffer struct {
	id          unsafe.Pointer
	deviceID    unsafe.Pointer
	uncommitted int64
	completed   bool
	released    bool
	mu          sync.Mutex
}

func (b *MTLCommandBuffer) Release() {
	if !b.released {
		C.releaseCommandBuffer(b.id)
		b.released = true
		b.completed = true
	}
}

func (b *MTLCommandBuffer) ReLuMTLBuffer(destinationBuffer, sourceBuffer *MTLBuffer) {
	cKernelReLU := C.CString(kernelReLU)
	defer C.free(unsafe.Pointer(cKernelReLU))

	b.mu.Lock()
	C.reluMTLBuffer(b.deviceID, b.id, destinationBuffer.bufferID, sourceBuffer.bufferID, cKernelReLU)
	b.uncommitted++
	b.mu.Unlock()
}

func (b *MTLCommandBuffer) ReLuMTLBufferBwd(destinationBuffer, sourceBuffer, maskBuffer *MTLBuffer) {
	cKernelReLUBwd := C.CString(kernelReLUBwd)
	defer C.free(unsafe.Pointer(cKernelReLUBwd))

	b.mu.Lock()
	C.reluMTLBufferBwd(b.deviceID, b.id, destinationBuffer.bufferID, sourceBuffer.bufferID, maskBuffer.bufferID, cKernelReLUBwd)
	b.uncommitted++
	b.mu.Unlock()
}

func (b *MTLCommandBuffer) ClearMTLBuffer(buffer *MTLBuffer) {
	cKernelFill := C.CString(kernelFill)
	defer C.free(unsafe.Pointer(cKernelFill))

	b.mu.Lock()
	C.fillMTLBuffer(cKernelFill, b.deviceID, b.id, buffer.bufferID, 0.0)
	b.uncommitted++
	b.mu.Unlock()
}

func (b *MTLCommandBuffer) FillMTLBuffer(buffer *MTLBuffer, value float32) {
	cKernelFill := C.CString(kernelFill)
	defer C.free(unsafe.Pointer(cKernelFill))

	b.mu.Lock()
	C.fillMTLBuffer(cKernelFill, b.deviceID, b.id, buffer.bufferID, C.float(value))
	b.uncommitted++
	b.mu.Unlock()
}

func (b *MTLCommandBuffer) FillMTLBufferPart(buffer *MTLBuffer, value float32, offset, length int) {
	cKernelFill := C.CString(kernelFill)
	defer C.free(unsafe.Pointer(cKernelFill))

	b.mu.Lock()
	C.fillPartMTLBuffer(cKernelFill, b.deviceID, b.id, buffer.bufferID, C.uint(offset*4), C.uint(length*4), C.float(value))
	b.uncommitted++
	b.mu.Unlock()
}

func (b *MTLCommandBuffer) MulBuffer(destinationBuffer, multiplierBuffer *MTLBuffer) {
	cKernelString := C.CString(kernelMul)
	defer C.free(unsafe.Pointer(cKernelString))

	b.mu.Lock()
	C.mulBuffer(b.deviceID, b.id, destinationBuffer.bufferID, multiplierBuffer.bufferID, cKernelString)
	b.uncommitted++
	b.mu.Unlock()
}

func (b *MTLCommandBuffer) DropoutBuffer(
	destinationBuffer,
	sourceBufferBuffer,
	maskOutBuffer *MTLBuffer,
	probability float32,
) {
	cKernelString := C.CString(kernelDropout)
	defer C.free(unsafe.Pointer(cKernelString))

	b.mu.Lock()
	C.dropoutBuffer(
		b.deviceID, b.id,
		destinationBuffer.bufferID,
		sourceBufferBuffer.bufferID,
		maskOutBuffer.bufferID,
		C.float(probability),
		cKernelString,
	)
	b.uncommitted++
	b.mu.Unlock()
}

func (b *MTLCommandBuffer) Wait() {
	b.mu.Lock()
	if b.uncommitted > 0 {
		C.commitAndWaitUntilCompletedCommandBuffer(b.id)
		b.completed = true
	}
	b.uncommitted = 0
	b.mu.Unlock()
}
