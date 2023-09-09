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
		device:   queue.device,
	}

	queue.buffer = buffer
	return queue.buffer
}

type MTLCommandBuffer struct {
	id       unsafe.Pointer
	deviceID unsafe.Pointer
	device   *MTLDevice

	uncommitted int64
	completed   bool
	released    bool

	mu sync.Mutex
}

func (b *MTLCommandBuffer) Release() {
	if !b.released {
		C.releaseCommandBuffer(b.id)
		b.released = true
		b.completed = true
	}
}

func (b *MTLCommandBuffer) ClearMTLBuffer(buffer *MTLBuffer) {
	b.mu.Lock()
	C.fillMTLBuffer(b.device.kernels.GetKernelID("fill"), b.id, buffer.bufferID, 0.0)
	b.uncommitted++
	b.mu.Unlock()
}

func (b *MTLCommandBuffer) FillMTLBuffer(buffer *MTLBuffer, value float32) {
	b.mu.Lock()
	C.fillMTLBuffer(b.device.kernels.GetKernelID("fill"), b.id, buffer.bufferID, C.float(value))
	b.uncommitted++
	b.mu.Unlock()
}

func (b *MTLCommandBuffer) FillMTLBufferPart(buffer *MTLBuffer, value float32, offset, length int) {
	b.mu.Lock()
	C.fillPartMTLBuffer(b.device.kernels.GetKernelID("fill"), b.id, buffer.bufferID, C.uint(offset*4), C.uint(length*4), C.float(value))
	b.uncommitted++
	b.mu.Unlock()
}

func (b *MTLCommandBuffer) ReLuMTLBuffer(destinationBuffer, sourceBuffer *MTLBuffer) {
	b.mu.Lock()
	C.reluMTLBuffer(b.device.kernels.GetKernelID("relu_fwd"), b.id, destinationBuffer.bufferID, sourceBuffer.bufferID)
	b.uncommitted++
	b.mu.Unlock()
}

func (b *MTLCommandBuffer) ReLuMTLBufferBwd(destinationBuffer, sourceBuffer, maskBuffer *MTLBuffer) {
	b.mu.Lock()
	C.reluMTLBufferBwd(b.device.kernels.GetKernelID("relu_bwd"), b.id, destinationBuffer.bufferID, sourceBuffer.bufferID, maskBuffer.bufferID)
	b.uncommitted++
	b.mu.Unlock()
}

func (b *MTLCommandBuffer) MulBuffer(destinationBuffer, multiplierBuffer *MTLBuffer) {
	b.mu.Lock()
	C.mulBuffer(b.device.kernels.GetKernelID("mul"), b.id, destinationBuffer.bufferID, multiplierBuffer.bufferID)
	b.uncommitted++
	b.mu.Unlock()
}

func (b *MTLCommandBuffer) DropoutBuffer(
	destinationBuffer,
	sourceBuffer,
	maskOutBuffer *MTLBuffer,
	probability float32,
) {
	b.mu.Lock()
	C.dropoutBuffer(
		b.device.kernels.GetKernelID("dropout"),
		b.id,
		destinationBuffer.bufferID,
		sourceBuffer.bufferID,
		maskOutBuffer.bufferID,
		C.float(probability),
	)
	b.uncommitted++
	b.mu.Unlock()
}

func (b *MTLCommandBuffer) SoftmaxBuffer(
	destinationBuffer *MTLBuffer,
	sourceBuffer *MTLBuffer,
	sumOutBuffer *MTLBuffer,
	colsCount, rowsCount, offset int,
) {
	cKernelString := C.CString(kernelSoftmax)
	defer C.free(unsafe.Pointer(cKernelString))

	b.mu.Lock()
	C.softmaxBuffer(
		b.deviceID, b.id,
		destinationBuffer.bufferID,
		sourceBuffer.bufferID,
		sumOutBuffer.bufferID,
		C.uint(colsCount),
		C.uint(rowsCount),
		C.uint(offset*4),
		cKernelString,
	)
	b.uncommitted++
	b.mu.Unlock()
}

func (b *MTLCommandBuffer) SoftmaxBufferTril(
	destinationBuffer *MTLBuffer,
	sourceBuffer *MTLBuffer,
	//maxOutBuffer *MTLBuffer,
	//sumOutBuffer *MTLBuffer,
	colsCount, rowsCount, offset int,
) {
	b.mu.Lock()
	C.softmaxBufferTril(
		b.device.kernels.GetKernelID("softmax_tril"),
		b.id,
		destinationBuffer.bufferID,
		sourceBuffer.bufferID,
		//maxOutBuffer.bufferID,
		//sumOutBuffer.bufferID,
		C.uint(colsCount),
		C.uint(rowsCount),
		C.uint(offset*4),
	)
	b.uncommitted++
	b.mu.Unlock()
}

func (b *MTLCommandBuffer) SoftmaxBufferTrilBwd(
	destinationBuffer *MTLBuffer,
	sourceBuffer *MTLBuffer,
	softmaxBuffer *MTLBuffer,
	//softmaxGradBuffer *MTLBuffer,
	//sumOutBuffer *MTLBuffer,
	colsCount, rowsCount, offset int,
) {
	b.mu.Lock()
	C.softmaxBufferTrilBwd(
		b.device.kernels.GetKernelID("softmax_tril_bwd"),
		b.id,
		destinationBuffer.bufferID,
		sourceBuffer.bufferID,
		softmaxBuffer.bufferID,
		//softmaxGradBuffer.bufferID,
		//sumOutBuffer.bufferID,
		C.uint(colsCount),
		C.uint(rowsCount),
		C.uint(offset*4),
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
