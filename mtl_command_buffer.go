package mps

import "C"
import (
	"context"
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
		id:       mtlCommandBufferCreate(queue.queueID),
		deviceID: queue.device.deviceID,
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
		mtlCommandBufferRelease(b.id)
		b.released = true
		b.completed = true
	}
}

func (b *MTLCommandBuffer) exclusive(operation func()) {
	b.mu.Lock()
	operation()
	b.uncommitted++
	b.mu.Unlock()
}

func (b *MTLCommandBuffer) ClearMTLBuffer(buffer *MTLBuffer) {
	b.exclusive(func() {
		customKernelFill(b.device.krnFill, b.id, buffer.bufferID, 0.0)
	})
}

func (b *MTLCommandBuffer) FillMTLBuffer(buffer *MTLBuffer, value float32) {
	b.exclusive(func() {
		customKernelFill(b.device.krnFill, b.id, buffer.bufferID, value)
	})
}

func (b *MTLCommandBuffer) FillMTLBufferPart(buffer *MTLBuffer, value float32, offset, length int) {
	b.exclusive(func() {
		customKernelFillPart(b.device.krnFill, b.id, buffer.bufferID, offset, length, value)
	})
}

func (b *MTLCommandBuffer) ReLuMTLBuffer(destinationBuffer, sourceBuffer *MTLBuffer) {
	b.exclusive(func() {
		customKernelReLUForward(b.device.krnReLUFwd, b.id, destinationBuffer.bufferID, sourceBuffer.bufferID)
	})
}

func (b *MTLCommandBuffer) ReLuMTLBufferBwd(destinationBuffer, sourceBuffer, maskBuffer *MTLBuffer) {
	b.exclusive(func() {
		customKernelReLUBackward(b.device.krnReLUBwd, b.id, destinationBuffer.bufferID, sourceBuffer.bufferID, maskBuffer.bufferID)
	})
}

func (b *MTLCommandBuffer) Mul(dst, src *MTLBuffer) {
	b.exclusive(func() {
		customKernelMul(b.device.krnMul, b.id, dst.bufferID, src.bufferID)
	})
}

func (b *MTLCommandBuffer) DropoutBuffer(
	destinationBuffer,
	sourceBuffer,
	maskOutBuffer *MTLBuffer,
	probability float32,
) {
	b.exclusive(func() {
		customKernelDropout(
			b.device.krnDropout,
			b.id,
			destinationBuffer.bufferID,
			sourceBuffer.bufferID,
			maskOutBuffer.bufferID,
			probability,
		)
	})
}

func (b *MTLCommandBuffer) SoftmaxBuffer(
	destinationBuffer *MTLBuffer,
	sourceBuffer *MTLBuffer,
	sumOutBuffer *MTLBuffer,
	colsCount, rowsCount, offset int,
) {
	b.exclusive(func() {
		customKernelSoftmaxForward(
			b.device.krnSoftmax,
			b.id,
			destinationBuffer.bufferID,
			sourceBuffer.bufferID,
			sumOutBuffer.bufferID,
			colsCount,
			rowsCount,
			offset,
		)
	})
}

func (b *MTLCommandBuffer) SoftmaxBufferTril(
	destinationBuffer *MTLBuffer,
	sourceBuffer *MTLBuffer,
	colsCount, rowsCount, offset int,
) {
	b.exclusive(func() {
		customKernelSoftmaxTrilFwdCreate(
			b.device.krnSoftmaxTrilFwd,
			b.id,
			destinationBuffer.bufferID,
			sourceBuffer.bufferID,
			colsCount,
			rowsCount,
			offset,
		)
	})
}

func (b *MTLCommandBuffer) SoftmaxBufferTrilBwd(
	destinationBuffer *MTLBuffer,
	sourceBuffer *MTLBuffer,
	softmaxBuffer *MTLBuffer,
	colsCount, rowsCount, offset int,
) {
	b.exclusive(func() {
		customKernelSoftmaxTrilBackward(
			b.device.krnSoftmaxTrilBwd,
			b.id,
			destinationBuffer.bufferID,
			sourceBuffer.bufferID,
			softmaxBuffer.bufferID,
			colsCount,
			rowsCount,
			offset,
		)
	})
}

func (b *MTLCommandBuffer) Wait() {
	b.exclusive(func() {
		if b.uncommitted > 0 {
			mtlCommandBufferCommitAndWaitUntilCompleted(b.id)
			b.completed = true
		}
		b.uncommitted = 0
	})
}
