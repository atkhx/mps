package mps

/*
#cgo CFLAGS: -x objective-c
#cgo LDFLAGS: -framework Metal -framework MetalPerformanceShaders -framework CoreGraphics
#include "mtl_command_buffer.h"
#include "mps_matrix_multiply.h"
*/
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

func (b *MTLCommandBuffer) ClearMTLBuffer(buffer *MTLBuffer) {
	b.mu.Lock()
	C.clearMTLBuffer(b.deviceID, b.id, buffer.bufferID, 0.0)
	b.uncommitted++
	b.mu.Unlock()
}

func (b *MTLCommandBuffer) FillMTLBuffer(buffer *MTLBuffer, value float32) {
	b.mu.Lock()
	C.clearMTLBuffer(b.deviceID, b.id, buffer.bufferID, C.float(value))
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
