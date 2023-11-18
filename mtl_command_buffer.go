package mps

import "C"
import (
	"context"
	"sync"
	"unsafe"

	"github.com/atkhx/mps/custom-kernel"
	"github.com/atkhx/mps/framework"
)

func ContextWithCommandBuffer(ctx context.Context, buffer *MTLCommandBuffer) context.Context {
	return context.WithValue(ctx, "MTLCommandBuffer", buffer)
}

func CommandBufferFromContext(ctx context.Context) *MTLCommandBuffer {
	return ctx.Value("MTLCommandBuffer").(*MTLCommandBuffer)
}

func NewMTLCommandBuffer(queue *MTLCommandQueue) *MTLCommandBuffer {
	return &MTLCommandBuffer{
		ID:       framework.MTLCommandBufferCreate(queue.queueID),
		deviceID: queue.device.deviceID,
		device:   queue.device,
	}
}

type MTLCommandBuffer struct {
	ID       unsafe.Pointer
	deviceID unsafe.Pointer
	device   *MTLDevice

	uncommitted int64
	completed   bool
	released    bool

	mu sync.Mutex
}

func (b *MTLCommandBuffer) Release() {
	if !b.released {
		framework.MTLCommandBufferRelease(b.ID)
		b.released = true
		b.completed = true
	}
}

func (b *MTLCommandBuffer) Wait() {
	b.Exclusive(func() {
		if b.uncommitted > 0 {
			framework.MTLCommandBufferCommitAndWaitUntilCompleted(b.ID)
			b.completed = true
		}
		b.uncommitted = 0
	})
}

func (b *MTLCommandBuffer) Exclusive(operation func()) {
	b.mu.Lock()
	operation()
	b.uncommitted++
	b.mu.Unlock()
}

func (b *MTLCommandBuffer) Copy(dst, src *MTLBuffer, dstOffset, srcOffset, length int) {
	b.Exclusive(func() {
		custom_kernel.CustomKernelCopy(b.device.CustomKernels, b.ID, dst.BufferID, src.BufferID, dstOffset, srcOffset, length)
	})
}

func (b *MTLCommandBuffer) CopyWHD(dst, src *MTLBuffer, W, H, D int) {
	b.Exclusive(func() {
		custom_kernel.CustomKernelCopyWHD(b.device.CustomKernels, b.ID, dst.BufferID, src.BufferID, W, H, D)
	})
}

func (b *MTLCommandBuffer) ClearMTLBuffer(buffer *MTLBuffer) {
	b.Exclusive(func() {
		custom_kernel.CustomKernelFill(b.device.CustomKernels, b.ID, buffer.BufferID, 0.0, 0, buffer.Length)
	})
}

func (b *MTLCommandBuffer) FillMTLBuffer(buffer *MTLBuffer, value float32) {
	b.Exclusive(func() {
		custom_kernel.CustomKernelFill(b.device.CustomKernels, b.ID, buffer.BufferID, value, 0, buffer.Length)
	})
}

func (b *MTLCommandBuffer) FillMTLBufferPart(buffer *MTLBuffer, value float32, offset, length int) {
	b.Exclusive(func() {
		custom_kernel.CustomKernelFill(b.device.CustomKernels, b.ID, buffer.BufferID, value, offset, length)
	})
}

func (b *MTLCommandBuffer) Add(dst, src *MTLBuffer, dstOffset, srcOffset, length int) {
	b.Exclusive(func() {
		custom_kernel.CustomKernelAdd(b.device.CustomKernels, b.ID, dst.BufferID, src.BufferID, dstOffset, srcOffset, length)
	})
}

func (b *MTLCommandBuffer) AddTo(dst, aBuffer, bBuffer *MTLBuffer) {
	b.Exclusive(func() {
		custom_kernel.CustomKernelAddTo(b.device.CustomKernels, b.ID, dst.BufferID, aBuffer.BufferID, bBuffer.BufferID)
	})
}

func (b *MTLCommandBuffer) AddToWHD(dst, aBuffer, bBuffer *MTLBuffer, K float32, W, H, D int) {
	b.Exclusive(func() {
		custom_kernel.CustomKernelAddToWHD(b.device.CustomKernels, b.ID, dst.BufferID, aBuffer.BufferID, bBuffer.BufferID, K, W, H, D)
	})
}

func (b *MTLCommandBuffer) AddToWHDBwd(aGrad, bGrad, oGrad *MTLBuffer, W, H, D int) {
	b.Exclusive(func() {
		custom_kernel.CustomKernelAddToWHDBwd(b.device.CustomKernels, b.ID, aGrad.BufferID, bGrad.BufferID, oGrad.BufferID, W, H, D)
	})
}

func (b *MTLCommandBuffer) AddScalar(dst *MTLBuffer, value float32) {
	b.Exclusive(func() {
		custom_kernel.CustomKernelAddScalar(b.device.CustomKernels, b.ID, dst.BufferID, value)
	})
}

func (b *MTLCommandBuffer) Mul(dst, src *MTLBuffer, dstOffset, srcOffset, length int) {
	b.Exclusive(func() {
		custom_kernel.CustomKernelMul(b.device.CustomKernels, b.ID, dst.BufferID, src.BufferID, dstOffset, srcOffset, length)
	})
}

func (b *MTLCommandBuffer) ReLuMTLBuffer(dstBuffer, srcBuffer *MTLBuffer) {
	b.Exclusive(func() {
		custom_kernel.CustomKernelReLU(b.device.CustomKernels, b.ID, dstBuffer.BufferID, srcBuffer.BufferID)
	})
}

func (b *MTLCommandBuffer) ReLuMTLBufferBwd(dstBuffer, srcBuffer, maskBuffer *MTLBuffer) {
	b.Exclusive(func() {
		custom_kernel.CustomKernelReLUBackward(b.device.CustomKernels, b.ID, dstBuffer.BufferID, srcBuffer.BufferID, maskBuffer.BufferID)
	})
}

func (b *MTLCommandBuffer) SoftmaxBuffer(
	destinationBuffer *MTLBuffer,
	sourceBuffer *MTLBuffer,
	sumOutBuffer *MTLBuffer,
	colsCount, rowsCount, offset int,
) {
	b.Exclusive(func() {
		custom_kernel.CustomKernelSoftmaxForward(
			b.device.CustomKernels,
			b.ID,
			destinationBuffer.BufferID,
			sourceBuffer.BufferID,
			sumOutBuffer.BufferID,
			colsCount,
			rowsCount,
			offset,
		)
	})
}

func (b *MTLCommandBuffer) DropoutBuffer(
	dstBuffer,
	srcBuffer,
	mskBuffer *MTLBuffer,
	probability float32,
) {
	b.Exclusive(func() {
		custom_kernel.CustomKernelDropout(
			b.device.CustomKernels,
			b.ID,
			dstBuffer.BufferID,
			srcBuffer.BufferID,
			mskBuffer.BufferID,
			probability,
		)
	})
}

func (b *MTLCommandBuffer) DropoutBwdBuffer(
	dstBuffer,
	srcBuffer,
	mskBuffer *MTLBuffer,
	probability float32,
) {
	b.Exclusive(func() {
		custom_kernel.CustomKernelDropoutBwd(
			b.device.CustomKernels,
			b.ID,
			dstBuffer.BufferID,
			srcBuffer.BufferID,
			mskBuffer.BufferID,
			probability,
		)
	})
}

func (b *MTLCommandBuffer) UpdateWithAdam(
	dataBuffer,
	gradBuffer,
	mBuffer,
	vBuffer *MTLBuffer,

	beta1,
	beta2,
	beta1powIterationLR,
	beta2powIteration float32,
) {
	b.Exclusive(func() {
		custom_kernel.CustomKernelUpdateWithAdam(
			b.device.CustomKernels, b.ID,
			dataBuffer.BufferID,
			gradBuffer.BufferID,
			mBuffer.BufferID,
			vBuffer.BufferID,
			beta1,
			beta2,
			beta1powIterationLR,
			beta2powIteration,
		)
	})
}

// not refactored part

func (b *MTLCommandBuffer) SoftmaxBufferTril(
	destinationBuffer *MTLBuffer,
	sourceBuffer *MTLBuffer,
	colsCount, rowsCount, offset int,
) {
	b.Exclusive(func() {
		custom_kernel.CustomKernelSoftmaxTrilFwdCreate(
			b.device.CustomKernels,
			b.ID,
			destinationBuffer.BufferID,
			sourceBuffer.BufferID,
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
	b.Exclusive(func() {
		custom_kernel.CustomKernelSoftmaxTrilBackward(
			b.device.CustomKernels,
			b.ID,
			destinationBuffer.BufferID,
			sourceBuffer.BufferID,
			softmaxBuffer.BufferID,
			colsCount,
			rowsCount,
			offset,
		)
	})
}

func (b *MTLCommandBuffer) CrossEntropyPos(
	dstBuffer *MTLBuffer,
	srcBuffer *MTLBuffer,
	smxBuffer *MTLBuffer,
	sumBuffer *MTLBuffer,
	tgtBuffer *MTLBuffer,
	chunkSize int,
) {
	b.Exclusive(func() {
		custom_kernel.CustomKernelCrossEntropyPos(
			b.device.CustomKernels,
			b.ID,
			dstBuffer.BufferID,
			srcBuffer.BufferID,
			smxBuffer.BufferID,
			sumBuffer.BufferID,
			tgtBuffer.BufferID,
			chunkSize,
		)
	})
}

func (b *MTLCommandBuffer) CrossEntropyPosBwd(
	oGrad *MTLBuffer,
	aGrad *MTLBuffer,
	tgtBuffer *MTLBuffer,
	smxBuffer *MTLBuffer,
	chunkSize int,
) {
	b.Exclusive(func() {
		custom_kernel.CustomKernelCrossEntropyPosBwd(
			b.device.CustomKernels,
			b.ID,
			oGrad.BufferID,
			aGrad.BufferID,
			tgtBuffer.BufferID,
			smxBuffer.BufferID,
			chunkSize,
		)
	})
}

func (b *MTLCommandBuffer) RMSNorm(
	dstBuffer *MTLBuffer,
	srcBuffer *MTLBuffer,
	sumBuffer *MTLBuffer,
	chunkSize int,
) {
	b.Exclusive(func() {
		custom_kernel.CustomKernelRMSNorm(
			b.device.CustomKernels,
			b.ID,
			dstBuffer.BufferID,
			srcBuffer.BufferID,
			sumBuffer.BufferID,
			chunkSize,
		)
	})
}

func (b *MTLCommandBuffer) MPSMatrixRandomMTGP32Encode(randomizer *MatrixRandomMTGP32, aM *MPSMatrix) {
	b.Exclusive(func() {
		framework.MPSMatrixRandomMTGP32Encode(randomizer.id, b.ID, aM.matrixID)
	})
}

func (b *MTLCommandBuffer) MatrixMultiplyWithKernel(kernelID unsafe.Pointer, aMatrix, bMatrix, cMatrix *MPSMatrix) {
	b.Exclusive(func() {
		framework.MPSMatrixMultiplicationEncode(
			b.ID,
			kernelID,
			aMatrix.matrixID,
			bMatrix.matrixID,
			cMatrix.matrixID,
		)
	})
}
