package operation

import "github.com/atkhx/mps"

func NewOpMatrixMultiply(
	device *mps.MTLDevice,

	aDataBuffer *mps.MTLBuffer,
	aGradBuffer *mps.MTLBuffer,
	bDataBuffer *mps.MTLBuffer,

	bGradBuffer *mps.MTLBuffer,
	cDataBuffer *mps.MTLBuffer,
	cGradBuffer *mps.MTLBuffer,

	aWidth, aHeight, aDepth int,
	bWidth, bHeight, bDepth int,
	cWidth, cHeight, cDepth int,

	alpha float32,
) *OpMatrixMultiply {
	operation := &OpMatrixMultiply{}

	switch {
	case aDepth == bDepth:
		batchSize := aDepth

		aDataM := aDataBuffer.CreateMatrixBatch(aWidth, aHeight, batchSize, aWidth*aHeight, 0)
		aGradM := aGradBuffer.CreateMatrixBatch(aWidth, aHeight, batchSize, aWidth*aHeight, 0)
		bDataM := bDataBuffer.CreateMatrixBatch(bWidth, bHeight, batchSize, bWidth*bHeight, 0)
		bGradM := bGradBuffer.CreateMatrixBatch(bWidth, bHeight, batchSize, bWidth*bHeight, 0)
		cDataM := cDataBuffer.CreateMatrixBatch(cWidth, cHeight, batchSize, cWidth*cHeight, 0)
		cGradM := cGradBuffer.CreateMatrixBatch(cWidth, cHeight, batchSize, cWidth*cHeight, 0)

		calcCDataKernelID := device.CreateMatrixMultiplyKernel(aHeight, bWidth, aWidth, alpha, 0.0, false, false)
		calcAGradKernelID := device.CreateMatrixMultiplyKernel(aHeight, aWidth, cWidth, alpha, 1.0, false, true)
		calcBGradKernelID := device.CreateMatrixMultiplyKernel(bHeight, bWidth, aHeight, alpha, 1.0, true, false)

		operation.forward = func(b *mps.MTLCommandBuffer) {
			b.MatrixMultiplyWithKernel(calcCDataKernelID, aDataM, bDataM, cDataM)
		}

		operation.backward = func(b *mps.MTLCommandBuffer) {
			b.MatrixMultiplyWithKernel(calcAGradKernelID, cGradM, bDataM, aGradM)
			b.MatrixMultiplyWithKernel(calcBGradKernelID, aDataM, cGradM, bGradM)
		}
	case bDepth == 1:
		batchSize := aDepth

		aDataM := aDataBuffer.CreateMatrixBatch(aWidth, aHeight, batchSize, aWidth*aHeight, 0)
		aGradM := aGradBuffer.CreateMatrixBatch(aWidth, aHeight, batchSize, aWidth*aHeight, 0)
		bDataM := bDataBuffer.CreateMatrixBatch(bWidth, bHeight, batchSize, 0, 0)
		bGradM := bGradBuffer.CreateMatrixBatch(bWidth, bHeight, 1, 0, 0)
		cDataM := cDataBuffer.CreateMatrixBatch(cWidth, cHeight, batchSize, cWidth*cHeight, 0)
		cGradM := cGradBuffer.CreateMatrixBatch(cWidth, cHeight, batchSize, cWidth*cHeight, 0)

		aDataMBig := aDataBuffer.CreateMatrix(aWidth, aHeight*aDepth, 0)
		cGradMBig := cGradBuffer.CreateMatrix(cWidth, cHeight*cDepth, 0)

		calcCDataKernelID := device.CreateMatrixMultiplyKernel(aHeight, bWidth, aWidth, alpha, 0.0, false, false)
		calcAGradKernelID := device.CreateMatrixMultiplyKernel(aHeight, aWidth, cWidth, alpha, 1.0, false, true)
		calcBGradKernelID := device.CreateMatrixMultiplyKernel(bHeight, bWidth, aHeight*aDepth, alpha, 1.0, true, false)

		operation.forward = func(b *mps.MTLCommandBuffer) {
			b.MatrixMultiplyWithKernel(calcCDataKernelID, aDataM, bDataM, cDataM)
		}

		operation.backward = func(b *mps.MTLCommandBuffer) {
			b.MatrixMultiplyWithKernel(calcAGradKernelID, cGradM, bDataM, aGradM)
			b.MatrixMultiplyWithKernel(calcBGradKernelID, aDataMBig, cGradMBig, bGradM)
		}
	default:
		panic("not implemented case aDepth == 1")
	}

	return operation
}

type OpMatrixMultiply struct {
	forward  func(b *mps.MTLCommandBuffer)
	backward func(b *mps.MTLCommandBuffer)
}

func (op *OpMatrixMultiply) Forward(b *mps.MTLCommandBuffer) {
	op.forward(b)
}

func (op *OpMatrixMultiply) Backward(b *mps.MTLCommandBuffer) {
	op.backward(b)
}
