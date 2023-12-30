package operation

import (
	"github.com/atkhx/mps"
)

func NewOpTriangularLowedSoftmax3(device *mps.MTLDevice, inputData, inputGrad, outputData, outputGrad *mps.MTLBuffer, colsCount, rowsCount int) *OpTriangularLowedSoftmax3 {
	maskedData := device.CreateBufferWithLength(inputData.Length)

	return &OpTriangularLowedSoftmax3{
		trilMaskOp: NewOpTrilMask2(device, inputData, inputGrad, maskedData, colsCount, rowsCount),
		softmaxOp:  NewOpSoftmax(device, maskedData, inputGrad, outputData, outputGrad, colsCount),
	}
}

type OpTriangularLowedSoftmax3 struct {
	trilMaskOp *OpTrilMask2
	softmaxOp  *OpSoftmax
}

func (op *OpTriangularLowedSoftmax3) Forward(b *mps.MTLCommandBuffer) {
	op.trilMaskOp.Forward(b)
	op.softmaxOp.Forward(b)
}

func (op *OpTriangularLowedSoftmax3) Backward(b *mps.MTLCommandBuffer) {
	op.softmaxOp.Backward(b)
	op.trilMaskOp.Backward(b)
}
