package operation

import (
	"github.com/atkhx/mps"
)

func NewOpTriangularLowedSoftmax2(device *mps.MTLDevice, inputData, inputGrad, outputData, outputGrad *mps.MTLBuffer, colsCount, rowsCount int) *OpTriangularLowedSoftmax2 {
	maskedData := device.CreateBufferWithLength(inputData.Length)
	//maskedGrad := device.CreateBufferWithLength(inputData.Length)

	return &OpTriangularLowedSoftmax2{
		//trilMaskOp: NewOpTrilMask(device, inputData, inputGrad, maskedData, maskedGrad, colsCount, rowsCount),
		//softmaxOp:  NewOpSoftmax(device, maskedData, maskedGrad, outputData, outputGrad, colsCount),
		trilMaskOp: NewOpTrilMask(device, inputData, inputGrad, maskedData, inputGrad, colsCount, rowsCount),
		softmaxOp:  NewOpSoftmax(device, maskedData, inputGrad, outputData, outputGrad, colsCount),
	}
}

type OpTriangularLowedSoftmax2 struct {
	trilMaskOp *OpTrilMask
	softmaxOp  *OpSoftmax
}

func (op *OpTriangularLowedSoftmax2) Forward(b *mps.MTLCommandBuffer) {
	op.trilMaskOp.Forward(b)
	op.softmaxOp.Forward(b)
}

func (op *OpTriangularLowedSoftmax2) Backward(b *mps.MTLCommandBuffer) {
	op.softmaxOp.Backward(b)
	op.trilMaskOp.Backward(b)
}
