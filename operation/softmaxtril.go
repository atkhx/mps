package operation

import (
	"unsafe"

	"github.com/atkhx/mps"
	"github.com/atkhx/mps/operation/softmaxtril"
)

func NewOpTriangularLowedSoftmax(device *mps.MTLDevice, inputData, inputGrad, outputData, outputGrad *mps.MTLBuffer, colsCount, rowsCount int) *OpTriangularLowedSoftmax {
	return &OpTriangularLowedSoftmax{
		kernel: softmaxtril.New(device.DeviceID),

		inputData:  inputData.BufferID,
		inputGrad:  inputGrad.BufferID,
		outputData: outputData.BufferID,
		outputGrad: outputGrad.BufferID,

		length:    inputData.Length,
		colsCount: colsCount,
		rowsCount: rowsCount,
	}
}

type OpTriangularLowedSoftmax struct {
	kernel *softmaxtril.Kernel

	inputData unsafe.Pointer
	inputGrad unsafe.Pointer

	outputData unsafe.Pointer
	outputGrad unsafe.Pointer

	length    int
	colsCount int
	rowsCount int
}

func (op *OpTriangularLowedSoftmax) Forward(b *mps.MTLCommandBuffer) {
	b.Exclusive(func() {
		op.kernel.Forward(
			b.ID,
			op.inputData,
			op.outputData,
			op.colsCount,
			op.rowsCount,
		)
	})
}

func (op *OpTriangularLowedSoftmax) Backward(b *mps.MTLCommandBuffer) {
	b.Exclusive(func() {
		op.kernel.Backward(
			b.ID,
			op.inputGrad,
			op.outputGrad,
			op.outputData,
			op.colsCount,
			op.rowsCount,
		)
	})
}
