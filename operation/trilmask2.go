package operation

import (
	"unsafe"

	"github.com/atkhx/mps"
	"github.com/atkhx/mps/operation/trilmask2"
)

func NewOpTrilMask2(device *mps.MTLDevice, inputData, inputGrad, outputData *mps.MTLBuffer, colsCount, rowsCount int) *OpTrilMask2 {
	return &OpTrilMask2{
		kernel: trilmask2.New(device.DeviceID),

		inputData:  inputData.BufferID,
		inputGrad:  inputGrad.BufferID,
		outputData: outputData.BufferID,

		length:    inputData.Length,
		colsCount: colsCount,
		rowsCount: rowsCount,
	}
}

type OpTrilMask2 struct {
	kernel *trilmask2.Kernel

	inputData unsafe.Pointer
	inputGrad unsafe.Pointer

	outputData unsafe.Pointer

	length    int
	colsCount int
	rowsCount int
}

func (op *OpTrilMask2) Forward(b *mps.MTLCommandBuffer) {
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

func (op *OpTrilMask2) Backward(b *mps.MTLCommandBuffer) {
	b.Exclusive(func() {
		op.kernel.Backward(
			b.ID,
			op.inputGrad,
			op.colsCount,
			op.rowsCount,
		)
	})
}
