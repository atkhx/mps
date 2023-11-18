package operation

import "github.com/atkhx/mps"

func NewOpTriangularLowedSoftmax(device *mps.MTLDevice, inputData, inputGrad, outputData, outputGrad *mps.MTLBuffer, colsCount, rowsCount int) *OpTriangularLowedSoftmax {
	return &OpTriangularLowedSoftmax{
		device:     device,
		inputData:  inputData,
		inputGrad:  inputGrad,
		outputData: outputData,
		outputGrad: outputGrad,

		colsCount: colsCount,
		rowsCount: rowsCount,
	}
}

type OpTriangularLowedSoftmax struct {
	device *mps.MTLDevice

	inputData *mps.MTLBuffer
	inputGrad *mps.MTLBuffer

	outputData *mps.MTLBuffer
	outputGrad *mps.MTLBuffer

	colsCount int
	rowsCount int
}

func (op *OpTriangularLowedSoftmax) Forward(b *mps.MTLCommandBuffer) {
	WH := op.colsCount * op.rowsCount

	for offset := 0; offset < op.inputData.Length; offset += WH {
		b.SoftmaxBufferTril(
			op.outputData,
			op.inputData,
			op.colsCount,
			op.rowsCount,
			offset,
		)
	}
}

func (op *OpTriangularLowedSoftmax) Backward(b *mps.MTLCommandBuffer) {
	WH := op.colsCount * op.rowsCount

	for offset := 0; offset < op.inputData.Length; offset += WH {
		b.SoftmaxBufferTrilBwd(
			op.inputGrad,
			op.outputGrad,
			op.outputData,
			op.colsCount,
			op.rowsCount,
			offset,
		)
	}
}
