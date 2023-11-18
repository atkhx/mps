package operation

import (
	"github.com/atkhx/mps"
	"github.com/atkhx/mps/custom-kernel"
)

func NewOpConcatByRows(device *mps.MTLDevice, inputData, inputGrad []*mps.MTLBuffer, outputData, outputGrad *mps.MTLBuffer, inputWidth int) *OpConcatByRows {
	return &OpConcatByRows{
		device:     device,
		inputData:  inputData,
		inputGrad:  inputGrad,
		outputData: outputData,
		outputGrad: outputGrad,
		inputWidth: inputWidth,
	}
}

type OpConcatByRows struct {
	device *mps.MTLDevice

	inputData []*mps.MTLBuffer
	inputGrad []*mps.MTLBuffer

	outputData *mps.MTLBuffer
	outputGrad *mps.MTLBuffer

	inputWidth int
}

func (op *OpConcatByRows) Forward(b *mps.MTLCommandBuffer) {
	inputWidth := op.inputWidth
	outputWidth := inputWidth * len(op.inputData)

	for i, inputData := range op.inputData {
		b.Exclusive(func() {
			custom_kernel.CustomKernelConcatByRows(
				op.device.CustomKernels,
				b.ID,
				inputData.BufferID,
				op.outputData.BufferID,
				inputWidth,
				outputWidth,
				i*inputWidth, // outputData offset
			)
		})
	}
}

func (op *OpConcatByRows) Backward(b *mps.MTLCommandBuffer) {
	inputWidth := op.inputWidth
	outputWidth := inputWidth * len(op.inputData)

	for i, inputGrad := range op.inputGrad {
		b.Exclusive(func() {
			custom_kernel.CustomKernelConcatByRowsBwd(
				op.device.CustomKernels,
				b.ID,
				inputGrad.BufferID,
				op.outputGrad.BufferID,
				inputWidth,
				outputWidth,
				i*inputWidth, // outputData offset
			)
		})
	}
}
