package operation

import (
	"github.com/atkhx/mps"
	"github.com/atkhx/mps/custom-kernel"
)

func NewOpMean(device *mps.MTLDevice, inputData, inputGrad, outputData, outputGrad *mps.MTLBuffer, chunkSize int) *OpMean {
	return &OpMean{
		device:     device,
		inputData:  inputData,
		inputGrad:  inputGrad,
		outputData: outputData,
		outputGrad: outputGrad,
		chunkSize:  chunkSize,
	}
}

type OpMean struct {
	device *mps.MTLDevice

	inputData *mps.MTLBuffer
	inputGrad *mps.MTLBuffer

	outputData *mps.MTLBuffer
	outputGrad *mps.MTLBuffer

	chunkSize int
}

func (op *OpMean) Forward(b *mps.MTLCommandBuffer) {
	b.Exclusive(func() {
		custom_kernel.CustomKernelMeanByRows(
			op.device.CustomKernels,
			b.ID,
			op.inputData.BufferID,
			op.outputData.BufferID,
			op.chunkSize,
		)
	})
}

func (op *OpMean) Backward(b *mps.MTLCommandBuffer) {
	b.Exclusive(func() {
		custom_kernel.CustomKernelMeanByRowsBwd(
			op.device.CustomKernels,
			b.ID,
			op.inputGrad.BufferID,
			op.outputGrad.BufferID,
			op.chunkSize,
		)
	})
}
