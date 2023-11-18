package operation

import "C"
import (
	"github.com/atkhx/mps"
	"github.com/atkhx/mps/custom-kernel"
)

func NewOpNormRMS(device *mps.MTLDevice, inputData, inputGrad, outputData, outputGrad *mps.MTLBuffer, chunkSize int) *OpNormRMS {
	aggBuff := device.CreateBufferWithLength(inputData.Length / chunkSize)
	aggGrad := device.CreateBufferWithLength(inputData.Length / chunkSize)

	return &OpNormRMS{
		device:     device,
		aggData:    aggBuff,
		aggGrad:    aggGrad,
		inputData:  inputData,
		inputGrad:  inputGrad,
		outputData: outputData,
		outputGrad: outputGrad,
		chunkSize:  chunkSize,
	}
}

type OpNormRMS struct {
	device *mps.MTLDevice

	inputData *mps.MTLBuffer
	inputGrad *mps.MTLBuffer

	outputData *mps.MTLBuffer
	outputGrad *mps.MTLBuffer

	aggData *mps.MTLBuffer
	aggGrad *mps.MTLBuffer

	chunkSize int
}

func (op *OpNormRMS) Forward(b *mps.MTLCommandBuffer) {
	b.RMSNorm(op.outputData, op.inputData, op.aggData, op.chunkSize)
}

func (op *OpNormRMS) Backward(b *mps.MTLCommandBuffer) {
	b.Exclusive(func() {
		custom_kernel.CustomKernelRMSNormBwd(
			op.device.CustomKernels,
			b.ID,
			op.inputData.BufferID,
			op.inputGrad.BufferID,
			op.outputData.BufferID,
			op.outputGrad.BufferID,
			op.aggData.BufferID,
			op.aggGrad.BufferID,
			op.chunkSize,
		)
	})
}
