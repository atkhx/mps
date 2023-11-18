package operation

import (
	"github.com/atkhx/mps"
	"github.com/atkhx/mps/custom-kernel"
)

func NewOpEmbeddings(
	device *mps.MTLDevice,
	tokenEmbeddingData,
	tokenEmbeddingGrad,
	positionEmbeddingData,
	inputData,
	outputData,
	outputGrad *mps.MTLBuffer,
	featuresCount, contextLength int,
) *OpEmbeddings {
	return &OpEmbeddings{
		device:                device,
		tokenEmbeddingData:    tokenEmbeddingData,
		tokenEmbeddingGrad:    tokenEmbeddingGrad,
		positionEmbeddingData: positionEmbeddingData,
		inputData:             inputData,
		outputData:            outputData,
		outputGrad:            outputGrad,
		featuresCount:         featuresCount,
		contextLength:         contextLength,
	}
}

type OpEmbeddings struct {
	device *mps.MTLDevice

	tokenEmbeddingData *mps.MTLBuffer
	tokenEmbeddingGrad *mps.MTLBuffer

	positionEmbeddingData *mps.MTLBuffer

	inputData  *mps.MTLBuffer
	outputData *mps.MTLBuffer
	outputGrad *mps.MTLBuffer

	featuresCount int
	contextLength int
}

func (op *OpEmbeddings) Forward(b *mps.MTLCommandBuffer) {
	b.Exclusive(func() {
		custom_kernel.CustomKernelEmbeddings(
			op.device.CustomKernels,
			b.ID,
			op.inputData.BufferID,
			op.outputData.BufferID,
			op.positionEmbeddingData.BufferID,
			op.tokenEmbeddingData.BufferID,
			op.featuresCount,
			op.contextLength,
		)
	})
}

func (op *OpEmbeddings) Backward(b *mps.MTLCommandBuffer) {
	b.Exclusive(func() {
		custom_kernel.CustomKernelEmbeddingsBwd(
			op.device.CustomKernels,
			b.ID,
			op.inputData.BufferID,
			op.outputGrad.BufferID,
			op.tokenEmbeddingGrad.BufferID,
			op.featuresCount,
		)
	})
}
