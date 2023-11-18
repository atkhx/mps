package operation

import "github.com/atkhx/mps"

func NewOpDropout(device *mps.MTLDevice, randomizer *mps.MatrixRandomMTGP32, inputData, inputGrad, outputData, outputGrad *mps.MTLBuffer, probability float32) *OpDropout {
	maskBuffer := device.CreateBufferWithLength(inputData.Length)
	maskMatrix := maskBuffer.CreateMatrix(inputData.Length, 1, 0)

	return &OpDropout{
		device:     device,
		randomizer: randomizer,

		maskBuffer: maskBuffer,
		maskMatrix: maskMatrix,

		inputData: inputData,
		inputGrad: inputGrad,

		outputData: outputData,
		outputGrad: outputGrad,

		probability: probability,
	}
}

type OpDropout struct {
	device *mps.MTLDevice

	randomizer *mps.MatrixRandomMTGP32

	maskBuffer *mps.MTLBuffer
	maskMatrix *mps.MPSMatrix

	inputData *mps.MTLBuffer
	inputGrad *mps.MTLBuffer

	outputData *mps.MTLBuffer
	outputGrad *mps.MTLBuffer

	probability float32
}

func (op *OpDropout) Forward(b *mps.MTLCommandBuffer) {
	b.MPSMatrixRandomMTGP32Encode(op.randomizer, op.maskMatrix)
	b.DropoutBuffer(op.outputData, op.inputData, op.maskBuffer, op.probability)
}

func (op *OpDropout) Backward(b *mps.MTLCommandBuffer) {
	b.DropoutBwdBuffer(op.inputGrad, op.outputGrad, op.maskBuffer, op.probability)
}
