package addequal

import (
	"testing"

	"github.com/atkhx/mps/framework"
	"github.com/stretchr/testify/require"
)

func TestKernel(t *testing.T) {
	device := framework.MTLDeviceCreate()
	defer framework.MTLDeviceRelease(device)

	cmdQueue := framework.MTLCommandQueueCreate(device)
	defer framework.MTLCommandQueueRelease(cmdQueue)

	cmdBuffer := framework.MTLCommandBufferCreate(cmdQueue)
	defer framework.MTLCommandBufferRelease(cmdBuffer)

	inputGrad := framework.MTLBufferCreateWithLength(device, 6)
	inputData := framework.MTLBufferCreateWithBytes(device, []float32{
		1, 2, 3,
		4, 5, 6,
	})

	weightsGrad := framework.MTLBufferCreateWithLength(device, 6)
	weightsData := framework.MTLBufferCreateWithBytes(device, []float32{
		2, 3, 4,
		5, 6, 7,
	})

	outputData := framework.MTLBufferCreateWithLength(device, 6)
	outputGrad := framework.MTLBufferCreateWithBytes(device, []float32{1, 1, 1, 1, 1, 1})

	kernel := New(device)
	kernel.Forward(cmdBuffer, inputData, weightsData, outputData)
	kernel.Backward(cmdBuffer, inputGrad, weightsGrad, outputGrad)
	framework.MTLCommandBufferCommitAndWaitUntilCompleted(cmdBuffer)

	require.Equal(t, []float32{
		3, 5, 7,
		9, 11, 13,
	}, framework.MTLBufferGetContentsFloats(outputData, 6))

	require.Equal(t, []float32{1, 1, 1, 1, 1, 1}, framework.MTLBufferGetContentsFloats(inputGrad, 6))

	require.Equal(t, []float32{
		1, 1, 1,
		1, 1, 1,
	}, framework.MTLBufferGetContentsFloats(weightsGrad, 6))
}
