package concatrows

import (
	"testing"

	"github.com/atkhx/mps/framework"
	"github.com/stretchr/testify/require"
)

func TestKernel_Forward(t *testing.T) {
	device := framework.MTLDeviceCreate()
	defer framework.MTLDeviceRelease(device)

	cmdQueue := framework.MTLCommandQueueCreate(device)
	defer framework.MTLCommandQueueRelease(cmdQueue)

	cmdBuffer := framework.MTLCommandBufferCreate(cmdQueue)
	defer framework.MTLCommandBufferRelease(cmdBuffer)

	inputData1 := framework.MTLBufferCreateWithBytes(device, []float32{
		1, 2,
		3, 4,
		5, 6,
	})

	inputData2 := framework.MTLBufferCreateWithBytes(device, []float32{
		11, 12,
		13, 14,
		15, 16,
	})

	inputWidth := 2
	outputWidth := 4

	outputData := framework.MTLBufferCreateWithLength(device, 12)

	kernel := New(device)
	kernel.Forward(cmdBuffer, inputData1, outputData, inputWidth, outputWidth, 0)
	kernel.Forward(cmdBuffer, inputData2, outputData, inputWidth, outputWidth, inputWidth)

	framework.MTLCommandBufferCommitAndWaitUntilCompleted(cmdBuffer)

	require.Equal(t, []float32{
		1, 2,
		3, 4,
		5, 6,
	}, framework.MTLBufferGetContentsFloats(inputData1, 6))

	require.Equal(t, []float32{
		11, 12,
		13, 14,
		15, 16,
	}, framework.MTLBufferGetContentsFloats(inputData2, 6))

	require.Equal(t, []float32{
		1, 2, 11, 12,
		3, 4, 13, 14,
		5, 6, 15, 16,
	}, framework.MTLBufferGetContentsFloats(outputData, 12))
}
