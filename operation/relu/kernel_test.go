package relu

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

	inputData := framework.MTLBufferCreateWithBytes(device, []float32{
		+0.1, 0,
		-0.1, 1,
	})

	outputData := framework.MTLBufferCreateWithLength(device, 4)

	kernel := New(device)
	kernel.Forward(cmdBuffer, inputData, outputData)
	framework.MTLCommandBufferCommitAndWaitUntilCompleted(cmdBuffer)

	require.Equal(t, []float32{
		+0.1, 0,
		-0.1, 1,
	}, framework.MTLBufferGetContentsFloats(inputData, 4))

	require.Equal(t, []float32{
		0.1, 0,
		0, 1,
	}, framework.MTLBufferGetContentsFloats(outputData, 4))
}
