package rmsnorm

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

	rowsCount := 2
	chunkSize := 3
	inputData := framework.MTLBufferCreateWithBytes(device, []float32{
		1, 2, 3,
		4, 5, 6,
	})
	outputData := framework.MTLBufferCreateWithLength(device, chunkSize*rowsCount)
	aggData := framework.MTLBufferCreateWithLength(device, rowsCount)

	kernel := New(device)
	kernel.Forward(cmdBuffer, inputData, outputData, aggData, chunkSize)

	framework.MTLCommandBufferCommitAndWaitUntilCompleted(cmdBuffer)

	require.Equal(t, []float32{
		1, 2, 3,
		4, 5, 6,
	}, framework.MTLBufferGetContentsFloats(inputData, chunkSize*rowsCount))

	require.Equal(t, []float32{
		2.1602492, // ~ sqrt(eps + (1 + 4 + 9) / 3)
		5.0662293, // ~ sqrt(eps + (16 + 25 + 36) / 3)
	}, framework.MTLBufferGetContentsFloats(aggData, rowsCount))

	require.Equal(t, []float32{
		// ~ 1/2.1602492, 2/2.1602492, 3/2.1602492,
		// ~ 4/5.0662293, 5/5.0662293, 6/5.0662293,
		0.46290955, 0.9258191, 1.3887286,
		0.78954184, 0.9869273, 1.1843128,
	}, framework.MTLBufferGetContentsFloats(outputData, chunkSize*rowsCount))
}
