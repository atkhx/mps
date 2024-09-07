package custom_kernel

import (
	"testing"

	"github.com/atkhx/mps/framework"
	"github.com/stretchr/testify/require"
)

func TestKernel_Fill(t *testing.T) {
	// Create metal device
	deviceID := framework.MTLDeviceCreate()
	defer framework.MTLDeviceRelease(deviceID)

	// Create command queue
	commandQueueID := framework.MTLCommandQueueCreate(deviceID)
	defer framework.MTLCommandQueueRelease(commandQueueID)

	// Create command buffer
	commandBufferID := framework.MTLCommandBufferCreate(commandQueueID)
	defer framework.MTLCommandBufferRelease(commandBufferID)

	// Create data buffer with length 10
	bufferID := framework.MTLBufferCreateWithLength(deviceID, 10)
	defer framework.MTLBufferRelease(bufferID)

	// Create custom kernel object
	kernel := New(deviceID)
	defer kernel.Release()

	// Schedule operation: fill data buffer[2:6] with the constant value '77'
	kernel.Fill(commandBufferID, bufferID, 77, 2, 6)
	// Schedule operation: fill data buffer[0:3] with the constant value '11'
	kernel.Fill(commandBufferID, bufferID, 11, 0, 3)
	// Schedule operation: fill data buffer[9:10] with the constant value '88'
	kernel.Fill(commandBufferID, bufferID, 88, 9, 1)

	// Commit scheduled commands and wait calculations completed
	framework.MTLCommandBufferCommitAndWaitUntilCompleted(commandBufferID)

	// Assert result
	content := framework.MTLBufferGetContentsFloats(bufferID, 10)
	require.Equal(t, []float32{11, 11, 11, 77, 77, 77, 77, 77, 0, 88}, content)
}
