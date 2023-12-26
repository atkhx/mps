package nllpos

import (
	"fmt"
	"testing"

	"github.com/atkhx/mps/framework"
)

func TestKernel_Forward(t *testing.T) {
	device := framework.MTLDeviceCreate()
	defer framework.MTLDeviceRelease(device)

	cmdQueue := framework.MTLCommandQueueCreate(device)
	defer framework.MTLCommandQueueRelease(cmdQueue)

	cmdBuffer := framework.MTLCommandBufferCreate(cmdQueue)
	defer framework.MTLCommandBufferRelease(cmdBuffer)

	kernel := New(device)
	fmt.Println(kernel)
}
