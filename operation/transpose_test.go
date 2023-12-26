package operation

import (
	"testing"

	"github.com/atkhx/mps"
	"github.com/stretchr/testify/require"
)

func TestOpTranspose(t *testing.T) {
	device := mps.NewMTLDevice()
	defer device.Release()

	width, height := 3, 2

	inputData := []float32{
		1, 2, 3,
		4, 5, 6,

		7, 8, 9,
		10, 11, 12,
	}

	expectedOutputData := []float32{
		1, 4,
		2, 5,
		3, 6,

		7, 10,
		8, 11,
		9, 12,
	}

	outputGrads := []float32{
		11, 14,
		12, 15,
		13, 16,

		17, 20,
		18, 21,
		19, 22,
	}

	expectedInputGrads := []float32{
		11, 12, 13,
		14, 15, 16,

		17, 18, 19,
		20, 21, 22,
	}

	aData := device.CreateBufferWithBytes(inputData)
	bData := device.CreateBufferWithLength(len(inputData))
	aGrads := device.CreateBufferWithLength(len(inputData))
	bGrads := device.CreateBufferWithBytes(outputGrads)

	b := device.CreateCommandQueue().GetCommandBuffer()
	operation := NewOpTranspose(device, aData, aGrads, bData, bGrads, width, height)
	operation.Forward(b)
	operation.Backward(b)
	b.Wait()

	require.Equal(t, expectedOutputData, bData.GetData())
	require.Equal(t, expectedInputGrads, aGrads.GetData())
}
