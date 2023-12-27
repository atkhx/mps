package operation

import (
	"testing"

	"github.com/atkhx/mps"
	"github.com/stretchr/testify/require"
)

func TestOpConcatByRows(t *testing.T) {
	device := mps.NewMTLDevice()
	defer device.Release()

	inputWidth := 2
	inputHeight := 3

	inputData := []*mps.MTLBuffer{
		device.CreateBufferWithBytes([]float32{
			1, 2,
			3, 4,
			5, 6,
		}),
		device.CreateBufferWithBytes([]float32{
			11, 12,
			13, 14,
			15, 16,
		}),
	}

	expectedOutput := []float32{
		1, 2, 11, 12,
		3, 4, 13, 14,
		5, 6, 15, 16,
	}

	outputGradients := []float32{
		23, 24, 25, 26,
		33, 34, 35, 36,
		43, 44, 45, 46,
	}

	expectedInputGradients := [][]float32{
		{
			23, 24,
			33, 34,
			43, 44,
		},
		{
			25, 26,
			35, 36,
			45, 46,
		},
	}

	inputGrads := []*mps.MTLBuffer{
		device.CreateBufferWithLength(inputWidth * inputHeight),
		device.CreateBufferWithLength(inputWidth * inputHeight),
	}

	outputData := device.CreateBufferWithLength(inputWidth * inputHeight * len(inputData))
	outputGrads := device.CreateBufferWithBytes(outputGradients)

	b := device.CreateCommandQueue().GetCommandBuffer()
	operation := NewOpConcatByRows(device, inputData, inputGrads, outputData, outputGrads, inputWidth)
	operation.Forward(b)
	operation.Backward(b)
	b.Wait()

	require.Equal(t, expectedOutput, outputData.GetData())

	for inputDataIndex, expectedGradients := range expectedInputGradients {
		require.Equal(t, expectedGradients, inputGrads[inputDataIndex].GetData())
	}
}
