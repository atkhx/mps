package operation

import (
	"testing"

	"github.com/atkhx/mps"
	"github.com/stretchr/testify/require"
)

func TestOpReLu(t *testing.T) {
	device := mps.NewMTLDevice()
	defer device.Release()

	inputValues := []float32{0.0, -0.5, 0.7, -1, 2}
	expectedResult := []float32{0.0, 0.0, 0.7, 0.0, 2}

	cGradsValues := []float32{3, 4, 5, 6, 7}
	expectedInputGrads := []float32{0.0, 0.0, 5, 0.0, 7}

	inputData := device.CreateBufferWithBytes(inputValues)
	inputGrads := device.CreateBufferWithLength(len(inputValues))

	outputData := device.CreateBufferWithLength(len(inputValues))
	outputGrads := device.CreateBufferWithBytes(cGradsValues)

	b := device.CreateCommandQueue().GetCommandBuffer()
	operation := NewOpReLu(device, inputData, inputGrads, outputData, outputGrads)
	operation.Forward(b)
	operation.Backward(b)
	b.Wait()

	require.Equal(t, expectedResult, outputData.GetData())
	require.Equal(t, expectedInputGrads, inputGrads.GetData())
}
