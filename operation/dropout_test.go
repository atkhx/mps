package operation

import (
	"testing"

	"github.com/atkhx/mps"
	"github.com/stretchr/testify/require"
)

func TestOpDropout(t *testing.T) {
	device := mps.NewMTLDevice()
	defer device.Release()

	inputData := device.CreateBufferWithBytes([]float32{
		1, 2, 3,
		4, 5, 6,
		7, 8, 9,
	})

	inputGrads := device.CreateBufferWithLength(9)
	outputData := device.CreateBufferWithLength(9)

	outputGrads := device.CreateBufferWithBytes([]float32{
		9, 8, 7,
		6, 5, 4,
		3, 2, 1,
	})

	prob := float32(0.5)

	operation := NewOpDropout(device, inputData, inputGrads, outputData, outputGrads, prob)

	b := device.CreateCommandQueue().GetCommandBuffer()
	operation.Forward(b)
	operation.Backward(b)
	b.Wait()

	for i, x := range inputData.GetData() {
		p := operation.maskBuffer.GetData()[i]
		g := outputGrads.GetData()[i]

		if p > prob {
			require.Equal(t, x, outputData.GetData()[i])
			require.Equal(t, g, inputGrads.GetData()[i])
		} else {
			require.Equal(t, float32(0.0), outputData.GetData()[i])
			require.Equal(t, float32(0.0), inputGrads.GetData()[i])
		}
	}
}
