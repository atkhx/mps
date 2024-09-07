package operation

import (
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/atkhx/mps"
)

func TestOpRMSNormByRows2(t *testing.T) {
	device := mps.NewMTLDevice()
	defer device.Release()

	rowsCount := 2
	chunkSize := 3
	inputData := device.CreateBufferWithBytes([]float32{
		1, 2, 3,
		4, 5, 6,
	})
	inputGrads := device.CreateBufferWithLength(chunkSize * rowsCount)

	outputData := device.CreateBufferWithLength(chunkSize * rowsCount)
	outputGrads := device.CreateBufferWithBytes([]float32{
		11, 12, 13,
		14, 15, 16,
	})

	b := device.CreateCommandQueue().GetCommandBuffer()
	operation := NewOpRMSNormByRows2(device, inputData, inputGrads, outputData, outputGrads, chunkSize)
	operation.Forward(b)
	operation.Backward(b)
	b.Wait()

	require.InDeltaSlice(t, []float32{
		0.46290955, 0.9258191, 1.3887286,
		0.78954184, 0.9869273, 1.1843128,
	}, outputData.GetData(), 0.00001)

	require.InDeltaSlice(t, []float32{
		2.6452026, 0.6613091, -1.3225837,
		0.4357872, 0.051270146, -0.33324647,
	}, inputGrads.GetData(), 0.00001)
}
