package operation

import (
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/atkhx/mps"
)

func TestOpRMSByRows(t *testing.T) {
	device := mps.NewMTLDevice()
	defer device.Release()

	rowsCount := 2
	chunkSize := 3

	inputData := device.CreateBufferWithBytes([]float32{
		1, 2, 3,
		4, 5, 6,
	})

	inputGrads := device.CreateBufferWithLength(chunkSize * rowsCount)

	outputData := device.CreateBufferWithLength(rowsCount)
	outputGrads := device.CreateBufferWithBytes([]float32{
		11,
		14,
	})

	b := device.CreateCommandQueue().GetCommandBuffer()
	operation := NewOpRMSByRows(device, inputData, inputGrads, outputData, outputGrads, chunkSize)
	operation.Forward(b)
	operation.Backward(b)
	b.Wait()

	require.InDeltaSlice(t, []float32{
		2.1602492,
		5.0662293,
	}, outputData.GetData(), 0.00001)

	require.InDeltaSlice(t, []float32{
		1.6973351, 3.3946702, 5.0920053,
		3.6845288, 4.605661, 5.526793,
	}, inputGrads.GetData(), 0.00001)
}
