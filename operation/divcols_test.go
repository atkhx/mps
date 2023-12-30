package operation

import (
	"testing"

	"github.com/atkhx/mps"
	"github.com/stretchr/testify/require"
)

func TestOpDivCols(t *testing.T) {
	device := mps.NewMTLDevice()
	defer device.Release()

	rowWidth := 2
	colHeight := 3
	batchSize := 2

	inputData := device.CreateBufferWithBytes([]float32{
		10, 20,
		30, 40,
		50, 60,

		100, 200,
		300, 400,
		500, 600,
	})

	inputGrad := device.CreateBufferWithLength(rowWidth * colHeight * batchSize)
	weightsData := device.CreateBufferWithBytes([]float32{
		5,
		10,
		5,
	})
	weightsGrad := device.CreateBufferWithLength(colHeight)

	outputData := device.CreateBufferWithLength(rowWidth * colHeight * batchSize)
	outputGrad := device.CreateBufferWithBytes([]float32{
		1, 2,
		3, 4,
		5, 6,

		7, 8,
		9, 10,
		11, 12,
	})

	operation := NewOpDivCols(device, inputData, inputGrad, weightsData, weightsGrad, outputData, outputGrad, rowWidth, colHeight)

	b := device.CreateCommandQueue().GetCommandBuffer()
	operation.Forward(b)
	operation.Backward(b)
	b.Wait()

	require.Equal(t, []float32{
		2, 4,
		3, 4,
		10, 12,

		20, 40,
		30, 40,
		100, 120,
	}, outputData.GetData())

	require.Equal(t, []float32{
		1. / 5, 2. / 5,
		3. / 10, 4. / 10,
		5. / 5, 6. / 5,

		7. / 5, 8. / 5,
		0.90000004, 10. / 10,
		11. / 5, 12. / 5,
	}, inputGrad.GetData())

	require.Equal(t, []float32{
		-94,    // - (1 * 10/25  +  2*20/25  + 7*100/25  +  8*200/25)
		-69.5,  // - (3 * 30/100 +  4*40/100 + 9*300/100 + 10*400/100)
		-532.4, // - (5 * 50/25  +  6*60/25  + 11*500/25 + 12*600/25)
	}, weightsGrad.GetData())
}
