package operation

import (
	"fmt"
	"testing"

	"github.com/atkhx/mps"
	"github.com/atkhx/mps/framework"
	"github.com/atkhx/mps/operation/trilmask"
)

func TestSoftmaxTril(t *testing.T) {
	device := mps.NewMTLDevice()
	defer device.Release()

	w, h, d := 3, 3, 2

	inputData := device.CreateBufferWithBytes([]float32{
		1, 2, 3,
		3, 1, 2,
		2, 3, 1,

		1, 2, 3,
		2, 3, 1,
		3, 1, 2,
	})

	inputMatrixBatch := inputData.CreateMatrixBatch(w, h, d, w*h, 0)

	outputData := device.CreateBufferWithLength(w * h * d)
	outputDataBatch := outputData.CreateMatrixBatch(w, h, d, w*h, 0)

	maskKernel := trilmask.New(device.DeviceID)
	kernel := framework.MPSMatrixSoftMaxCreate(device.DeviceID)

	b := device.CreateCommandQueue().GetCommandBuffer()
	b.Exclusive(func() {
		maskKernel.Forward(
			b.ID,
			inputData.BufferID,
			inputData.BufferID,
			w, h,
		)
		framework.MPSMatrixSoftMaxEncode(
			b.ID,
			kernel,
			inputMatrixBatch.MatrixID,
			outputDataBatch.MatrixID,
		)
	})
	b.Wait()

	printMatrixBatch(w, h, d, inputData.GetData(), "input")
	printMatrixBatch(w, h, d, outputData.GetData(), "output")
}

func printMatrixBatch(w, h, d int, data []float32, label string) {
	if label != "" {
		fmt.Println(label)
	}

	i := 0
	for z := 0; z < d; z++ {
		for y := 0; y < h; y++ {
			for x := 0; x < w; x++ {
				fmt.Print(data[i], " ")
				i++
			}
			fmt.Println()
		}
		fmt.Println()
	}
}

// output
// 1 0 0
// 0.88079715 0.11920292 0
// 0.24472848 0.665241 0.090030566
//
// 1 0 0
// 0.2689414 0.7310586 0
// 0.665241 0.090030566 0.24472848
