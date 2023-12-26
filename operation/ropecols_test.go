package operation

import (
	"fmt"
	"testing"

	"github.com/atkhx/mps"
)

func TestNewOpRopeCols(t *testing.T) {
	device := mps.NewMTLDevice()
	defer device.Release()

	contextLength := 3
	featuresCount := 4
	batchSize := 2
	headSize := 2

	inputGrad := device.CreateBufferWithLength(contextLength * featuresCount * batchSize)
	inputData := device.CreateBufferWithBytes([]float32{
		1, 1, 1,
		1, 1, 1,
		1, 1, 1,
		1, 1, 1,

		1, 1, 1,
		1, 1, 1,
		1, 1, 1,
		1, 1, 1,

		//1, 2, 3,
		//4, 5, 6,
		//7, 8, 9,
		//10, 11, 12,
		//
		//13, 14, 15,
		//16, 17, 18,
		//19, 20, 21,
		//22, 23, 24,
	})

	outputData := device.CreateBufferWithLength(contextLength * featuresCount * batchSize)
	outputGrad := device.CreateBufferWithLength(contextLength * featuresCount * batchSize)

	op := NewOpRopeCols(
		device,
		inputData,
		inputGrad,
		outputData,
		outputGrad,
		featuresCount,
		headSize,
		contextLength,
	)

	commandQueue := device.CreateCommandQueue()
	defer commandQueue.Release()

	commandBuffer := commandQueue.GetCommandBuffer()
	defer commandBuffer.Release()

	op.Forward(commandBuffer)
	op.Backward(commandBuffer)
	commandBuffer.Wait()

	i := 0
	for b := 0; b < batchSize; b++ {
		for y := 0; y < featuresCount; y++ {
			for x := 0; x < contextLength; x++ {
				fmt.Print(outputData.GetData()[i], " ")
				i++
			}
			fmt.Println()
		}
		fmt.Println()
	}

	fmt.Println(outputData.GetData())
}
