package custom_kernel

/*
#cgo CFLAGS: -x objective-c
#cgo LDFLAGS: -framework Metal -framework MetalPerformanceShaders -framework CoreGraphics -framework Foundation
#include "framework.h"
*/
import "C"
import (
	_ "embed"
	"unsafe"
)

//go:embed kernel.metal
var customKernelFunctions string

func CustomKernelCreate(deviceID unsafe.Pointer) unsafe.Pointer {
	cKernelString := C.CString(customKernelFunctions)
	defer C.free(unsafe.Pointer(cKernelString))
	return C.customKernelCreate(deviceID, cKernelString)
}

func CustomKernelCopy(kernelID, commandBufferID, dstBufferID, srcBufferID unsafe.Pointer, dstOffset, srcOffset, length int) {
	C.customKernelCopy(kernelID, commandBufferID, dstBufferID, srcBufferID, C.uint(dstOffset*4), C.uint(srcOffset*4), C.uint(length*4))
}

func CustomKernelCopyWHD(kernelID, commandBufferID, dstBufferID, srcBufferID unsafe.Pointer, W, H, D int) {
	C.customKernelCopyWHD(kernelID, commandBufferID, dstBufferID, srcBufferID, C.uint(W), C.uint(H), C.uint(D))
}

func CustomKernelFill(kernelID, commandBufferID, bufferID unsafe.Pointer, value float32, offset, length int) {
	C.customKernelFill(kernelID, commandBufferID, bufferID, C.float(value), C.uint(offset*4), C.uint(length*4))
}

func CustomKernelAdd(kernelID, commandBufferID, dstBufferID, srcBufferID unsafe.Pointer, dstOffset, srcOffset, length int) {
	C.customKernelAdd(kernelID, commandBufferID, dstBufferID, srcBufferID, C.uint(dstOffset*4), C.uint(srcOffset*4), C.uint(length*4))
}

func CustomKernelAddTo(kernelID, commandBufferID, dstBufferID, aBuffer, bBuffer unsafe.Pointer) {
	C.customKernelAddTo(kernelID, commandBufferID, dstBufferID, aBuffer, bBuffer)
}

func CustomKernelAddToWHD(kernelID, commandBufferID, dstBufferID, aBuffer, bBuffer unsafe.Pointer, K float32, W, H, D int) {
	C.customKernelAddToWHD(kernelID, commandBufferID, dstBufferID, aBuffer, bBuffer, C.float(K), C.uint(W), C.uint(H), C.uint(D))
}

func CustomKernelAddToWHDBwd(kernelID, commandBufferID, aGrad, bGrad, oGrad unsafe.Pointer, W, H, D int) {
	C.customKernelAddToWHDBwd(kernelID, commandBufferID, aGrad, bGrad, oGrad, C.uint(W), C.uint(H), C.uint(D))
}

func CustomKernelAddScalar(kernelID, commandBufferID, dstBufferID unsafe.Pointer, value float32) {
	C.customKernelAddScalar(kernelID, commandBufferID, dstBufferID, C.float(value))
}

func CustomKernelMul(kernelID, commandBufferID, dstBufferID, srcBufferID unsafe.Pointer, dstOffset, srcOffset, length int) {
	C.customKernelMul(kernelID, commandBufferID, dstBufferID, srcBufferID, C.uint(dstOffset*4), C.uint(srcOffset*4), C.uint(length*4))
}

func CustomKernelReLU(kernelID, commandBufferID, dstBufferID, srcBufferID unsafe.Pointer) {
	C.customKernelReLU(kernelID, commandBufferID, dstBufferID, srcBufferID)
}

func CustomKernelReLUBackward(kernelID, commandBufferID, dstBufferID, srcBufferID, maskBufferID unsafe.Pointer) {
	C.customKernelReLUBwd(kernelID, commandBufferID, dstBufferID, srcBufferID, maskBufferID)
}

func CustomKernelSoftmaxForward(
	kernelID,
	commandBufferID,
	dstBufferID,
	srcBufferID,
	sumOutBufferID unsafe.Pointer,
	colsCount, rowsCount, offset int,
) {
	C.customKernelSoftmax(
		kernelID,
		commandBufferID,
		dstBufferID,
		srcBufferID,
		sumOutBufferID,
		C.uint(colsCount),
		C.uint(rowsCount),
		C.uint(offset*4),
	)
}

func CustomKernelDropout(
	kernelID,
	commandBufferID,
	dstBufferID,
	srcBufferID,
	mskBufferID unsafe.Pointer,
	probability float32,
) {
	C.customKernelDropout(
		kernelID,
		commandBufferID,
		dstBufferID,
		srcBufferID,
		mskBufferID,
		C.float(probability),
	)
}

func CustomKernelDropoutBwd(
	kernelID,
	commandBufferID,
	dstBufferID,
	srcBufferID,
	mskBufferID unsafe.Pointer,
	probability float32,
) {
	C.customKernelDropoutBwd(
		kernelID,
		commandBufferID,
		dstBufferID,
		srcBufferID,
		mskBufferID,
		C.float(probability),
	)
}

func CustomKernelUpdateWithAdam(
	kernelID,
	commandBufferID,

	dataBufferID,
	gradBufferID,
	mBufferID,
	vBufferID unsafe.Pointer,

	beta1,
	beta2,
	beta1powIterationLR,
	beta2powIteration float32,
) {
	C.customKernelUpdateWithAdam(
		kernelID,
		commandBufferID,

		dataBufferID,
		gradBufferID,
		mBufferID,
		vBufferID,

		C.float(beta1),
		C.float(beta2),
		C.float(beta1powIterationLR),
		C.float(beta2powIteration),
	)
}

func CustomKernelSoftmaxTrilFwdCreate(
	kernelID,
	commandBufferID,
	dstBufferID,
	srcBufferID unsafe.Pointer,

	colsCount,
	rowsCount,
	offset int,
) {
	C.customKernelSoftmaxTrilFwd(
		kernelID,
		commandBufferID,
		dstBufferID,
		srcBufferID,
		C.uint(colsCount),
		C.uint(rowsCount),
		C.uint(offset*4),
	)
}

func CustomKernelSoftmaxTrilBackward(
	kernelID,
	commandBufferID,
	dstBufferID,
	srcBufferID,
	softmaxBufferID unsafe.Pointer,
	colsCount, rowsCount, offset int,
) {
	C.customKernelSoftmaxTrilBwd(
		kernelID,
		commandBufferID,
		dstBufferID,
		srcBufferID,
		softmaxBufferID,
		C.uint(colsCount),
		C.uint(rowsCount),
		C.uint(offset*4),
	)
}

func CustomKernelCrossEntropyPos(
	kernelID,
	commandBufferID,
	dstBufferID,
	srcBufferID,
	smxBufferID,
	sumBufferID,
	tgtBufferID unsafe.Pointer,
	chunkSize int,
) {
	C.customKernelCrossEntropyPos(
		kernelID,
		commandBufferID,
		dstBufferID,
		srcBufferID,
		smxBufferID,
		sumBufferID,
		tgtBufferID,
		C.uint(chunkSize),
	)
}

func CustomKernelCrossEntropyPosBwd(
	kernelID,
	commandBufferID,
	oGrad,
	aGrad,
	tgtBufferID,
	smxBufferID unsafe.Pointer,
	chunkSize int,
) {
	C.customKernelCrossEntropyPosBwd(
		kernelID,
		commandBufferID,
		oGrad,
		aGrad,
		tgtBufferID,
		smxBufferID,
		C.uint(chunkSize),
	)
}

func CustomKernelRMSNorm(
	kernelID,
	commandBufferID,
	dstBufferID,
	srcBufferID,
	sumBufferID unsafe.Pointer,
	chunkSize int,
) {
	C.customKernelRMSNorm(
		kernelID,
		commandBufferID,
		dstBufferID,
		srcBufferID,
		sumBufferID,
		C.uint(chunkSize),
	)
}

func CustomKernelRMSNormBwd(
	kernelID,
	commandBufferID,
	inputDataID,
	inputGradID,
	outputDataID,
	outputGradID,
	aggDataID,
	aggGradID unsafe.Pointer,
	chunkSize int,
) {
	C.customKernelRMSNormBwd(
		kernelID,
		commandBufferID,

		inputDataID,
		inputGradID,
		outputDataID,
		outputGradID,
		aggDataID,
		aggGradID,
		C.uint(chunkSize),
	)
}

func CustomKernelMeanByRows(
	kernelID,
	commandBufferID,
	inputDataID,
	outputDataID unsafe.Pointer,
	chunkSize int,
) {
	C.customKernelMeanByRows(
		kernelID,
		commandBufferID,
		inputDataID,
		outputDataID,
		C.uint(chunkSize),
	)
}

func CustomKernelMeanByRowsBwd(
	kernelID,
	commandBufferID,
	inputGradID,
	outputGradID unsafe.Pointer,
	chunkSize int,
) {
	C.customKernelMeanByRowsBwd(
		kernelID,
		commandBufferID,
		inputGradID,
		outputGradID,
		C.uint(chunkSize),
	)
}

func CustomKernelConcatByRows(
	kernelID,
	commandBufferID,

	inputDataID,
	outputDataID unsafe.Pointer,

	inputWidth,
	outputWidth,
	outputOffset int,
) {
	C.customKernelConcatByRows(
		kernelID,
		commandBufferID,
		inputDataID,
		outputDataID,
		C.uint(inputWidth),
		C.uint(outputWidth),
		C.uint(outputOffset),
	)
}

func CustomKernelConcatByRowsBwd(
	kernelID,
	commandBufferID,

	inputGradID,
	outputGradID unsafe.Pointer,

	inputWidth,
	outputWidth,
	outputOffset int,
) {
	C.customKernelConcatByRowsBwd(
		kernelID,
		commandBufferID,
		inputGradID,
		outputGradID,
		C.uint(inputWidth),
		C.uint(outputWidth),
		C.uint(outputOffset),
	)
}

func CustomKernelEmbeddings(
	kernelID,
	commandBufferID,

	inputDataID,
	outputDataID,
	posEmbeddingBufferID,
	tokenEmbeddingBufferID unsafe.Pointer,

	featuresCount,
	contextLength int,
) {
	C.customKernelEmbeddings(
		kernelID,
		commandBufferID,
		inputDataID,
		outputDataID,
		posEmbeddingBufferID,
		tokenEmbeddingBufferID,
		C.uint(featuresCount),
		C.uint(contextLength),
	)
}

func CustomKernelEmbeddingsBwd(
	kernelID,
	commandBufferID,

	inputDataID,
	outputGradID,
	tokenEmbeddingGradBufferID unsafe.Pointer,

	featuresCount int,
) {
	C.customKernelEmbeddingsBwd(
		kernelID,
		commandBufferID,
		inputDataID,
		outputGradID,
		tokenEmbeddingGradBufferID,
		C.uint(featuresCount),
	)
}

func TransposeTo(
	kernelID,
	commandBufferID,

	inputData,
	outputData unsafe.Pointer,

	width,
	height int,
) {
	C.transposeTo(
		kernelID,
		commandBufferID,
		inputData,
		outputData,
		C.uint(width),
		C.uint(height),
	)
}

func TransposeAndAddTo(
	kernelID,
	commandBufferID,

	inputData,
	outputData unsafe.Pointer,

	width,
	height int,
) {
	C.transposeAndAddTo(
		kernelID,
		commandBufferID,
		inputData,
		outputData,
		C.uint(width),
		C.uint(height),
	)
}
