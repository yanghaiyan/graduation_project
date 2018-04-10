#define CUDACCs
#include"cuda_runtime.h"
#include "cuda.h"
#include"device_launch_parameters.h"

texture<float, 1, cudaReadModeElementType> gridData1D;
texture<float, 1, cudaReadModeElementType> sumData2D;
texture<float, 1, cudaReadModeElementType> squmData2D;
texture<float, 1, cudaReadModeElementType> imageData2D;
texture<float, 1, cudaReadModeElementType> features2D;
texture<float, 1, cudaReadModeElementType> totalFeatures1D;
texture<float, 1, cudaReadModeElementType>  poster2D;

void setGird(float  sgrid[], int gridLength) {
	float *dev_grid;
	int grid_data_size = sizeof(float)*gridLength;
	cudaMalloc((void**)&dev_grid, grid_data_size);
	cudaMemcpy(dev_grid, sgrid, grid_data_size, cudaMemcpyHostToDevice);
	cudaBindTexture(0, gridData1D, dev_grid);
}

void setSumAndSQum(int w, int h, int squm[], float sum[], int squmLen, int sumLen) {
	cudaArray *sumArray;
	cudaArray *squmArray;
	int squm_data_size = sizeof(float)*squmLen;
	int sum_data_size = sizeof(float)*sumLen;

	cudaChannelFormatDesc chDesc6 = cudaCreateChannelDesc<int>();
	cudaChannelFormatDesc  chDesc7 = cudaCreateChannelDesc<float>();
	cudaMallocArray(&sumArray, &chDesc6, w, h);
	cudaMallocArray(&squmArray, &chDesc7, w, h);

	cudaMemcpyToArray(sumArray, 0, 0, sum, sum_data_size, cudaMemcpyHostToDevice);
	cudaMemcpyToArray(squmArray, 0, 0, squm, squm_data_size, cudaMemcpyHostToDevice);
	cudaBindTextureToArray(sumData2D, sumArray);
	cudaBindTextureToArray(squmData2D, squmArray);
}
// 效果有限 不建议使用
__global__ void varClassifier(float *tfans, int gird_w, float var) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= gird_w) {
		return;
	}
	int index = tid;
	int box_x = tex1Dfetch(gridData1D, index); //使用纹理内存
	int box_y = tex1Dfetch(gridData1D, index + gird_w);
	int box_w = tex1Dfetch(gridData1D, index + gird_w * 2);
	int box_h = tex1Dfetch(gridData1D, index + gird_w * 3);
	int scale_idx = tex1Dfetch(gridData1D, index + gird_w * 4);

	float brs = tex2D(sumData2D, box_x + box_w, box_y + box_h);
	float bls = tex2D(sumData2D, box_y + box_h);
	float trs = tex2D(sumData2D, box_x + box_w, box_y);
	float tls = tex2D(sumData2D, box_x, box_y);

	float brsq = tex2D(squmData2D, box_x + box_w, box_y + box_h);
	float blsq = tex2D(squmData2D, box_x, box_y + box_h);
	float trsq = tex2D(squmData2D, box_x + box_w, box_y);
	float tlsq = tex2D(squmData2D, box_x, box_y);

	float mean = (brs + tls - trs - bls) / ((float)box_w*box_h);
	float sqmean = (brsq + tlsq - trsq - blsq) / ((float)box_w*box_h);
	float temp = sqmean - mean*mean;//return   sqmean-mean*mean;

	if (temp >= var) {
		tfans[tid] = tid;
	}
	else {
		tfans[tid] = -1;
	}

}

__global__ void upPoker(float *dev_poater, int *dev_upPosInd, float * dev_upPos, int pos_pitch)
{
	int tid = threadIdx.x;
	int idx = dev_upPos[tid];
	float var = dev_upPos[tid];
	*((float*)((char*)dev_poater + pos_pitch *tid) + idx) = var;
}

__global__ void collectionClassifier(int *ans, int h, float threhold, int grid_w, int nstructs, int structSize)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= h) return;

	int index = tex1Dfetch(totalFeatures1D, tid);
	int box_x = tex1Dfetch(gridData1D, tid); //使用纹理内存
	int box_y = tex1Dfetch(gridData1D, index + grid_w);
	int box_w = tex1Dfetch(gridData1D, index + grid_w * 2);
	int box_h = tex1Dfetch(gridData1D, index + grid_w * 3);
	int scale_idx = tex1Dfetch(gridData1D, index + grid_w * 4);
	int leaf, x1, x2, y1, y2, imbig = 0;
	float votes = 0;
	float point1, point2;

	for (int t = 0; t < nstructs; t++) {
		leaf = 0;

		for (int f = 0; f < structSize; f++) {
			x1 = tex2D(features2D, t*structSize + f, scale_idx);
			x2 = tex2D(features2D, t*structSize + structSize *nstructs * 2 + f, scale_idx);
			y1 = tex2D(features2D, t*structSize + structSize *nstructs + f, scale_idx);
			y2 = tex2D(features2D, t*structSize + structSize *nstructs * 3 + f, scale_idx);

			point1 = tex2D(imageData2D, box_x + x1, box_y + y1);
			point2 = tex2D(imageData2D, box_x + x2, box_y + y2);

			if (point1 > point2) {
				imbig = 1;
			}
			else {
				imbig = 0;
			}
			leaf = (leaf << 1) + imbig;
		}
		ans[tid*(nstructs + 2) + t] = leaf;
		votes = tex2D(poster2D, leaf, t);
	}

	float conf = votes;
	ans[tid*(nstructs + 2) + nstructs] = conf;

	if (conf >threhold) {
		ans[tid*(nstructs + 2) + nstructs + 1] = index;
	}
	else {
		ans[tid*(nstructs + 2) + nstructs + 1] = -1;
	}

	/*

	*/
	int* runCollectionClassifier(int varisNum,  int ansLength,int h, float threhold, int grid_w, int nstructs, int structSize) {
		
		int * inAns;
		int *outAns;
		outAns = new int [ansLength];
		cudaMalloc((void**)&inAns, ansLength);
		dim3 block(varisNum,1,1);
		dim3 grid(10, 13,1);
		collectionClassifier << <grid, block, 0 >> > (inAns, h, threhold, grid_w, nstructs, structSize);
		cudaMemcpy(outAns, inAns, ansLength, cudaMemcpyDeviceToHost);
		return outAns；
	}
}
