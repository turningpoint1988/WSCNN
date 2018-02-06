#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layers/pooling_new_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {
/*
#define CUDA_KERNEL_LOOP(i,n) \

for(int i = blockIdx.x * blockDim.x + threadIdx.x; \

i < (n); i +=blockDim.x * gridDim.x)
*/

struct pair_ {
	float data;
	int index;
};

template <typename Dtype>
__global__ void MaxPoolForward(const int nthreads,
    const Dtype* const bottom_data, const int num, const int channels, const int height, const int width, const int pooled_height,
    const int pooled_width, const int kernel_h, const int kernel_w, const int top_instances, const int bottom_instances,
    Dtype* const top_data, int* mask, struct pair_* pairdata, const bool aver_flag) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int pw = index % pooled_width;
    const int ph = (index / pooled_width) % pooled_height;
    const int c = (index / pooled_width / pooled_height) % channels;
    const int n = index / pooled_width / pooled_height / channels;
    int hstart = ph;
    int wstart = pw;
    const int hend = min(hstart + kernel_h, height);
    const int wend = min(wstart + kernel_w, width);
    hstart = max(hstart, 0);
    wstart = max(wstart, 0);
	int count = 0;
    const Dtype* const bottom_slice =
        bottom_data + (n * channels + c) * height * width;
	int* mask_slice = 
		mask + (n * channels + c) * (top_instances + bottom_instances) * pooled_width;
	struct pair_* pairdata_slice = 
		pairdata + (n * channels + c) * height * width;
    for (int h = hstart; h < hend; ++h) {
      for (int w = wstart; w < wend; ++w) {
		pairdata_slice[count].data = static_cast<float>(bottom_slice[h * width + w]);
		pairdata_slice[count].index = h * width + w;
		count++;
      }
    }
	//sort the pair data
	for (int i = 0; i < count - 1; ++i) {
	  for (int j = 0; j < count - 1 - i; ++j) {
		if (pairdata_slice[j].data > pairdata_slice[j+1].data) {
		   struct pair_ temp = pairdata_slice[j];
		   pairdata_slice[j] = pairdata_slice[j+1];
		   pairdata_slice[j+1] = temp;
		}
	  }
	}
	int top_index = count - 1;
	int max_min = 0;
	int top_instances_t = top_instances;
	Dtype top_values = Dtype(0.);
	while (top_instances_t > 0) {
		top_values += Dtype(pairdata_slice[top_index].data);
		mask_slice[max_min] = pairdata_slice[top_index].index;
		top_index--;
		top_instances_t--;
		max_min++;
	}
	int bottom_index = 0;
	int bottom_instances_t = bottom_instances;
	Dtype bottom_values = Dtype(0.);
	while (bottom_instances_t > 0) {
		bottom_values += Dtype(pairdata_slice[bottom_index].data);
		mask_slice[max_min] = pairdata_slice[bottom_index].index;
		bottom_index++;
		bottom_instances_t--;
		max_min++;
	}
	if (aver_flag) {
		top_data[index] = (top_values + bottom_values)/(top_instances + bottom_instances);
	} else {
	    top_data[index] = top_values + bottom_values;
	}
  }
}

template <typename Dtype>
__global__ void gpu_set(const int nthreads, struct pair_* pairdata) {
	CUDA_KERNEL_LOOP(index, nthreads) {
		pairdata[index].data = 0.;
		pairdata[index].index = 0;
	}
}
template <typename Dtype>
void PoolingNewLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  int count = top[0]->count();
  int* mask = NULL;
  mask = max_idx_.mutable_gpu_data();
  caffe_gpu_set(count, Dtype(0.), top_data);
  caffe_gpu_set(max_idx_.count(), -1, mask);
  int count_bottom = bottom[0]->count();
  struct pair_* pairdata;
  cudaMallocManaged((void **)&pairdata, count_bottom * sizeof(struct pair_));
  gpu_set<Dtype><<<CAFFE_GET_BLOCKS(count_bottom), CAFFE_CUDA_NUM_THREADS>>>(count_bottom, pairdata);
  cudaDeviceSynchronize();
  // NOLINT_NEXT_LINE(whitespace/operators)
  MaxPoolForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      count, bottom_data, bottom[0]->num(), channels_, height_, width_, pooled_height_, pooled_width_, kernel_h_,
      kernel_w_, top_instances_, bottom_instances_, top_data, mask, pairdata, aver_flag_);
  cudaDeviceSynchronize();
  CUDA_POST_KERNEL_CHECK;
  cudaFree(pairdata);
}


template <typename Dtype>
__global__ void MaxPoolBackward(const int nthreads, const Dtype* const top_diff,
    const int* const mask, const int num, const int channels, const int height, const int width,const int pooled_height, 
    const int pooled_width, const int kernel_h, const int kernel_w, const int top_instances, const int bottom_instances,
    Dtype* const bottom_diff, const bool aver_flag) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    //const int w = index % width;
    //const int h = (index / width) % height;
    const int c = (index / pooled_width / pooled_height) % channels;
    const int n = index / pooled_width / pooled_height / channels;
	const int offset1 = (n * channels + c) * (top_instances + bottom_instances) * pooled_width;
	const int offset2 = (n * channels + c) * height * width;
    const int* const mask_slice = mask + offset1;
    Dtype* const bottom_diff_slice = bottom_diff + offset2;
    for (int min_max = 0; min_max < top_instances; ++min_max) {
       const int bottom_diff_index = mask_slice[min_max];
       if (aver_flag) {
          bottom_diff_slice[bottom_diff_index] = top_diff[index]/(top_instances + bottom_instances);
       } else {
          bottom_diff_slice[bottom_diff_index] = top_diff[index];
       }
    }
  }
}


template <typename Dtype>
void PoolingNewLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (!propagate_down[0]) {
    return;
  }
  const Dtype* top_diff = top[0]->gpu_diff();
  Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
  caffe_gpu_set(bottom[0]->count(), Dtype(0.), bottom_diff);
  const int count = top[0]->count();
  const int* mask = NULL;
  mask = max_idx_.gpu_data();
  // NOLINT_NEXT_LINE(whitespace/operators)
  MaxPoolBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      count, top_diff, mask, top[0]->num(), channels_, height_, width_, pooled_height_, pooled_width_,
      kernel_h_, kernel_w_, top_instances_, bottom_instances_, bottom_diff, aver_flag_);
  cudaDeviceSynchronize();
  CUDA_POST_KERNEL_CHECK;
}


INSTANTIATE_LAYER_GPU_FUNCS(PoolingNewLayer);


}  // namespace caffe
