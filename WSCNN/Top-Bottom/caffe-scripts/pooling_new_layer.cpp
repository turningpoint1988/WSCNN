#include <algorithm>
#include <cfloat>
#include <vector>
#include <utility>

#include "caffe/layers/pooling_new_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

using namespace std;

template <typename Dtype>
void PoolingNewLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  PoolingNewParameter pool_new_param = this->layer_param_.pooling_new_param();
  kernel_h_ = bottom[0]->height();
  kernel_w_ = bottom[0]->width();
  CHECK_GT(kernel_h_, 0) << "Filter dimensions cannot be zero.";
  CHECK_GT(kernel_w_, 0) << "Filter dimensions cannot be zero.";
  top_instances_ = pool_new_param.top_instances();
  bottom_instances_ = pool_new_param.bottom_instances();
  CHECK_GT(top_instances_ + bottom_instances_, 0) << "(top_instances_+bottom_instances_) cannot be zero.";
  aver_flag_ = pool_new_param.aver_flag();
}


template <typename Dtype>
void PoolingNewLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(4, bottom[0]->num_axes()) << "Input must have 4 axes, "
      << "corresponding to (num, channels, height, width)";
  channels_ = bottom[0]->channels();
  height_ = bottom[0]->height();
  width_ = bottom[0]->width();
  CHECK_LE(top_instances_ + bottom_instances_, height_ * width_) << "top_instances_ + bottom_instances_ should be less than or equal to height_*width_";
  pooled_height_ = static_cast<int>(ceil(static_cast<float>(
      height_ - kernel_h_))) + 1;
  pooled_width_ = static_cast<int>(ceil(static_cast<float>(
      width_ - kernel_w_))) + 1;
  CHECK_EQ(pooled_height_,1) << "pooled_height_ must be one";
  CHECK_EQ(pooled_width_,1) << "pooled_width_ must be one";
  top[0]->Reshape(bottom[0]->num(), channels_, pooled_height_,
      pooled_width_);
  max_idx_.Reshape(bottom[0]->num(), channels_, top_instances_ + bottom_instances_, pooled_width_);
}
template <typename Dtype>
bool Comparator(const pair<Dtype,int> l1, const pair<Dtype,int> l2) {
		return (l1.first < l2.first);
}
// TODO(Yangqing): Is there a faster way to do pooling in the channel-first
// case?
template <typename Dtype>
void PoolingNewLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  const int top_count = top[0]->count();
  int* mask = NULL;  // suppress warnings about uninitalized variables
  // Initialize
  mask = max_idx_.mutable_cpu_data();
  caffe_set(max_idx_.count(), -1, mask);
  caffe_set(top_count, Dtype(0.), top_data);
  // The main loop
  for (int n = 0; n < bottom[0]->num(); ++n) {
    for (int c = 0; c < channels_; ++c) {
      for (int ph = 0; ph < pooled_height_; ++ph) {
        for (int pw = 0; pw < pooled_width_; ++pw) {
           int hstart = ph;
           int wstart = pw;
           int hend = min(hstart + kernel_h_, height_);
           int wend = min(wstart + kernel_w_, width_);
           hstart = max(hstart, 0);
           wstart = max(wstart, 0);
           const int pool_index = ph * pooled_width_ + pw;
		   vector< pair<Dtype, int> > temp;
		   int count = 0;
           for (int h = hstart; h < hend; ++h) {
             for (int w = wstart; w < wend; ++w) {
                const int index = h * width_ + w;
				count++;
				pair<Dtype, int> element(bottom_data[index], index);
				temp.push_back(element);
             }
           }
		   sort(temp.begin(), temp.end(), Comparator<Dtype>);
		   //CHECK_EQ(count-1,index) << "the number must be equal.";
		   int top_index = count - 1;
		   int bottom_index = 0;
		   int max_min = 0;
		   int top_instances_t = top_instances_;
		   int bottom_instances_t = bottom_instances_;
		   Dtype top_values = Dtype(0.);
		   while (top_instances_t > 0) {
				top_values += temp[top_index].first;
				mask[max_min] = temp[top_index].second;
				top_index--;
				top_instances_t--;
				max_min++;
		   }
		   Dtype bottom_values = Dtype(0.);
		   while (bottom_instances_t > 0) {
				bottom_values += temp[bottom_index].first;
				mask[max_min] = temp[bottom_index].second;
				bottom_index++;
				bottom_instances_t--;
				max_min++;
		   }
		   if (aver_flag_) {
		      top_data[pool_index] = (top_values + bottom_values)/(top_instances_ + bottom_instances_);
		   } else {
		      top_data[pool_index] = top_values + bottom_values;
		   }
        }
      }
      // compute offset
      bottom_data += bottom[0]->offset(0, 1);
      top_data += top[0]->offset(0, 1);
	  mask += max_idx_.offset(0, 1);
      //mask += top_instances_ + bottom_instances_;
    }
  }
  
}

template <typename Dtype>
void PoolingNewLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (!propagate_down[0]) {
    return;
  }
  const Dtype* top_diff = top[0]->cpu_diff();
  Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
  // Different pooling methods. We explicitly do the switch outside the for
  // loop to save time, although this results in more codes.
  caffe_set(bottom[0]->count(), Dtype(0), bottom_diff);
  const int* mask = NULL;  // suppress warnings about uninitialized variables
  // The main loop
  mask = max_idx_.cpu_data();
  for (int n = 0; n < top[0]->num(); ++n) {
    for (int c = 0; c < channels_; ++c) {
      for (int ph = 0; ph < pooled_height_; ++ph) {
        for (int pw = 0; pw < pooled_width_; ++pw) {
          const int index = ph * pooled_width_ + pw;
		  for (int max_min = 0; max_min < top_instances_; ++max_min) {
		  	const int bottom_index = mask[max_min];
		  	if (aver_flag_) {
		  		bottom_diff[bottom_index] = top_diff[index]/(top_instances_ + bottom_instances_);
		  	} else {
		  	    bottom_diff[bottom_index] += top_diff[index];
		  	}
		  } 
        }
      }
      bottom_diff += bottom[0]->offset(0, 1);
      top_diff += top[0]->offset(0, 1);
      mask += max_idx_.offset(0, 1);
    }
  }

}


#ifdef CPU_ONLY
STUB_GPU(PoolingNewLayer);
#endif

INSTANTIATE_CLASS(PoolingNewLayer);
REGISTER_LAYER_CLASS(PoolingNew);
}  // namespace caffe
