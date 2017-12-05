#ifndef _REGISTER_H_
#define _REGISTER_H_
#include "caffe/common.hpp"
#include "caffe/layers/input_layer.hpp"
#include "caffe/layers/inner_product_layer.hpp"
#include "caffe/layers/dropout_layer.hpp"
#include "caffe/layers/conv_layer.hpp"
#include "caffe/layers/relu_layer.hpp"
#include "caffe/layers/reshape_layer.hpp"
#include "caffe/layers/pooling_layer.hpp"
#include "caffe/layers/lrn_layer.hpp"
#include "caffe/layers/softmax_layer.hpp"
#include "caffe/layers/roi_pooling_layer.hpp"
#include "caffe/layers/prelu_layer.hpp"
#include "caffe/layers/batch_norm_layer.hpp"
#include "caffe/layers/scale_layer.hpp"
#include "caffe/layers/eltwise_layer.hpp"
#include "caffe/layers/bias_layer.hpp"
#include "caffe/layers/flatten_layer.hpp"
#include "caffe/layers/concat_layer.hpp"
#include "caffe/layers/flip_layer.hpp"
#include "caffe/layers/split_layer.hpp"
namespace caffe
{
	REGISTER_LAYER_CLASS(Input);
	extern INSTANTIATE_CLASS(InputLayer);

	REGISTER_LAYER_CLASS(Split);
	extern INSTANTIATE_CLASS(SplitLayer);

	REGISTER_LAYER_CLASS(InnerProduct);
	extern INSTANTIATE_CLASS(InnerProductLayer);

	REGISTER_LAYER_CLASS(Dropout);
	extern INSTANTIATE_CLASS(DropoutLayer);

	REGISTER_LAYER_CLASS(ROIPooling);
	extern INSTANTIATE_CLASS(ROIPoolingLayer);

	REGISTER_LAYER_CLASS(Convolution);
	extern INSTANTIATE_CLASS(ConvolutionLayer);

	REGISTER_LAYER_CLASS(ReLU);
	extern INSTANTIATE_CLASS(ReLULayer);
	
	REGISTER_LAYER_CLASS(Pooling);
	extern INSTANTIATE_CLASS(PoolingLayer);
	
	REGISTER_LAYER_CLASS(LRN);
	extern INSTANTIATE_CLASS(LRNLayer);
	
	REGISTER_LAYER_CLASS(Softmax);
	extern INSTANTIATE_CLASS(SoftmaxLayer);
	
	extern INSTANTIATE_CLASS(ReshapeLayer);
	REGISTER_LAYER_CLASS(PReLU);
	extern INSTANTIATE_CLASS(PReLULayer);
	extern INSTANTIATE_CLASS(BatchNormLayer);
	extern INSTANTIATE_CLASS(ScaleLayer);
	extern INSTANTIATE_CLASS(EltwiseLayer);
	extern INSTANTIATE_CLASS(BiasLayer);
	extern INSTANTIATE_CLASS(FlattenLayer);
	extern INSTANTIATE_CLASS(ConcatLayer);
	extern INSTANTIATE_CLASS(FlipLayer);
	//	REGISTER_LAYER_CLASS(Reshape);
}
#endif