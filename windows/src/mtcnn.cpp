#include "lbf/mtcnn.h"



/******************************************************
// 函数名:CompareBBox
// 说明: bbox得分paixu
// 作者: 张峰
// 时间: 2017.11.14
// 备注:
/*******************************************************/
bool CompareBBox(const FaceInfo & a, const FaceInfo & b) {
	return a.bbox.score > b.bbox.score;
}


/******************************************************
// 函数名:IoU
// 说明:计算两个矩形的IOU
// 作者:张峰
// 时间:2017.11.14
// 备注:
/*******************************************************/
float MTCNN::IoU(float xmin, float ymin, float xmax, float ymax,
	float xmin_, float ymin_, float xmax_, float ymax_, bool is_iom) {
	float iw = std::min(xmax, xmax_) - std::max(xmin, xmin_) + 1;
	float ih = std::min(ymax, ymax_) - std::max(ymin, ymin_) + 1;
	if (iw <= 0 || ih <= 0)
		return 0;
	float s = iw*ih;
	if (is_iom) {
		float ov = s / min((xmax - xmin + 1)*(ymax - ymin + 1), (xmax_ - xmin_ + 1)*(ymax_ - ymin_ + 1));
		return ov;
	}
	else {
		float ov = s / ((xmax - xmin + 1)*(ymax - ymin + 1) + (xmax_ - xmin_ + 1)*(ymax_ - ymin_ + 1) - s);
		return ov;
	}
}


/******************************************************
// 函数名:NMS
// 说明: 非极大值抑制
// 作者:张峰
// 时间:2017.11.14
// 备注:
/*******************************************************/
std::vector<FaceInfo> MTCNN::NMS(std::vector<FaceInfo>& bboxes,
	float thresh, char methodType) {
	std::vector<FaceInfo> bboxes_nms;
	if (bboxes.size() == 0) {
		return bboxes_nms;
	}
	std::sort(bboxes.begin(), bboxes.end(), CompareBBox);

	int32_t select_idx = 0;
	int32_t num_bbox = static_cast<int32_t>(bboxes.size());
	std::vector<int32_t> mask_merged(num_bbox, 0);
	bool all_merged = false;

	while (!all_merged) {
		while (select_idx < num_bbox && mask_merged[select_idx] == 1)
			select_idx++;
		if (select_idx == num_bbox) {
			all_merged = true;
			continue;
		}

		bboxes_nms.push_back(bboxes[select_idx]);
		mask_merged[select_idx] = 1;

		FaceBox select_bbox = bboxes[select_idx].bbox;
		float area1 = static_cast<float>((select_bbox.xmax - select_bbox.xmin + 1) * (select_bbox.ymax - select_bbox.ymin + 1));
		float x1 = static_cast<float>(select_bbox.xmin);
		float y1 = static_cast<float>(select_bbox.ymin);
		float x2 = static_cast<float>(select_bbox.xmax);
		float y2 = static_cast<float>(select_bbox.ymax);

		select_idx++;
#pragma omp parallel for num_threads(threads_num)
		for (int32_t i = select_idx; i < num_bbox; i++) {
			if (mask_merged[i] == 1)
				continue;

			FaceBox & bbox_i = bboxes[i].bbox;
			float x = std::max<float>(x1, static_cast<float>(bbox_i.xmin));
			float y = std::max<float>(y1, static_cast<float>(bbox_i.ymin));
			float w = std::min<float>(x2, static_cast<float>(bbox_i.xmax)) - x + 1;
			float h = std::min<float>(y2, static_cast<float>(bbox_i.ymax)) - y + 1;
			if (w <= 0 || h <= 0)
				continue;

			float area2 = static_cast<float>((bbox_i.xmax - bbox_i.xmin + 1) * (bbox_i.ymax - bbox_i.ymin + 1));
			float area_intersect = w * h;

			switch (methodType) {
			case 'u':
				if (static_cast<float>(area_intersect) / (area1 + area2 - area_intersect) > thresh)
					mask_merged[i] = 1;
				break;
			case 'm':
				if (static_cast<float>(area_intersect) / std::min(area1, area2) > thresh)
					mask_merged[i] = 1;
				break;
			default:
				break;
			}
		}
	}
	return bboxes_nms;
}



/******************************************************
// 函数名:BBoxRegression
// 说明:BBox回归
// 作者:张峰
// 时间:2017.11.4
// 备注:
/*******************************************************/
void MTCNN::BBoxRegression(vector<FaceInfo>& bboxes) {
#pragma omp parallel for num_threads(threads_num)
	for (int i = 0; i < bboxes.size(); ++i) {
		FaceBox &bbox = bboxes[i].bbox;
		float *bbox_reg = bboxes[i].bbox_reg;
		float w = bbox.xmax - bbox.xmin + 1;
		float h = bbox.ymax - bbox.ymin + 1;
		bbox.xmin += bbox_reg[0] * w;
		bbox.ymin += bbox_reg[1] * h;
		bbox.xmax += bbox_reg[2] * w;
		bbox.ymax += bbox_reg[3] * h;
	}
}


/******************************************************
// 函数名:BBoxPad
// 说明:BBox边界判定
// 作者:张峰
// 时间:2017.11.4
// 备注:防止图片越界
/*******************************************************/
void MTCNN::BBoxPad(vector<FaceInfo>& bboxes, int width, int height) {
#pragma omp parallel for num_threads(threads_num)
	for (int i = 0; i < bboxes.size(); ++i) {
		FaceBox &bbox = bboxes[i].bbox;
		bbox.xmin = round(max(bbox.xmin, 0.f));
		bbox.ymin = round(max(bbox.ymin, 0.f));
		bbox.xmax = round(min(bbox.xmax, width - 1.f));
		bbox.ymax = round(min(bbox.ymax, height - 1.f));
	}
}



/******************************************************
// 函数名:BBoxPadSquare
// 说明:BBox边界判定
// 作者:张峰
// 时间:2017.11.14
// 备注:
/*******************************************************/
void MTCNN::BBoxPadSquare(vector<FaceInfo>& bboxes, int width, int height) {
#pragma omp parallel for num_threads(threads_num)
	for (int i = 0; i < bboxes.size(); ++i) {
		FaceBox &bbox = bboxes[i].bbox;
		float w = bbox.xmax - bbox.xmin + 1;
		float h = bbox.ymax - bbox.ymin + 1;
		float side = h>w ? h : w;
		bbox.xmin = round(max(bbox.xmin + (w - side)*0.5f, 0.f));

		bbox.ymin = round(max(bbox.ymin + (h - side)*0.5f, 0.f));
		bbox.xmax = round(min(bbox.xmin + side - 1, width - 1.f));
		bbox.ymax = round(min(bbox.ymin + side - 1, height - 1.f));
	}
}


/******************************************************
// 函数名:GenerateBBox
// 说明: 生成Bbox
// 作者:张峰
// 时间:2017.11.4
// 备注:
/*******************************************************/
void MTCNN::GenerateBBox(Blob<float>* confidence, Blob<float>* reg_box,
	float scale, float thresh) {
	int feature_map_w_ = confidence->width();
	int feature_map_h_ = confidence->height();
	int spatical_size = feature_map_w_*feature_map_h_;
	const float* confidence_data = confidence->cpu_data() + spatical_size;
	const float* reg_data = reg_box->cpu_data();
	candidate_boxes_.clear();
	for (int i = 0; i<spatical_size; i++) {
		if (confidence_data[i] >= thresh) {

			int y = i / feature_map_w_;
			int x = i - feature_map_w_ * y;
			FaceInfo faceInfo;
			FaceBox &faceBox = faceInfo.bbox;

			faceBox.xmin = (float)(x * pnet_stride) / scale;
			faceBox.ymin = (float)(y * pnet_stride) / scale;
			faceBox.xmax = (float)(x * pnet_stride + pnet_cell_size - 1.f) / scale;
			faceBox.ymax = (float)(y * pnet_stride + pnet_cell_size - 1.f) / scale;

			faceInfo.bbox_reg[0] = reg_data[i];
			faceInfo.bbox_reg[1] = reg_data[i + spatical_size];
			faceInfo.bbox_reg[2] = reg_data[i + 2 * spatical_size];
			faceInfo.bbox_reg[3] = reg_data[i + 3 * spatical_size];

			faceBox.score = confidence_data[i];
			candidate_boxes_.push_back(faceInfo);
		}
	}
}


/******************************************************
// 函数名:FaceDetect
// 说明: 构造函数
// 作者:张峰
// 时间:加在模型
// 备注:
/*******************************************************/
MTCNN::MTCNN(const string& proto_model_dir) {
	Caffe::set_mode(Caffe::GPU);
	PNet_.reset(new Net<float>((proto_model_dir + "/det1.prototxt"), TEST));
	PNet_->CopyTrainedLayersFrom(proto_model_dir + "/det1.caffemodel");
	RNet_.reset(new Net<float>((proto_model_dir + "/det2.prototxt"), TEST));
	RNet_->CopyTrainedLayersFrom(proto_model_dir + "/det2.caffemodel");
	ONet_.reset(new Net<float>((proto_model_dir + "/det3.prototxt"), TEST));
	ONet_->CopyTrainedLayersFrom(proto_model_dir + "/det3.caffemodel");
	//ONet_.reset(new Net<float>((proto_model_dir + "/det3-half.prototxt"), TEST));
	//ONet_->CopyTrainedLayersFrom(proto_model_dir + "/det3-half.caffemodel");
	
	Blob<float>* input_layer;
	input_layer = PNet_->input_blobs()[0];
	int num_channels_ = input_layer->channels();
	CHECK(num_channels_ == 3) << "Input layer should have 3 channels.";
}


/******************************************************
// 函数名:ProposalNet
// 说明:PNet
// 作者:张峰
// 时间:2017.11.4
// 备注:Pnet过程，生成初步候选
/*******************************************************/
vector<FaceInfo> MTCNN::ProposalNet(const cv::Mat& img, int minSize, float threshold, float factor) {
	cv::Mat  resized;
	int width = img.cols;
	int height = img.rows;
	float scale = 12.f / minSize;
	float minWH = std::min(height, width) *scale;
	std::vector<float> scales;
	while (minWH >= 12) {
		scales.push_back(scale);
		minWH *= factor;
		scale *= factor;
	}
	Blob<float>* input_layer = PNet_->input_blobs()[0];
	total_boxes_.clear();
	for (int i = 0; i < scales.size(); i++) {
		int ws = (int)std::ceil(width*scales[i]);
		int hs = (int)std::ceil(height*scales[i]);
		cv::resize(img, resized, cv::Size(ws, hs), 0, 0, cv::INTER_LINEAR);
		input_layer->Reshape(1, 3, hs, ws);
		PNet_->Reshape();
		float * input_data = input_layer->mutable_cpu_data();
		cv::Vec3b * img_data = (cv::Vec3b *)resized.data;
		int spatial_size = ws* hs;
		for (int k = 0; k < spatial_size; ++k) {
			input_data[k] = float((img_data[k][0] - mean_val)* std_val);
			input_data[k + spatial_size] = float((img_data[k][1] - mean_val) * std_val);
			input_data[k + 2 * spatial_size] = float((img_data[k][2] - mean_val) * std_val);
		}
        PNet_->Forward();
		Blob<float>* confidence = PNet_->blob_by_name("prob1").get();
		Blob<float>* reg = PNet_->blob_by_name("conv4-2").get();
		GenerateBBox(confidence, reg, scales[i], threshold);
		std::vector<FaceInfo> bboxes_nms = NMS(candidate_boxes_, 0.5, 'u');
		if (bboxes_nms.size()>0) {
			total_boxes_.insert(total_boxes_.end(), bboxes_nms.begin(), bboxes_nms.end());
		}
	}
	int num_box = (int)total_boxes_.size();
	vector<FaceInfo> res_boxes;
	if (num_box != 0) {
		res_boxes = NMS(total_boxes_, 0.7f, 'u');
		BBoxRegression(res_boxes);
		BBoxPadSquare(res_boxes, width, height);
	}
	return res_boxes;
}


/******************************************************
// 函数名:NextStage
// 说明:Rnet
// 作者:张峰
// 时间:2017.11.4
// 备注:接着Pnet,输入上一个阶段生成的proposal
/*******************************************************/
vector<FaceInfo> MTCNN::NextStage(const cv::Mat& image, vector<FaceInfo> &pre_stage_res, int input_w, int input_h, int stage_num, const float threshold) {
	vector<FaceInfo> res;
	int batch_size = (int)pre_stage_res.size();
	if (batch_size == 0)
		return res;
	Blob<float>* input_layer = nullptr;
	Blob<float>* confidence = nullptr;
	Blob<float>* reg_box = nullptr;
	Blob<float>* reg_landmark = nullptr;

	switch (stage_num) {
	case 2: {
		input_layer = RNet_->input_blobs()[0];
		input_layer->Reshape(batch_size, 3, input_h, input_w);
		RNet_->Reshape();
	}break;
	case 3: {
		input_layer = ONet_->input_blobs()[0];
		input_layer->Reshape(batch_size, 3, input_h, input_w);
		ONet_->Reshape();
	}break;
	default:
		return res;
		break;
	}
	float * input_data = input_layer->mutable_cpu_data();
	int spatial_size = input_h*input_w;

#pragma omp parallel for num_threads(threads_num)
	for (int n = 0; n < batch_size; ++n) {
		FaceBox &box = pre_stage_res[n].bbox;
		Mat roi = image(Rect(Point((int)box.xmin, (int)box.ymin), Point((int)box.xmax, (int)box.ymax))).clone();
		resize(roi, roi, Size(input_w, input_h));
		float *input_data_n = input_data + input_layer->offset(n);
		Vec3b *roi_data = (Vec3b *)roi.data;
		CHECK_EQ(roi.isContinuous(), true);
		for (int k = 0; k < spatial_size; ++k) {
			input_data_n[k] = float((roi_data[k][0] - mean_val)*std_val);
			input_data_n[k + spatial_size] = float((roi_data[k][1] - mean_val)*std_val);
			input_data_n[k + 2 * spatial_size] = float((roi_data[k][2] - mean_val)*std_val);
		}
	}
	switch (stage_num) {
	case 2: {
		RNet_->Forward();
		confidence = RNet_->blob_by_name("prob1").get();
		reg_box = RNet_->blob_by_name("conv5-2").get();
	}break;
	case 3: {
		ONet_->Forward();
		confidence = ONet_->blob_by_name("prob1").get();
		reg_box = ONet_->blob_by_name("conv6-2").get();
		reg_landmark = ONet_->blob_by_name("conv6-3").get();
	}break;
	}
	const float* confidence_data = confidence->cpu_data();
	const float* reg_data = reg_box->cpu_data();
	const float* landmark_data = nullptr;
	if (reg_landmark) {
		landmark_data = reg_landmark->cpu_data();
	}
	for (int k = 0; k < batch_size; ++k) {
		if (confidence_data[2 * k + 1] >= threshold) {
			FaceInfo info;
			info.bbox.score = confidence_data[2 * k + 1];
			info.bbox.xmin = pre_stage_res[k].bbox.xmin;
			info.bbox.ymin = pre_stage_res[k].bbox.ymin;
			info.bbox.xmax = pre_stage_res[k].bbox.xmax;
			info.bbox.ymax = pre_stage_res[k].bbox.ymax;
			for (int i = 0; i < 4; ++i) {
				info.bbox_reg[i] = reg_data[4 * k + i];
			}
			if (reg_landmark) {
				float w = info.bbox.xmax - info.bbox.xmin + 1.f;
				float h = info.bbox.ymax - info.bbox.ymin + 1.f;
				for (int i = 0; i < 5; ++i){
					info.landmark[2 * i] = landmark_data[10 * k + 2 * i] * w + info.bbox.xmin;
					info.landmark[2 * i + 1] = landmark_data[10 * k + 2 * i + 1] * h + info.bbox.ymin;
				}
			}
			res.push_back(info);
		}
	}
	return res;
}


/******************************************************
// 函数名:Detect
// 说明:mtcnn函数检测人脸和特征点入口函数
// 作者:张峰
// 时间:2017.11.04
// 备注:
/*******************************************************/
vector<FaceInfo> MTCNN::Detect(const cv::Mat& image, const int minSize, const float* threshold, const float factor, const int stage) {
	vector<FaceInfo> pnet_res;
	vector<FaceInfo> rnet_res;
	vector<FaceInfo> onet_res;
	if (stage >= 1){
		pnet_res = ProposalNet(image, minSize, threshold[0], factor);
	}
	if (stage >= 2 && pnet_res.size()>0){
		if (pnet_max_detect_num < (int)pnet_res.size()){
			pnet_res.resize(pnet_max_detect_num);
		}
		int num = (int)pnet_res.size();
		int size = (int)ceil(1.f*num / step_size);
		for (int iter = 0; iter < size; ++iter){
			int start = iter*step_size;
			int end = min(start + step_size, num);
			vector<FaceInfo> input(pnet_res.begin() + start, pnet_res.begin() + end);
			vector<FaceInfo> res = NextStage(image, input, 24, 24, 2, threshold[1]);
			rnet_res.insert(rnet_res.end(), res.begin(), res.end());
		}
		rnet_res = NMS(rnet_res, 0.7f, 'u');
		BBoxRegression(rnet_res);
		BBoxPadSquare(rnet_res, image.cols, image.rows);

	}
	if (stage >= 3 && rnet_res.size()>0){
		int num = (int)rnet_res.size();
		int size = (int)ceil(1.f*num / step_size);
		for (int iter = 0; iter < size; ++iter){
			int start = iter*step_size;
			int end = min(start + step_size, num);
			vector<FaceInfo> input(rnet_res.begin() + start, rnet_res.begin() + end);
			vector<FaceInfo> res = NextStage(image, input, 48, 48, 3, threshold[2]);
			onet_res.insert(onet_res.end(), res.begin(), res.end());
		}
		BBoxRegression(onet_res);
		onet_res = NMS(onet_res, 0.7f, 'm');
		BBoxPad(onet_res, image.cols, image.rows);

	}
	if (stage == 1){
		return pnet_res;
	}
	else if (stage == 2){
		return rnet_res;
	}
	else if (stage == 3){
		return onet_res;
	}
	else{
		return onet_res;
	}
}
