#include "lbf/lbf.hpp"
#include "lbf/mtcnn.h"
#include <cstdio>
#include <chrono>
#include <opencv2/opencv.hpp>
using namespace cv;
using namespace std;
using namespace lbf;

// dirty but works
void parseTxt(string &txt, vector<Mat> &imgs, vector<Mat> &gt_shapes, vector<BBox> &bboxes);

/******************************************************
// 函数名:runVideo
// 说明:测试mtcnn+3000fps效果
// 作者:张峰
// 时间:2017.11.04
// 备注:half表明加在mtcnn 加速的model
/*******************************************************/
int runVideo(char* key){
	
	VideoCapture capture(0);
    if(!capture.isOpened()){
        std::cout<<"open camera failed\n";
        return -1;
    }

	// 读取模型配置 ，加在模型
    Config &config = Config::GetInstance();
	bool flag = false;
	if (!strcmp(key, "half")){
		flag = true;
		config.saved_file_name = "../model/68_half.model";
	}
    int landmark_n = config.landmark_n;
    vector<double> x(landmark_n), y(landmark_n);
    LbfCascador lbf_cascador;
    FILE *model = fopen(config.saved_file_name.c_str(), "rb");
    lbf_cascador.Read(model);
    fclose(model);
	
	// mtcnn 加在模型
	MTCNN det("../model", flag);
	float factor = 0.709f, fps = 15;
	float threshold[3]={0.7f,0.6f,0.6f};
	int minSize = 40, frame_count = 0;
	Mat frame;
	std::chrono::time_point<std::chrono::system_clock> p1, p0 = std::chrono::system_clock::now();//计算时间

    while(true){
        capture >>frame;
        if(frame.empty()){
            continue;
        }
        vector<FaceInfo> faceInfo = det.Detect(frame, minSize, threshold, factor, 3);
        if (faceInfo.size() == 0) 
            continue;
        for(int i = 0 ; i < faceInfo.size();++i){
			int x = (int)faceInfo[i].bbox.xmin;
			int y = (int)faceInfo[i].bbox.ymin;
			int w = (int)(faceInfo[i].bbox.xmax - faceInfo[i].bbox.xmin );
			int h = (int)(faceInfo[i].bbox.ymax - faceInfo[i].bbox.ymin );
			Rect r = Rect(x, y, w,h); // 人脸检测的框
           // Mat roi  = frame(r).clone();
            Mat gray;
            cvtColor(frame, gray, CV_BGR2GRAY);
            BBox tmp(x,y,w,h);
            //BBox tmp(r.x,r.y,r.width,r.height);
            Mat shape = lbf_cascador.Predict(gray, tmp);
            frame = drawShapeInImage(frame,shape,tmp);
			frame_count++;
			if (frame_count % 10 == 0)
			{
				p1 = std::chrono::system_clock::now();
				fps = 10.0 * 1000 / ((float)std::chrono::duration_cast<std::chrono::microseconds>(p1 - p0).count() / 1000);
				p0 = p1;
				frame_count = 0;
			}
			string fpsSt = "FPS:" + to_string(fps);
			cv::putText(frame, fpsSt.c_str(), cv::Point(10, 20), CV_FONT_HERSHEY_SIMPLEX, 0.5, CV_RGB(255, 0, 0), 1, CV_AA);
            imshow("landmark",frame);
            if( 'q' == waitKey(1)){
                return -1;
            }
        }
    }
	return -1;
}

int runVideo(void){
    VideoCapture capture(0);
    if(!capture.isOpened()){
        std::cout<<"open camera failed\n";
        return -1;
    }
    Mat frame;
    Config &config = Config::GetInstance();
    int N;
    int landmark_n = config.landmark_n;
    char img_path[256];
    double bbox[4];
    vector<double> x(landmark_n), y(landmark_n);

    LbfCascador lbf_cascador;
    FILE *model = fopen(config.saved_file_name.c_str(), "rb");
    lbf_cascador.Read(model);
    fclose(model);
    CascadeClassifier cc("../model/haarcascade_frontalface_alt.xml");
    while(true){
        capture >>frame;
        if(frame.empty()){
            continue;
        }
         vector<Rect> rects;
        cc.detectMultiScale(frame, rects, 1.05, 2, CV_HAAR_SCALE_IMAGE, Size(30, 30));
        if (rects.size() == 0) 
            continue;
        for(int i = 0 ; i < rects.size();++i){
            Rect r = rects[i];
            Mat frame = frame(r).clone();
            Mat gray;
            cvtColor(frame, gray, CV_BGR2GRAY);
            BBox tmp(r.x,r.y,r.width,r.height);
            Mat shape = lbf_cascador.Predict(gray, tmp);
            frame = drawShapeInImage(frame,shape,tmp);
            imshow("landmark",frame);
            if( 27 == waitKey(1)){
                break;
            }
        }
    }
    return 1;
}


int test(void) {
    Config &config = Config::GetInstance();

    LbfCascador lbf_cascador;
    FILE *fd = fopen(config.saved_file_name.c_str(), "rb");
    lbf_cascador.Read(fd);
    fclose(fd);

    //LOGG("Load test data from %s", config.dataset.c_str());
    string txt = config.dataset + "/test.txt";
    vector<Mat> imgs, gt_shapes;
    vector<BBox> bboxes;
    parseTxt(txt, imgs, gt_shapes, bboxes);

    int N = imgs.size();
    lbf_cascador.Test(imgs, gt_shapes, bboxes);

    return 0;
}

int run(void) {
    Config &config = Config::GetInstance();
    FILE *fd = fopen((config.dataset + "/test.txt").c_str(), "r");
    assert(fd);
    int N;
    int landmark_n = config.landmark_n;
    fscanf(fd, "%d", &N);
    char img_path[256];
    double bbox[4];
    vector<double> x(landmark_n), y(landmark_n);

    LbfCascador lbf_cascador;
    FILE *model = fopen(config.saved_file_name.c_str(), "rb");
    lbf_cascador.Read(model);
    fclose(model);

    for (int i = 0; i < N; i++) {
        fscanf(fd, "%s", img_path);
        for (int j = 0; j < 4; j++) {
            fscanf(fd, "%lf", &bbox[j]);
        }
        for (int j = 0; j < landmark_n; j++) {
            fscanf(fd, "%lf%lf", &x[j], &y[j]);
        }
        Mat img = imread(img_path);
        // crop img
        double x_min, y_min, x_max, y_max;
        x_min = *min_element(x.begin(), x.end());
        x_max = *max_element(x.begin(), x.end());
        y_min = *min_element(y.begin(), y.end());
        y_max = *max_element(y.begin(), y.end());
        x_min = max(0., x_min - bbox[2] / 2);
        x_max = min(img.cols - 1., x_max + bbox[2] / 2);
        y_min = max(0., y_min - bbox[3] / 2);
        y_max = min(img.rows - 1., y_max + bbox[3] / 2);
        double x_, y_, w_, h_;
        x_ = x_min; y_ = y_min;
        w_ = x_max - x_min; h_ = y_max - y_min;
        BBox bbox_(bbox[0] - x_, bbox[1] - y_, bbox[2], bbox[3]);
        Rect roi(x_, y_, w_, h_);
        std::cout<<"x_="<<x_<<"y_="<<y_<<"w_="<<w_<<"h_="<<h_<<std::endl;
        cout<<img.rows<<"\t"<<img.cols<<endl;
        img = img(roi).clone();
        cout<<img.rows<<"\t"<<img.cols<<endl;
        Mat gray;
        cvtColor(img, gray, CV_BGR2GRAY);
       // LOGG("Run %s", img_path);
        Mat shape = lbf_cascador.Predict(gray, bbox_);
        img = drawShapeInImage(img, shape, bbox_);
        imshow("landmark", img);
        waitKey(0);
    }
    fclose(fd);
    return 0;
}
