#include <cstdio>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include "lbf/common.hpp"
#include "lbf/mtcnn.h"
using namespace cv;
using namespace std;
using namespace lbf;
const int landmark_n = 68;
MTCNN detector("../model");
CascadeClassifier cc("../model/haarcascade_frontalface_alt.xml");
float factor =0.709f;
float thresholds[3]={0.7f,0.6f,0.6f};
int minSize = 40;


/******************************************************
// 函数名:getBbox
// 说明:给定的人脸矩形与给定的pts文件，找出适配的人脸
// 作者:张峰
// 时间:2017.11.30
// 备注:shape为x1,y1,x2,y2,.....
// 备注:MTCNN的人脸检测接口
/*******************************************************/
cv::Rect getBBox(Mat& img,vector<FaceInfo>& faceInfo,Mat_<double> &shape){
	vector<Rect> rects;
	if (faceInfo.size() == 0) return Rect(-1, -1, -1, -1);

	// 求出给的landmark的min_x,min_y,max_x,max_y,center_x,center_y;
	double center_x, center_y, x_min, x_max, y_min, y_max;
	center_x = center_y = 0;
	x_min = x_max = shape(0, 0);
    y_min = y_max = shape(0, 1);
    for (int i = 0; i < shape.rows; i++) {
        center_x += shape(i, 0);
        center_y += shape(i, 1);
        x_min = min(x_min, shape(i, 0));
        x_max = max(x_max, shape(i, 0));
        y_min = min(y_min, shape(i, 1));
        y_max = max(y_max, shape(i, 1));
    }
	center_x /= landmark_n;
	center_y /= landmark_n;


	//循环检查人脸
	for (int i = 0; i < faceInfo.size(); i++) {
		int x = (int)faceInfo[i].bbox.xmin;
		int y = (int)faceInfo[i].bbox.ymin;
		int w = (int)(faceInfo[i].bbox.xmax - faceInfo[i].bbox.xmin + 1);
		int h = (int)(faceInfo[i].bbox.ymax - faceInfo[i].bbox.ymin + 1);
		Rect r = Rect(x, y, w,h); // 人脸检测的框

		//shape超出了目标 或者对人脸问题
		if (x_max - x_min > r.width*1.5) continue; 
		if (y_max - y_min > r.height*1.5) continue;
		if (abs(center_x - (r.x + r.width / 2)) > r.width / 2) continue;
		if (abs(center_y - (r.y + r.height / 2)) > r.height / 2) continue;
		return r; // 一个pts只对应一个人脸
	}
	return Rect(-1, -1, -1, -1);
}



/******************************************************
// 函数名:getBbox
// 说明:给定的人脸矩形与给定的pts文件，找出适配的人脸
// 作者:张峰
// 时间:2017.11.30
// 备注:shape为x1,y1,x2,y2,.....
// 备注:opencv的人脸检测接口
/*******************************************************/
Rect getBBox(Mat &img, Mat_<double> &shape) {

	// opencv的人脸检测
    vector<Rect> rects;
    cc.detectMultiScale(img, rects, 1.05, 2, CV_HAAR_SCALE_IMAGE, Size(30, 30));

    if (rects.size() == 0) return Rect(-1, -1, -1, -1);

	// 求出给的landmark的min_x,min_y,max_x,max_y,center_x,center_y;
    double center_x, center_y, x_min, x_max, y_min, y_max;
    center_x = center_y = 0;
    x_min = x_max = shape(0, 0);
    y_min = y_max = shape(0, 1);
    for (int i = 0; i < shape.rows; i++) {
        center_x += shape(i, 0);
        center_y += shape(i, 1);
        x_min = min(x_min, shape(i, 0));
        x_max = max(x_max, shape(i, 0));
        y_min = min(y_min, shape(i, 1));
        y_max = max(y_max, shape(i, 1));
    }
    center_x /= shape.rows;
    center_y /= shape.rows;


	//循环检查人脸
    for (int i = 0; i < rects.size(); i++) {
        Rect r = rects[i];
        if (x_max - x_min > r.width*1.5) continue;
        if (y_max - y_min > r.height*1.5) continue;
        if (abs(center_x - (r.x + r.width / 2)) > r.width / 2) continue;
        if (abs(center_y - (r.y + r.height / 2)) > r.height / 2) continue;
		return r; // 一个pts只对应一个人脸
    }
    return Rect(-1, -1, -1, -1);
}



/******************************************************
// 函数名:genTxt
// 说明:从txt文件(imageList)解析pts和人脸，并将其写入输出文件中
// 作者:张峰
// 时间:2018.11.30
// 备注:读入imagelist,输出imagelist, x, y ,w ,h ,x1,y1,x2,y2,x3,y3.......x68,y68
// 备注:MTCNN的人脸方法
/*******************************************************/
void genTxt(const string &inTxt, const string &outTxt) {
    Config &config = Config::GetInstance();
    int landmark_n = config.landmark_n;
    Mat_<double> gt_shape(landmark_n, 2);

    FILE *inFile = fopen(inTxt.c_str(), "r");
    FILE *outFile = fopen(outTxt.c_str(), "w");
    assert(inFile && outFile);

    char line[256];
    char buff[1000];
    string out_string("");
    int N = 0;
    while (fgets(line, sizeof(line), inFile)) {
        string img_path(line, strlen(line) - 1);
        std::cout<<"Handle "<<img_path<<std::endl;
		// pts和img仅后缀名不同
        string pts = img_path.substr(0, img_path.find_last_of(".")) + ".pts";

		//打开对应pts文件，前三行丢到,version n {
        FILE *tmp = fopen(pts.c_str(), "r");
        assert(tmp);
        fgets(line, sizeof(line), tmp);
        fgets(line, sizeof(line), tmp);
        fgets(line, sizeof(line), tmp);
        for (int i = 0; i < landmark_n; i++) {
            fscanf(tmp, "%lf", &gt_shape(i, 0));
            fscanf(tmp, "%lf", &gt_shape(i, 1));
        }
        fclose(tmp);

		//调用MTCNN人脸检测算法，得到矩形人脸区域
        Mat img = imread(img_path);
		vector<FaceInfo> faceInfo = detector.Detect(img, minSize, thresholds, factor, 3);
	
		// 判断人脸区域和给的pts的landmark区域是否匹配，因为有多个人脸
		Rect bbox = getBBox(img,faceInfo,gt_shape);

        if (bbox.x != -1) {
            N++;
            sprintf(buff, "%s %d %d %d %d", img_path.c_str(), bbox.x, bbox.y, bbox.width, bbox.height);
            out_string += buff;
            for (int i = 0; i < landmark_n; i++) {
                sprintf(buff, " %lf %lf", gt_shape(i, 0), gt_shape(i, 1));
                out_string += buff;
            }
            out_string += "\n";
        }
    }
    fprintf(outFile, "%d\n%s", N, out_string.c_str());

    fclose(inFile);
    fclose(outFile);
}

int prepare(void) {
    Config &params = Config::GetInstance();
    string txt = params.dataset + "/Path_Images_train.txt";
    genTxt(txt, params.dataset + "/train_mtcnn.txt");
    txt = params.dataset + "/Path_Images_test.txt";
    genTxt(txt, params.dataset + "/test_mtcnn.txt");
    return 0;
}
