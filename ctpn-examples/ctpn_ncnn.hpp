#ifndef CTPN_NCNN_H
#define CTPN_NCNN_H

#include <string>
#include <iostream>
#include <vector>
using namespace std;

#include <opencv2/core/core.hpp>

#include "net.h"

class CtpnNcnn {
    public:
        ~CtpnNcnn();

        static CtpnNcnn* Get();
        static CtpnNcnn* Get(const string& paramFile, const string& binFile);
        static void setNumThreads(int nThreads) { _nThreads = nThreads; }

        vector<cv::Rect> Forward(const cv::Mat& inputImage);

    private:
        bool init(const string& paramFile, const string& binFile);
        bool clear();
        vector<cv::Rect> predict(const cv::Mat& inputImage);

        float computeScaleFactor(const cv::Mat& inputImage);
        void postprocess(int rows, int cols, vector<float>& scores, vector< vector<float> >& rois, vector< vector<float> >& textlines);

        CtpnNcnn(const string& paramFile, const string& binFile);

    private:
        static CtpnNcnn* ctpn;
        static int _nThreads;
        ncnn::Net net;

        const float minScale {600.};
        const float maxScale {1000.};
        const int numAnchors {10};
};

inline float threshold(float x, int min_, int max_) {
	float t = x>max_?max_:x;
	return t>min_?t:min_;
}

cv::Mat resizeImage(const cv::Mat&);
bool apply_deltas_to_anchors(vector< vector<float> >& res, vector<float>& bb_deltas, vector<float>& scores, int stride, int height, int width, int imgh, int imgw, float min_score );

#endif
