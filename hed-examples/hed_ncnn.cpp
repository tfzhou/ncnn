#include <stdio.h>
#include <iostream>
#include <algorithm>
#include <vector>
#include <string>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "hed_ncnn.hpp"

namespace ncnn {

    HED* HED::_hed = 0;

    HED* HED::Get() {
        return _hed;
    }

    HED* HED::Get(const std::string& param_file, const std::string& model_file) {
        if(!_hed) {
            _hed = new HED(param_file, model_file);
        }

        return _hed;
    }

    HED::HED(const std::string& param_file, const std::string& model_file) {
        _net.load_param(param_file.c_str());
        _net.load_model(model_file.c_str());
    }

    HED::~HED() {}

    cv::Mat HED::Forward(const cv::Mat& bgr) {

        ncnn::Mat in = ncnn::Mat::from_pixels(bgr.data, ncnn::Mat::PIXEL_BGR, bgr.cols, bgr.rows);
        ncnn::Mat out;

        const float mean_vals[3] = {104.00698793, 116.66876762, 122.67891434};
        in.substract_mean_normalize(mean_vals, 0);

        Extractor ex = _net.create_extractor();
        ex.set_light_mode(true);
        ex.input("data", in);
        ex.extract("sigmoid_fuse", out);

        return cv::Mat(bgr.size(), CV_32FC1, out.data).clone();
    }
}

int main(int argc, char* argv[]) {
    std::string model_file = "hed_ft_iter_100000_ncnn.bin";
    std::string param_file = "deploy_ncnn.param";

    cv::Mat im = cv::imread(argv[1]);

    int maxW, maxH;
    if(im.rows > im.cols) {
        maxH = 600;
        maxW = im.cols * maxH / im.rows;
    }
    else {
        maxW = 600;
        maxH = im.rows * maxW / im.cols;
    }
    cv::resize(im, im, cv::Size(maxW, maxH));

    ncnn::HED *hed =  ncnn::HED::Get(param_file, model_file);
    cv::Mat output = hed->Forward(im);

    cv::imshow("output", output);
    cv::waitKey();

    return 0;
}
