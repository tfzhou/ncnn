#include <stdio.h>
#include <iostream>
#include <string>
#include <algorithm>
#include <vector>
#include <fstream>
#include <opencv2/opencv.hpp>

#include "timer.hpp"
#include "net.h"

//#define BATCH

const int MAX_LEN = 640;

cv::Mat computeEdgeMap(const cv::Mat& src) {
    ncnn::Net hednet;
    //hednet.load_param("vgg16-30000.param");
    //hednet.load_model("vgg16-30000.bin");
    hednet.load_param("resnet-20000.param");
    hednet.load_model("resnet-20000.bin");

    ncnn::Mat in = ncnn::Mat::from_pixels(src.data, ncnn::Mat::PIXEL_BGR, src.cols, src.rows);

    const float mean_vals[3] = {104.00698793, 116.66876762, 122.67891434};
    in.substract_mean_normalize(mean_vals, 0);

    ncnn::Extractor ex = hednet.create_extractor();
    ex.set_light_mode(true);
    ex.input("data", in);

    ncnn::Mat out;
    ex.extract("sigmoid_fuse", out);

    return cv::Mat(src.size(), CV_32F, out.data).clone();
}

int main(int argc, char** argv)
{
#if defined(BATCH)
    if(argc != 3) {
        std::cerr << "./hed test.lst save_dir" << std::endl;
        exit(1);
    }
    std::ifstream infile(argv[1]);
    std::string filename;

    Timer timer;

    while(infile >> filename) {
        std::cout << filename << std::endl;
        cv::Mat m = cv::imread(filename, CV_LOAD_IMAGE_COLOR);

        int newH, newW;
        if(m.rows > m.cols) {
            newH = MAX_LEN;
            newW = int(newH * 1.0 / m.rows * m.cols);
        }
        else {
            newW = MAX_LEN;
            newH = int(newW * 1.0 / m.cols* m.rows);
        }

        cv::resize(m, m, cv::Size(newW, newH), cv::INTER_AREA);
        
        timer.start();
        cv::Mat edgeMap = computeEdgeMap(m);
        timer.stop();
        edgeMap.convertTo(edgeMap, CV_8U, 255.0);

        size_t pos = filename.find_last_of("/");
        std::string fn = filename.substr(pos+1);
        cv::imwrite(argv[2] + fn, edgeMap);
    }
#else
    const char* imagepath = argv[1];

    cv::Mat m = cv::imread(imagepath, CV_LOAD_IMAGE_COLOR);

    cv::Mat edgeMap = computeEdgeMap(m);
    cv::imshow("x", edgeMap);
    cv::waitKey();

    edgeMap.convertTo(edgeMap, CV_8U, 255.0);
    cv::imwrite("/tmp/edgemap.png", edgeMap);
#endif

    return 0;
}
