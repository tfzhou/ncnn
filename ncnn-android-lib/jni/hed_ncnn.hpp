#ifndef HED_NCNN_H
#define HED_NCNN_H

#include <string> 
#include <opencv2/core/core.hpp>

#include "include/net.h"

namespace ncnn {
    class HED {
        public:
            ~HED();

            static HED* Get();
            static HED* Get(const std::string& param_file, const std::string& model_file);
            static void setNumThreads(int num_threads) {_num_threads = num_threads;}

            cv::Mat Forward(const cv::Mat& im);

        private:
            static HED* _hed;
            static int _num_threads;
            ncnn::Net _net;

            HED(const std::string& param_file, const std::string& model_file);
    };
}

#endif
