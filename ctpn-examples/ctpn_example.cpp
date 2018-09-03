#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "ctpn_ncnn.hpp"

using namespace std;

int main(int argc, char** argv) {
    if(argc != 4) {
        cout << "usage: ./ctpn <image file> <param file> <bin file>" << endl;
        exit(1);
    }

    cv::Mat im = cv::imread(argv[1]);

    CtpnNcnn* ctpn = CtpnNcnn::Get(argv[2], argv[3]);
    vector<cv::Rect> boxes = ctpn->Forward(im);

    for(auto box : boxes) {
        cv::rectangle(im, box, cv::Scalar(0,0,255), 3);
    }

    cv::resize(im, im, cv::Size(0,0), 0.4, 0.4);
    cv::imshow("im", im);
    cv::waitKey();
}
