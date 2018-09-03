#include <string.h>
#include <jni.h>
#include <android/log.h>
#include <string>
#include <vector>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "ctpn_ncnn.hpp"


#ifdef __cplusplus
extern "C" {
#endif

    using std::string;
    using std::vector;
    using namespace cv;

    int getTimeSec() {
        struct timespec now;
        clock_gettime(CLOCK_MONOTONIC, &now);
        return (int)now.tv_sec;
    }

    string jstring2string(JNIEnv *env, jstring jstr) {
        const char *cstr = env->GetStringUTFChars(jstr, 0);
        string str(cstr);
        env->ReleaseStringUTFChars(jstr, cstr);
        return str;
    }

    string bytes2string(JNIEnv *env, jbyteArray buf) {
        jbyte *ptr = env->GetByteArrayElements(buf, 0);
        string s((char *)ptr, env->GetArrayLength(buf));
        env->ReleaseByteArrayElements(buf, ptr, 0);
        return s;
    }

    cv::Mat imgbuf2mat(JNIEnv *env, jbyteArray buf, int width, int height) {
        jbyte *ptr = env->GetByteArrayElements(buf, 0);
        cv::Mat img(height + height / 2, width, CV_8UC1, (unsigned char *)ptr);
        cv::cvtColor(img, img, CV_YUV2RGBA_NV21);
        env->ReleaseByteArrayElements(buf, ptr, 0);
        return img;
    }

    cv::Mat getImage(JNIEnv *env, jbyteArray buf, int width, int height) {
        return (width == 0 && height == 0) ? cv::imread(bytes2string(env, buf), -1)
            : imgbuf2mat(env, buf, width, height);
    }

    inline void vector_Rect_to_Mat(vector<cv::Rect>& v_rect, cv::Mat& mat)
    {
        mat = cv::Mat(v_rect, true);
    }

    JNIEXPORT jint JNICALL Java_com_lenovo_ctpn_CTPN_loadModel(JNIEnv *env, jobject thiz, jstring modelPath, jstring weightsPath) {
        __android_log_write(ANDROID_LOG_INFO, "ctpn", "LoadModel Start!");
        CtpnNcnn::Get(jstring2string(env, modelPath), jstring2string(env, weightsPath));
        __android_log_write(ANDROID_LOG_INFO, "ctpn", "LoadModel Succeed!");
        return 0;
    }

    JNIEXPORT void JNICALL Java_com_lenovo_ctpn_CTPN_predict(JNIEnv *env, jobject thiz, jlong srcAddr, jlong dstAddr) {
        Mat &src = *(Mat*)srcAddr;
        Mat &dst = *(Mat*)dstAddr;

        CtpnNcnn* ctpn = CtpnNcnn::Get();
        vector<Rect> textlines = ctpn->Forward(src);
        
        vector_Rect_to_Mat(textlines, dst);
    }

#ifdef __cplusplus
}
#endif

