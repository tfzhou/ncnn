#include <jni.h>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <string>
#include "hed_ncnn.hpp"
#include <android/log.h>

#ifdef __cplusplus
extern "C" {
#endif

    std::string jstring2string(JNIEnv *env, jstring jstr) {
        const char *cstr = env->GetStringUTFChars(jstr, 0);
        std::string str(cstr);
        env->ReleaseStringUTFChars(jstr, cstr);
        return str;
    }

    JNIEXPORT void JNICALL
    Java_com_argus_camera_generatedocument_card_1detection_EdgeModel_setNumThreads(JNIEnv *env, jobject thiz, jint numThreads) {
        int num_threads = numThreads;
        ncnn::HED::setNumThreads(num_threads);
    }

    JNIEXPORT jint JNICALL
        Java_com_argus_camera_generatedocument_card_1detection_EdgeModel_loadModel(JNIEnv *env, jobject thiz, jstring param_file, jstring model_file) {
            ncnn::HED::Get(jstring2string(env, param_file), jstring2string(env, model_file));
            return 0;
        }

    JNIEXPORT void JNICALL
        Java_com_argus_camera_generatedocument_card_1detection_EdgeModel_computeEdgeMap(JNIEnv *env, jobject thiz, jlong srcAddr, jlong dstAddr) {
            cv::Mat &src = *(cv::Mat*)srcAddr;
            cv::Mat &dst = *(cv::Mat*)dstAddr;

            ncnn::HED* hed = ncnn::HED::Get();
            dst = hed->Forward(src);
        }

#ifdef __cplusplus
}
#endif
