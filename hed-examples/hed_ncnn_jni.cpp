#include <jni.h>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <string>
#include "hed_ncnn.hpp"

#ifdef __cplusplus
extern "C" {
#endif

string jstring2string(JNIEnv *env, jstring jstr) {
    const char *cstr = env->GetStringUTFChars(jstr, 0);
    string str(cstr);
    env->ReleaseStringUTFChars(jstr, cstr);
    return str;
}

JNIEXPORT jint JNICALL
Java_com_lenovo_edgedetection_HED_loadModel(const std::string& param_file, const std::string& model_file) {
    HED::Get(jstring2string(env, param_file), jstring2string(env, model_file));
    return 0;
}

JNIEXPORT jobjectArray JNICALL
Java_com_lenovo_edgedetection_CaffeMobile_computeEdgeMap(JNIEnv *env, jobject thiz, jlong srcAddr, jlong dstAddr) {
        Mat &src = *(Mat*)srcAddr;
        Mat &dst = *(Mat*)dstAddr;

        HED* hed = HED::Get();
        dst = hed->Forward(src);

#ifdef __cplusplus
}
#endif
