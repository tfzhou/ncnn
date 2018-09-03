LOCAL_PATH := $(call my-dir)

NCNN_INSTALL_PATH := ../../build-android-aarch64/install

include $(CLEAR_VARS)
LOCAL_MODULE := ncnn
LOCAL_SRC_FILES := $(NCNN_INSTALL_PATH)/lib/libncnn.a
include $(PREBUILT_STATIC_LIBRARY)

include $(CLEAR_VARS)

LOCAL_C_INCLUDES += $(NCNN_INSTALL_PATH)/include
LOCAL_STATIC_LIBRARIES := ncnn

OPENCV_INSTALL_MODULES := on
OPENCV_LIB_TYPE := STATIC
include /home/lenovo/tfzhou/android/OpenCV-android-sdk/sdk/native/jni/OpenCV.mk

LOCAL_MODULE    := hed_ncnn_jni
LOCAL_SRC_FILES := hed_ncnn_jni.cpp hed_ncnn.cpp

LOCAL_CFLAGS := -O3 -std=c++11 -fvisibility=hidden -fomit-frame-pointer -fstrict-aliasing -ffunction-sections -fdata-sections -ffast-math -fopenmp
LOCAL_LDFLAGS += -Wl,--gc-sections -fopenmp

LOCAL_LDLIBS += -ldl -llog

include $(BUILD_SHARED_LIBRARY)
