#ifndef OPENCV_WARPPER_LIBSVM_H
#define OPENCV_WARPPER_LIBSVM_H

#include "opencv2/highgui/highgui.hpp"
#include "svm.h"

using namespace cv;

class opencv_warpper_libsvm
{
    public:
        opencv_warpper_libsvm();
        ~opencv_warpper_libsvm();

        /* 
         * ===  FUNCTION  ======================================================================
         *         Name:  fromMatToLibsvmNode
         *  Description:  convert the data form opencv Mat format to libsvm svm_node format
         * =====================================================================================
         */
        bool fromMatToLibsvmNode( const Mat &inputData,     // in  : only take CV_32F format, row samples. size : number_of_samples x feature_dimension
                                  svm_node **&nodesData );  // out : output libsvm svm_node format
};
#endif
