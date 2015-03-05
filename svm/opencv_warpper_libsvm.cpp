#include <iostream>
#include "opencv2/highgui/highgui.hpp"
#include "opencv_warpper_libsvm.h"
#include "svm.h"

using namespace std;
using namespace cv;

opencv_warpper_libsvm::opencv_warpper_libsvm()
{

}

opencv_warpper_libsvm::~opencv_warpper_libsvm()
{

}

bool opencv_warpper_libsvm::fromMatToLibsvmNode(   const Mat &inputData,       // in : number_of_samples x feature_dim
                                                    svm_node **&nodesData )     //out : 
{
    int number_of_samples = inputData.rows;
    int feature_dim       = inputData.cols;

    if( inputData.empty() || inputData.type()!=CV_32F)
    {
        cout<<"inputData invalid, only support CV_32F "<<endl;
        return false;
    }
    /*  allocate the memory for nodesData, remember to delete it after training */
    nodesData = new svm_node*[ number_of_samples ];

    for ( unsigned int sample_index=0; sample_index < number_of_samples ; sample_index++) 
    {
        nodesData[sample_index] = new svm_node[ feature_dim + 1];   //last node should have index = -1 indicates ending
        const float *input_sample_ptr = static_cast<const float*>( inputData.ptr<float>(sample_index) );

        int feature_count = 0;
        for( unsigned int fea_index = 0; fea_index < feature_dim; fea_index++)
        {
            if( input_sample_ptr[fea_index] == 0 )          // skip when the value is exact zero, libsvm use sparse representation, default is 0
                continue;
            nodesData[sample_index][feature_count].index = fea_index+1;             // !! svm_node index start with 1
            nodesData[sample_index][feature_count].value = input_sample_ptr[fea_index];
            feature_count++;
        }
        /* set -1 to the rest */
        while( feature_count < feature_dim+1 )
            nodesData[sample_index][feature_count++].index = -1;

    }
    return true;
}

