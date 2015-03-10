#ifndef OPENCV_WARPPER_LIBSVM_H
#define OPENCV_WARPPER_LIBSVM_H

#include <string>
#include "opencv2/highgui/highgui.hpp"
#include "svm.h"

using namespace cv;
using namespace std;

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
        bool fromMatToLibsvmNode( const Mat &inputData,             // in  : only take CV_32F format, row samples. size : number_of_samples x feature_dimension
                                  svm_node **&nodesData ) const;    // out : output libsvm svm_node format


        /* 
         * ===  FUNCTION  ======================================================================
         *         Name:  setSvmParameters
         *  Description:  set the parameters of the svm
         * =====================================================================================
         */
        bool setSvmParameters( const svm_parameter &paras);     // in : svm paras


        /* 
         * ===  FUNCTION  ======================================================================
         *         Name:  getSvmParameters
         *  Description:  return the parameter of the svm model
         * =====================================================================================
         */
        svm_parameter getSvmParameters();

        

        /* 
         * ===  FUNCTION  ======================================================================
         *         Name:  train
         *  Description:  train the svm model , should set the svm_parameter first
         * =====================================================================================
         */
        bool train( const Mat &positive_data,       //in : positive data, row samples, number_of_samples x feature_dim
                    const Mat &negative_data,       //in : negative data, row samples, number_of_samples x feature_dim
                    const string &path_to_save);    //in : where to save the model file


        /* 
         * ===  FUNCTION  ======================================================================
         *         Name:  predict
         *  Description:  predict giving feature, output value will be label for C-svm, or probability
         *                if probability is set 1 during training 
         * =====================================================================================
         */
        bool predict( const Mat &input_data,            // in : input feature, row format, CV_32F, number_of_sample x feature_dim
                      Mat &predict_value) const;        //out : CV_32F, number_of_samples x 1



        /* 
         * ===  FUNCTION  ======================================================================
         *         Name:  extract_weight_vector
         *  Description:  for linear svm model , extract the weight vector, use it directly will boost the speed
         * =====================================================================================
         */
        bool extract_weight_vector();


        /* 
         * ===  FUNCTION  ======================================================================
         *         Name:  predict_linear
         *  Description:  predict the giving data using extracted linear weight vector
         * =====================================================================================
         */
        bool predict_linear( const Mat &input_data,     // in : input feature, row format, CV_32F, number_of_sample x feature_dim
                             Mat &predict_value) const; // out: CV_32F, number_of_samples x 1  

        

        /* 
         * ===  FUNCTION  ======================================================================
         *         Name:  predict_general
         *  Description:  predict using libsvm's predict function, general case
         * =====================================================================================
         */
        bool predict_general( const Mat &input_data,    // in : input feature, row format, CV_32F, number_of_sample x feature_dim
                              Mat &predict_value) const; // out: CV_32F, number_of_samples x 1  
        /* 
         * ===  FUNCTION  ======================================================================
         *         Name:  get_weight_vector
         *  Description:  return the linear svm weight vector in opencv format
         * =====================================================================================
         */
        Mat get_weight_vector() const;

    private:
        float get_feature_weight_dot_value( const float *feature ) const;    

        svm_parameter m_svm_para;               // parameter of svm
        svm_problem m_svm_training_data;        // training data
        svm_model *m_model;                     // svm model
        float *m_weight_vector_for_linearsvm;   // extracted weight vector for linear svm model
        int m_feature_dim;                      // dimension of the feature
};
#endif
