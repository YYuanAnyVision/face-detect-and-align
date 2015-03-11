#ifndef SCANNER_H
#define SCANNER_h

#include <vector>
#include "opencv2/highgui/highgui.hpp"
#include "../chnfeature/Pyramid.h"

using namespace std;
using namespace cv;

class scanner
{
    public:
        scanner();
        ~scanner();
        

        /* 
         * ===  FUNCTION  ======================================================================
         *         Name:  detectMuitiScale
         *  Description:  detect target using slide fhog + linear svm, return true if no error occus
         * =====================================================================================
         */
        bool detectMultiScale( const Mat &input_image,      //in : input image
                               vector<Rect> &results,       //out: output targets' position
                               vector<double> &confidence,  //out: targets' confidence
                               const Size &minSize,         //in : min target size 
                               const Size &maxSize,         //in : max target size
                               double scale_factor,         //in : factor to scale the image
                               int stride_factor=1);        //in : step factor, actual step size will be stride_factor*m_fhog_binsize



        /* 
         * ===  FUNCTION  ======================================================================
         *         Name:  setParameters
         *  Description:  set parameters
         * =====================================================================================
         */
        bool setParameters( const int &fhog_binsize,        //in : fhog's binsize       
                            const int &fhog_orientation,    //in : fhog's number of orientation
                            const Size &target_size,        //in : target's size
                            const Size &padded_size,        //in : target's padded size( used to scan the image)
                            const Mat &weight_vector);      //in : svm's weight vector

        /* 
         * ===  FUNCTION  ======================================================================
         *         Name:  checkParameter
         *  Description:  return true if all the parameters are valid
         * =====================================================================================
         */
        bool checkParameter() const;


        /* 
         * ===  FUNCTION  ======================================================================
         *         Name:  slide_image
         *  Description:  use slide window to detect target in a fixed scale image
         * =====================================================================================
         */
        bool slide_image( const Mat &input_img,        // in: input image
                          vector<Rect> &results,        //out: output targets' position
                          vector<double> &confidence,   //out: targets' confidence
                          int stride_factor=1);        //in : step factor, actual step size will be stride_factor*m_fhog_binsize


    private:

        /* 
         * ===  FUNCTION  ======================================================================
         *         Name:  get_score
         *  Description:  get the svm score in position (x,y)
         * =====================================================================================
         */
        float get_score( const vector<Mat> &feature_chns,       // in : input feature channels
                         const int &x,                          // in : position in x direction
                         const int &y,                          // in : position in y direction
                         const int &slide_width,                // in : slide target's width in feature map
                         const int &slide_height);              // in : slide target's height in feature map


        /* 
         * ===  FUNCTION  ======================================================================
         *         Name:  get_scale_vector
         *  Description:  givin the maxSize, minSize and scale factor, return the scale vector
         *                in order to resize the image
         * =====================================================================================
         */
        bool get_scale_vector(  const Size &img_size,           // in : used to define the min_scale
                                const Size &minSize,            // in : minSize
                                const Size &maxSize,            // in : maxSize
                                double scale_factor,            // in : scale factor
                                vector<double> &scale_vec       // out: scale vector
                             ) const;


        /*  Feature Part : fhog */
        int m_fhog_binsize;
        int m_fhog_orientation;
        
        /*  Target Part  */
        Size m_target_size;
        Size m_padded_size;

        /*  SVM's weight vector */
        Mat m_weight_vector;
        float *m_weight;        //pointer to the data of m_weight_vector
        int m_feature_dim;

        /*  Feature generator */
        feature_Pyramids m_feature_geneartor;
};
#endif
