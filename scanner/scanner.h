#ifndef SCANNER_H
#define SCANNER_H

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
                               int stride_factor=1,         //in : step factor, actual step size will be stride_factor*m_fhog_binsize
                               double threshold =0);         //in : detection threshold



        /* 
         * ===  FUNCTION  ======================================================================
         *         Name:  setParameters
         *  Description:  set parameters
         * =====================================================================================
         */
        bool setParameters( const int &fhog_binsize,                //in : fhog's binsize       
                            const int &fhog_orientation,            //in : fhog's number of orientation
                            const Size &target_size,                //in : target's size
                            const Size &padded_size,                //in : target's padded size( used to scan the image)
                            const vector<Mat> &weight_vector);      //in : svm's weight vector

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
                          int stride_factor=1,        //in : step factor, actual step size will be stride_factor*m_fhog_binsize
                          double threshold=0);             //in : serve as a threshold



        /* 
         * ===  FUNCTION  ======================================================================
         *         Name:  saveModel
         *  Description:  save the Model
         * =====================================================================================
         */
        bool saveModel( const string &path_to_save,      // in :  path
                        const string &info="") const;   // in : additional information
        

        /* 
         * ===  FUNCTION  ======================================================================
         *         Name:  loadModel
         *  Description:  load the model
         * =====================================================================================
         */
        bool loadModel( const string &path_to_load);        // in : path

        /* 
         * ===  FUNCTION  ======================================================================
         *         Name:  visualizeDetector
         *  Description:  visualize the learned detector, only using the 0-180 part of the 
         *                fhog feature.
         * =====================================================================================
         */
        bool visualizeDetector();


        /* 
         * ===  FUNCTION  ======================================================================
         *         Name:  loadModels
         *  Description:  load model files, they should have the same size of filter
         * =====================================================================================
         */
        bool loadModels( const vector<string> &model_files);    // in : paths of the model files
        

        /* 
         * ===  FUNCTION  ======================================================================
         *         Name:  setPad
         *  Description:  whether to pad the image or not
         * =====================================================================================
         */
        void setPad( bool pad_or_not );         // in : pad the image or not
     
    private:
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


        /* 
         * ===  FUNCTION  ======================================================================
         *         Name:  load_weight_to_filters
         *  Description:  seperate the weight vector to numbers of filter matrix, later use convolution
         *                to do the detection
         * =====================================================================================
         */
        bool load_weight_to_filters();



		/* 
		 * ===  FUNCTION  ======================================================================
		 *         Name:  form_filter_bank
		 *  Description:  decompose the m_filters into row filter and col filter if possible
		 *				  using seperable filter is way faster
		 * =====================================================================================
		 */
		bool form_filter_bank(
								const double relative_ratio_to_max = 0.001	/*  in : if a filter's singular value less than relative_ratio_to_max*max_singular_value
																					 it will be discarded, DO NOT set this too large*/
								);

        
        /* 
         * ===  FUNCTION  ======================================================================
         *         Name:  get_saliency_map
         *  Description:  Run one template on giving feature, return the saliency map
         * =====================================================================================
         */
        void get_saliency_map( const vector<Mat> &features_vec,         // in : input feature
                               const int template_index,                // in : index of the template
                               Mat &buffer_matirx,                      // in : buffer matrix shared between filters
                               Mat &saliency_map);                      // out: output saliency map( detect confidence)

        
        /* 
         * ===  FUNCTION  ======================================================================
         *         Name:  threshold_detection
         *  Description:  return the detect results above the threshold value
         * =====================================================================================
         */
        void threshold_detection( const vector<Mat> &saliency_maps,     // in : saliency_maps
                                  vector<Rect> &det_results,            // out: detect results            
                                  vector<double> &det_confs,           // out : detect confidence
                                  const double threshold);              // in : threshold


        /* 
         * ===  FUNCTION  ======================================================================
         *         Name:  adjust_detection
         *  Description:  adjust the detected Rect, scale back , crop the out-of-image part
         * =====================================================================================
         */
        void adjust_detection( Rect &det_result,        // in&out 
                               double scale,            // in : scale factor
                               const Mat &input_image); // in : input_image , used for border check


        /* 
         * ===  FUNCTION  ======================================================================
         *         Name:  pad_features
         *  Description:  pad the channel features with a fixed size( pad_size == 3 )
         * =====================================================================================
         */
        bool pad_features( vector<Mat> &feature_chns); // in&out: feature channels

        /*  Feature Part : fhog */
        int m_fhog_binsize;
        int m_fhog_orientation;
        
        /*  Target Part  */
        Size m_target_size;
        Size m_padded_size;

        /*  SVM's weight vector */
        vector<Mat> m_weight_vector;    // may contain several templates, one for frontal ,one for profile etc..
        int m_feature_dim;

        /*  Feature generator */
        feature_Pyramids m_feature_geneartor;

        /*  additional infos */
        string m_info;

        /*  filters, m_filters are the original filter kernel,
		 *  but if it's seperable, we'd like to use row_filter
		 *  and col_filter to reduce the computation, formed by
		 *  form_filter_bank() function */

        /*  firsr index is for number of template
         *  second index is for number of feature channel
         *  third index is for separable filter inside on feature channel*/
        vector<vector<Mat> > m_filters;
		vector<vector<vector<Mat> > > m_row_filters; 
		vector<vector<vector<Mat> > >m_col_filters;
        
        /* pad the feature map, so we can detect those targets near the border */
        bool m_pad_feature;
        int  m_border_size;
};
#endif
