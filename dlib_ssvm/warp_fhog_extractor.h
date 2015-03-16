#ifndef WARP_FHOG_EXTRACTOR_H
#define WARP_FHOG_EXTRACTOR_H
#include <vector>
#include <dlib/geometry.h>
#include <dlib/array2d.h>
#include <dlib/image_processing/object_detector_abstract.h>
#include "opencv2/highgui/highgui.hpp"

#include <dlib/opencv.h>
/* 
  * implement the interface defined in default_fhog_feature_extractor, so that we can use our own
  * fhog implementation in dlib's  <scan_fhog_pyramid>
  * 
  */
class warp_fhog_extractor
{
    /*!
    WHAT THIS OBJECT REPRESENTS
        The scan_fhog_pyramid object defined below is primarily meant to be used
        with the feature extraction technique implemented by extract_fhog_features().  
        This technique can generally be understood as taking an input image and
        outputting a multi-planed output image of floating point numbers that
        somehow describe the image contents.  Since there are many ways to define
        how this feature mapping is performed, the scan_fhog_pyramid allows you to
        replace the extract_fhog_features() method with a customized method of your
        choosing.  To do this you implement a class with the same interface as
        default_fhog_feature_extractor.       
        */
    public:
        dlib::rectangle image_to_feats (
            const dlib::rectangle& rect,       
            int cell_size,              
            int filter_rows_padding,    
            int filter_cols_padding     
            ) const
        {
            dlib::point top_left  = rect.tl_corner()/cell_size - dlib::point((filter_cols_padding-1)/2,(filter_rows_padding-1)/2);
            dlib::point bot_right = rect.br_corner()/cell_size - dlib::point((filter_cols_padding-1)/2,(filter_rows_padding-1)/2);
            return dlib::rectangle( top_left, bot_right);

        }
            /*!
                requires
                    - cell_size > 0
                    - filter_rows_padding > 0,  equals one means no padding
                    - filter_cols_padding > 0
                ensures
                    - Maps a rectangle from the coordinates in an input image to the corresponding
                      area in the output feature image.
            !*/

        dlib::rectangle feats_to_image (
            const dlib::rectangle& rect,
            int cell_size,
            int filter_rows_padding,
            int filter_cols_padding
            ) const
        {
            dlib::point top_left  = rect.tl_corner()*cell_size - dlib::point((filter_cols_padding-1)/2,(filter_rows_padding-1)/2)*cell_size;
            dlib::point bot_right = rect.br_corner()*cell_size - dlib::point((filter_cols_padding-1)/2,(filter_rows_padding-1)/2)*cell_size;
            return dlib::rectangle( top_left, bot_right );
        }
            /*!
                requires
                    - cell_size > 0
                    - filter_rows_padding > 0
                    - filter_cols_padding > 0
                ensures
                    - Maps a rectangle from the coordinates of the hog feature image back to
                      the input image.
                    - Mapping from feature space to image space is an invertible
                      transformation.  That is, for any rectangle R we have:
                        R == image_to_feats(feats_to_image(R,cell_size,filter_rows_padding,filter_cols_padding),
                                                             cell_size,filter_rows_padding,filter_cols_padding).
            !*/

             template <typename image_type>
            void operator()(
                image_type& img,                      // in  : input image 
                dlib::array<dlib::array2d<float> >& hog,    // out : output fhog feature map
                int cell_size,                              // in : cell_size , eg 8 
                int filter_rows_padding,                    // in : padding ~~ eg 1( no padding )
                int filter_cols_padding                     // in :
            ) const
            /*!
                requires
                    - image_type == is an implementation of array2d/array2d_kernel_abstract.h
                    - img contains some kind of pixel type. 
                      (i.e. pixel_traits<typename image_type::type> is defined)
                ensures
                    - Extracts FHOG features by calling extract_fhog_features().  The results are
                      stored into #hog.  Note that if you are implementing your own feature extractor you can
                      pretty much do whatever you want in terms of feature extraction so long as the following
                      conditions are met:
                        - #hog.size() == get_num_planes()
                        - Each image plane in of #hog has the same dimensions.
                        - for all valid i, r, and c:
                            - #hog[i][r][c] == a feature value describing the image content centered at the 
                              following pixel location in img: 
                                feats_to_image(point(c,r),cell_size,filter_rows_padding,filter_cols_padding)
            !*/
            
            {
                   const cv::Mat mat_img  = dlib::toMat( img );
                   std::cout<<"img's size is "<<mat_img.size()<<" channel : "<<mat_img.channels()<<std::endl;
            }
            inline unsigned long get_num_planes (
            ) const { return 31; }
                   /*!
                       ensures
                           - returns the number of planes in the hog image output by the operator()
                             method.
                   !*/
            
};

#endif
