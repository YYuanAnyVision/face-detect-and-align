#include <iostream>
#include <cmath>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/contrib/contrib.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include "../chnfeature/Pyramid.h"
#include "../misc/NonMaxSupress.h"

#include "../chnfeature/sseFun.h"

#include "scanner.h"

using namespace cv;
using namespace std;


scanner::scanner()
{
    m_fhog_orientation = 9;
    m_fhog_binsize = 8;
    m_number_of_seperable_filters = 0;
}

scanner::~scanner()
{

}


bool scanner::visualizeDetector()
{
    if(!checkParameter())
        return false;
    
    /*  the struct of the fhog feature */
    /* 
     *    2*m_fhog_orientation  --> sensitive orientation channel, 0-2pi
     *    m_fhog_orientation    --> unsensitive orientation channel, 0-pi
     *    4 texture channel
     *    1 zero pad
     *
     *    extract the unsensitive orientation channel and show it
     * */
     vector<Mat> unsentivechannels;
     const float *weight_data = (const float*)(m_weight_vector.data);

     for( unsigned int c=0;c<m_fhog_orientation;c++)
     {
        Mat tmp_channel = Mat::zeros( m_padded_size.height/m_fhog_binsize, m_padded_size.width/m_fhog_binsize, CV_32F);
        /*  the weigth on the same orientation bin */
        const float *weight_pointer1 = weight_data+(c+2*m_fhog_orientation)*( tmp_channel.rows*tmp_channel.cols);
        const float *weight_pointer2 = weight_data+(c+m_fhog_orientation)*( tmp_channel.rows*tmp_channel.cols);
        const float *weight_pointer3 = weight_data+(c)*( tmp_channel.rows*tmp_channel.cols);

        for( int r_index=0;r_index<tmp_channel.rows;r_index++)
        {
            for( int c_index=0;c_index<tmp_channel.cols;c_index++)
            {
                float weight_value = *(weight_pointer1++)+*(weight_pointer2++)+*(weight_pointer3++);
                tmp_channel.at<float>(r_index, c_index) =( weight_value>0?weight_value:0);
            }
        }
        unsentivechannels.push_back( tmp_channel );
     }
     Mat draw;
     m_feature_geneartor.visualizeHog(unsentivechannels, draw, 20, 0.1);
     imshow("detector", draw);
     waitKey(0);
     return true;
}

bool scanner::slide_image( const Mat &input_img,		// in: input image
                           vector<Rect> &results,       //out: output targets' position
                           vector<double> &confidence,	//out: targets' confidence
                           int stride_factor,           //in : step factor, actual step size will be stride_factor*m_fhog_binsize
                           double threshold)            //in : threshold
{
    /*  Compute the fhog feature  */
    vector<Mat> feature_chns;
    Mat computed_feature;
    m_feature_geneartor.fhog( input_img, computed_feature, feature_chns, 0, m_fhog_binsize, m_fhog_orientation, 0.2); // 0 -> fhog, 0.2 -> clip value
    
    /* remove the all zero channel */
    feature_chns.resize( feature_chns.size()-1);

    /*  compute useful constant  中国好注释*/
    /*      
     | ------ feature_width ------- |
     --------------x----------------     -----
    |   |- slide_width--|           |      |
    |    ---------------  --        |      |
    |   |               |           |      |
 1  y   |            slide_height   |  feature_heigth 
    |   |               |           |      |
    |    ---------------  --        |      |
    |                               |      |
     -------------------------------     -----
    |                               |
    |    ---------------            |      
    |   |               |           |
 2  |   |               |           |
    |   |               |           |
    |    ---------------            |
    |                               |
     -------------------------------
            .
            .
            .
     -------------------------------
    |                               |
    |    ---------------            |      
    |   |               |           |
 n  |   |               |           |
    |   |               |           |
    |    ---------------            |
    |                               |
     -------------------------------
    */
    const int number_of_channels = feature_chns.size();         // equals n in above figure
    /*  see figure above */
    const int feature_width  = feature_chns[0].cols;
    const int feature_heigth = feature_chns[0].rows;
    const int slide_width  = m_padded_size.width/m_fhog_binsize;
    const int slide_height = m_padded_size.height/m_fhog_binsize;

    /*  store the detect confidence */
    if( m_filters.size() != feature_chns.size())
    {
        cout<<"number of channels should equal the number of filters "<<endl;
        return false;
    }

    Mat saliency_map = Mat::zeros( feature_chns[0].size(), CV_32F);
    Mat tmp_saliency;

    /*  1 original convolution */
    for(unsigned int c=0;c<m_filters.size();c++)
    {
        cv::filter2D( feature_chns[c], tmp_saliency, CV_32F, m_filters[c] );
        saliency_map = saliency_map + tmp_saliency;
    }

    /*  2 seperable convolution */
    //for(unsigned int i=0;i<m_row_filters.size();i++)
    //{
    //    for(unsigned int j=0;j<m_row_filters[i].size();j++)
    //    {
    //        cv::filter2D( feature_chns[i], tmp_saliency, CV_32F, m_col_filters[i][j]);
    //        cv::filter2D( tmp_saliency, tmp_saliency, CV_32F, m_row_filters[i][j]);
    //        saliency_map = saliency_map + tmp_saliency;
    //    }
    //}

    /*  add the bias term */
    saliency_map = saliency_map + m_weight_vector.at<float>(m_feature_dim,0);
    for(unsigned int r=0;r<saliency_map.rows;r++)
    {
        for(unsigned int c=0;c<saliency_map.cols;c++)
        {
            if(saliency_map.at<float>(r,c) > threshold)
            {
                results.push_back(cv::Rect( (c-slide_width/2)*m_fhog_binsize, 
                                            (r-slide_height/2)*m_fhog_binsize, 
                                            m_padded_size.width, 
                                            m_padded_size.height));
                confidence.push_back(saliency_map.at<float>(r,c));
            }
        }
    }

    return true;
}

bool scanner::detectMultiScale( const Mat &input_image,      //in : input image
                                vector<Rect> &results,       //out: output targets' position
                                vector<double> &confidence,  //out: targets' confidence
                                const Size &minSize,         //in : min target size 
                                const Size &maxSize,         //in : max target size
                                double scale_factor,         //in : factor to scale the image
                                int stride_factor,           //in : step factor, actual step size will be stride_factor*m_fhog_binsize
                                double threshold)            //in : threshold of detection
{
    if( !checkParameter())
        return false;

    /*  compute the scales we will work on  */
    vector<double> scale_vec;
    get_scale_vector( input_image.size(),  minSize ,maxSize, scale_factor, scale_vec);

    /* detect target in each scale */
    Mat processing_image;
    results.clear();
    confidence.clear();
    for( unsigned int scale_index=0;scale_index<scale_vec.size();scale_index++)
    {
        vector<Rect> det_results;
        vector<double> det_confs;
        resize( input_image, processing_image, Size(0,0), scale_vec[scale_index], scale_vec[scale_index]);

        /*  compute stride in each scale */
        int s_stride = int(stride_factor*scale_vec[scale_index]);
        s_stride = s_stride < 1?1:s_stride;
        slide_image( processing_image, det_results, det_confs, s_stride, threshold);
        
        for( unsigned int c=0;c<det_results.size();c++)
        {
            if( det_confs[c] < threshold)
                continue;

            Rect tmp_target = det_results[c];
            tmp_target.x /= scale_vec[scale_index];
            tmp_target.y /= scale_vec[scale_index];
            tmp_target.width /= scale_vec[scale_index];
            tmp_target.height /= scale_vec[scale_index];
            
            /*  convert from padded_size to target_size */
            tmp_target.x += tmp_target.width*1.0*(m_padded_size.width-m_target_size.width)/(2*m_padded_size.width);
            tmp_target.y += tmp_target.height*1.0*(m_padded_size.height-m_target_size.height)/(2*m_padded_size.height);
            tmp_target.width *= 1.0*m_target_size.width/m_padded_size.width;
            tmp_target.height *= 1.0*m_target_size.height/m_padded_size.height;

            /*  sometimes the detected result will be slightly out of the image , crop it */
            if( tmp_target.x < 0)
                tmp_target.x = 0;
            if( tmp_target.y < 0)
                tmp_target.y = 0;
            if( tmp_target.x + tmp_target.width > input_image.cols)
                tmp_target.width = input_image.cols - tmp_target.x -1;
            if( tmp_target.y + tmp_target.height > input_image.rows)
                tmp_target.height = input_image.rows - tmp_target.y - 1;

            results.push_back( tmp_target);
            confidence.push_back( det_confs[c] );
        }
    }
    
    /*  group the detected results */
    NonMaxSupress( results, confidence );
    return true;
}

bool scanner::get_scale_vector( const Size &img_size,           // in : used to define the min_Scale
                                const Size &minSize,            // in : minSize
                                const Size &maxSize,            // in : maxSize
                                double scale_factor,            // in : scale factor
                                vector<double> &scale_vec       // out: scale vector
                              ) const 
{
    if( minSize.width>maxSize.width || minSize.height > maxSize.height )
    {
        cout<<"Make sure maxSize > minSize "<<endl;
        return false;
    }
    if( scale_factor < 1 )
    {
        cout<<"Scale_factor should larger than 1 "<<endl;
        return false;
    }

    /*  use height to compute the scale vector, "height" is more robust in image than "width" */
    double max_scale = m_target_size.height*1.0 / minSize.height;
    double min_scale = m_target_size.height*1.0 / maxSize.height;

    /*  also we should keep the image is larger than the slide_window(size m_padded_size) */
    double img_min_scale = max( 1.0*m_padded_size.width/ img_size.width, 1.0*m_padded_size.height/ img_size.height);
    
    double current_scale = max( min_scale, img_min_scale) ;
    while( current_scale < max_scale  )
    {
        scale_vec.push_back( current_scale );
        current_scale *= scale_factor;
    }

    return true; 
}

bool scanner::setParameters( const int &fhog_binsize,        //in : fhog's binsize       
                             const int &fhog_orientation,    //in : fhog's number of orientation
                             const Size &target_size,        //in : target's size
                             const Size &padded_size,        //in : target's padded size( used to scan the image)
                             const Mat &weight_vector)       //in : svm's weight vector
{
    m_fhog_binsize = fhog_binsize;
    m_fhog_orientation = fhog_orientation;

    m_target_size = target_size;
    m_padded_size = padded_size;

    weight_vector.copyTo( m_weight_vector);

    /*  Set the pointer the the data */
    if( weight_vector.type() != CV_32F)
        return false;
    m_weight = (float*)( m_weight_vector.data);
    /*  dimension of m_weight_vector is feature_dim + 1 ( bias term is not counted as feature_dim ) */
    m_feature_dim = (m_weight_vector.rows > m_weight_vector.cols? m_weight_vector.rows: m_weight_vector.cols) - 1;

    /*  check the parameters */
    if(!checkParameter())
    {
        m_weight = NULL;
        return false;
    }
    return true;
}

bool scanner::checkParameter() const
{
    if( m_fhog_binsize <=0 || m_fhog_orientation <=0 )
    {
        cout<<"Check the binsize and orientation"<<endl;
        return false;
    }

    if( m_padded_size.width < m_target_size.width ||
            m_padded_size.height < m_target_size.height ||
            m_target_size.width <=0 || m_target_size.height <=0)
    {
        cout<<"Check the target_size and padded_size "<<endl;
        return false;
    }

    if( m_weight_vector.empty() || !m_weight_vector.isContinuous() || m_weight_vector.type()!=CV_32F)
    {
        cout<<"Weight vector should be continuous and format shoule be CV_32F"<<endl;
        return false;
    }

    if( !m_weight )
    {
        cout<<"m_weight is not set "<<endl;
        return false;
    }

    if( m_weight_vector.cols!=1 && m_weight_vector.rows!=1 )
    {
        cout<<"Weight vector should be one column or one row "<<endl;
        return false;
    }
    return true;
}



/* 
 * ===  FUNCTION  ======================================================================
 *         Name:  saveModel
 *  Description:  save the Model
 * =====================================================================================
 */
bool scanner::saveModel( const string &path_to_save,    // in : path
                         const string &infos ) const    // in : informations
{
    if(!checkParameter())
    {
        cout<<"Model is invalid .."<<endl;
        return false;
    }

    FileStorage fs( path_to_save, FileStorage::WRITE);
    if( !fs.isOpened())
    {
        cout<<"Can not open the file "<<path_to_save<<endl;
        return false;
    }
    
    fs<<"fhog_binsize"<<m_fhog_binsize;
    fs<<"fhog_orientation"<<m_fhog_orientation;
    fs<<"target_size"<<m_target_size;
    fs<<"padded_size"<<m_padded_size;
    fs<<"infos"<<infos;
    fs<<"weight_vector"<<m_weight_vector;
    
    return true;
}


/* 
 * ===  FUNCTION  ======================================================================
 *         Name:  loadModel
 *  Description:  load the model
 * =====================================================================================
 */
bool scanner::loadModel( const string &path_to_load)        // in : path
{
    FileStorage fs( path_to_load, FileStorage::READ);
    if( !fs.isOpened())
    {
        cout<<"Can not open file "<<path_to_load<<endl;
        return true;
    }
    
    Mat weight_vector;
    Size padded_size;
    Size target_size;
    int feature_dim;
    int fhog_binsize;
    int fhog_orientation;

    fs["weight_vector"]>>weight_vector;
    fs["padded_size"]>>padded_size;
    fs["fhog_binsize"]>>fhog_binsize;
    fs["fhog_orientation"]>>fhog_orientation;
    fs["target_size"]>>target_size;
    fs["infos"]>>m_info;

    setParameters( fhog_binsize, fhog_orientation, target_size, padded_size, weight_vector);
    if( !checkParameter())
    {
        cout<<"Parameters wrong "<<endl;
        return false;
    }
    
    /*  load the weight to filter matrix */
    if(!load_weight_to_filters())
    {
        cout<<"Can not load_weight_to_filters "<<endl;
        return false;
    }

    /* form the filter bank */
    //if(!form_filter_bank())
    //{
    //    cout<<"Forming the filter bank failed "<<endl;
    //    return false;
    //}

    return true;
}

bool scanner::load_weight_to_filters()
{
    if(m_weight_vector.empty() || m_weight_vector.type()!=CV_32F)
	{
		cout<<"m_weight_vector is empty or type is not CV_32F"<<endl;
        return false;
	}

    // 4 texture, m_fhog_orientation insensitiev channel, 2*m_fhog_orientation sensitive channel
    int number_of_channels = m_fhog_orientation*3+4; 
    
    int filter_width  = m_padded_size.width/m_fhog_binsize;
    int filter_height = m_padded_size.height/m_fhog_binsize;

    /*  check another time */
    if(filter_width*filter_height*number_of_channels!= std::max(m_weight_vector.cols-1, m_weight_vector.rows-1))
    {
        cout<<"filter's size shoule equals feature's dim or length of m_weight_vector"<<endl;
        return false;
    }

    /* load to the filter matrix */
    int counter = 0;
    m_filters.reserve(number_of_channels);
    for(unsigned int i=0;i<number_of_channels;i++)
    {
        Mat filter = Mat::zeros(filter_height, filter_width, CV_32F);
        /* assign, row major*/
        for(unsigned int r=0;r<filter.rows;r++)
        {
            for(unsigned int c=0;c<filter.cols;c++)
            {
                filter.at<float>(r,c)=m_weight_vector.at<float>(counter++,0);
            }
        }
        m_filters.push_back( filter);
    }
    return true;
}

bool scanner::form_filter_bank(
								const double relative_ratio_to_max 	/*  in : if a filter's singular value less than relative_ratio_to_max*max_singular_value
																					 it will be discarded, DO NOT set this too large*/
								)
{
	if( m_filters.empty() || m_filters[0].type()!=CV_32F)
	{
		cout<<"Error, m_filters is empty() or wrong type, run load_weight_to_filters() function first..."<<endl;
		return false;
	}

    /* prepare for the row and col filters */
    m_row_filters.resize( m_filters.size() );
    m_col_filters.resize( m_filters.size() );

	/* decompose the m_filters, and save the row filter and col filter */
	for (unsigned int i=0 ;i<m_filters.size() ;i++ ) 
	{
		/*  save the SVD results, w is a column vector, stores the singular values*/
		Mat w,u,vt;
		SVD::compute( m_filters[i], w, u, vt);

		/*  find the max singular value */
		double min_w, max_w;
		cv::Point min_p,max_p;
		minMaxLoc( w, &min_w, &max_w, &min_p, &max_p );
		double threshold = std::max(1e-4, relative_ratio_to_max*max_w);

		for ( unsigned int j=0;j<w.rows;j++)
		{
			/*  adding those important filter */
			if( w.at<float>(j,0) > threshold)
			{
				Mat row_filer, col_filter;
				u.col(j).copyTo(col_filter);
				vt.row(j).copyTo(row_filer);
				m_row_filters[i].push_back( row_filer*std::sqrt( w.at<float>(j,0)));
				m_col_filters[i].push_back( col_filter*std::sqrt( w.at<float>(j,0)));
                m_number_of_seperable_filters++;
			}
		}
	}
    return true;
}
