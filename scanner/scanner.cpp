#include <iostream>
#include <cmath>
#include <sstream>
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

    for(unsigned int template_index=0;template_index<m_row_filters.size();template_index++)
    {
        vector<Mat> unsentivechannels;
        const float *weight_data = (const float*)(m_weight_vector[template_index].data);

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
        stringstream ss;ss<<template_index;string index_s;ss>>index_s;
        imshow("detector"+index_s, draw);
    }
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
    if( m_filters[0].size() != feature_chns.size())
    {
        cout<<"number of channels should equal the number of filters "<<endl;
        return false;
    }

    vector<Mat> saliency_map;
    saliency_map.resize( m_filters.size());
    for( unsigned int c=0;c<saliency_map.size();c++)
        saliency_map[c] = Mat::zeros( feature_chns[0].size(), CV_32F);
    
    //TickMeter tt;tt.start();
   Mat tmp_saliency;

   // /*  1 original convolution */
   // for(unsigned int template_index=0;template_index<m_row_filters.size();template_index++)
   // {
   //     for(unsigned int c=0;c<m_filters[template_index].size();c++)
   //     {
   //         cv::filter2D( feature_chns[c], tmp_saliency, CV_32F, m_filters[template_index][c] );
   //         saliency_map[template_index] = saliency_map[template_index] + tmp_saliency;
   //     }
   // }
   // tt.stop();cout<<"time 1 "<<tt.getTimeMilli()<<endl;tt.reset();


    /*  2 separable convolution , typically , when the number of seperable filter is less
     *  than 100, we should use separable convolution instead */
    //tt.start();
    for(unsigned int template_index=0;template_index<m_row_filters.size();template_index++)
    {
        get_saliency_map( feature_chns, template_index, saliency_map[template_index] );
    }
    //tt.stop();cout<<"time 2 "<<tt.getTimeMilli()<<endl;tt.reset();

    /*  add the bias term */
    for(unsigned int template_index=0;template_index<m_row_filters.size();template_index++)
        saliency_map[template_index] = saliency_map[template_index] + m_weight_vector[template_index].at<float>(m_feature_dim,0);

    /* adding detect results  */
    threshold_detection( saliency_map, results, confidence, threshold);

    return true;
}


void scanner::get_saliency_map( const vector<Mat> &feature_chns,         // in : input feature
                                const int template_index,               // in : index of the template
                               Mat &saliency_map)                       // out: output saliency map( detect confidence)
{
    Mat tmp_saliency;
    for(unsigned int i=0;i<m_row_filters[template_index].size();i++)
    {
        for(unsigned int j=0;j<m_row_filters[template_index][i].size();j++)
        {
            cv::filter2D( feature_chns[i], tmp_saliency, CV_32F, m_col_filters[template_index][i][j]);
            cv::filter2D( tmp_saliency, tmp_saliency, CV_32F, m_row_filters[template_index][i][j]);
            saliency_map = saliency_map + tmp_saliency;
        }
    }
}


void scanner::threshold_detection( const vector<Mat> &saliency_map,      // in : saliency_maps
                                   vector<Rect> &det_results,            // out: detect results            
                                   vector<double> &det_confs,            // out : detect confidence
                                   const double threshold)               // in : threshold         
{    
    const int slide_width  = m_padded_size.width/m_fhog_binsize;
    const int slide_height = m_padded_size.height/m_fhog_binsize;

    /* adding detect results  */
    for(unsigned int template_index=0;template_index<saliency_map.size();template_index++)
    {
        for(unsigned int r=0;r<saliency_map[template_index].rows;r++)
        {
            for(unsigned int c=0;c<saliency_map[template_index].cols;c++)
            {
                if(saliency_map[template_index].at<float>(r,c) > threshold)
                {
                    det_results.push_back(cv::Rect( (c-slide_width/2)*m_fhog_binsize, 
                                                (r-slide_height/2)*m_fhog_binsize, 
                                                m_padded_size.width, 
                                                m_padded_size.height));
                    det_confs.push_back(saliency_map[template_index].at<float>(r,c));
                }
            }
        }
    }
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
            adjust_detection( tmp_target, scale_vec[scale_index], input_image);

            results.push_back( tmp_target);
            confidence.push_back( det_confs[c] );
        }
    }
    
    /*  group the detected results */
    NonMaxSupress( results, confidence );
    return true;
}

void scanner::adjust_detection( Rect &det_result,           // in&out 
                                double scale_factor,        // in 
                                const Mat &input_image)     // in 
{ 
    det_result.x /= scale_factor;
    det_result.y /= scale_factor;
    det_result.width  /= scale_factor;
    det_result.height /= scale_factor;
    
    /*  convert from padded_size to target_size */
    det_result.x += det_result.width*1.0*(m_padded_size.width-m_target_size.width)/(2*m_padded_size.width);
    det_result.y += det_result.height*1.0*(m_padded_size.height-m_target_size.height)/(2*m_padded_size.height);
    det_result.width *= 1.0*m_target_size.width/m_padded_size.width;
    det_result.height *= 1.0*m_target_size.height/m_padded_size.height;

    /*  sometimes the detected result will be slightly out of the image , crop it */
    if( det_result.x < 0)
        det_result.x = 0;
    if( det_result.y < 0)
        det_result.y = 0;
    if( det_result.x + det_result.width > input_image.cols)
        det_result.width = input_image.cols - det_result.x -1;
    if( det_result.y + det_result.height > input_image.rows)
        det_result.height = input_image.rows - det_result.y - 1;

    
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

bool scanner::setParameters( const int &fhog_binsize,               //in : fhog's binsize       
                             const int &fhog_orientation,           //in : fhog's number of orientation
                             const Size &target_size,               //in : target's size
                             const Size &padded_size,               //in : target's padded size( used to scan the image)
                             const vector<Mat> &weight_vectors)      //in : svm's weight vector
{
    m_weight_vector.clear();

    m_fhog_binsize = fhog_binsize;
    m_fhog_orientation = fhog_orientation;

    m_target_size = target_size;
    m_padded_size = padded_size;

    for(unsigned int c=0;c<weight_vectors.size();c++)
    {
        if( weight_vectors[c].type() != CV_32F)
        {
            cout<<"weight_vector should be type CV_32F"<<endl;
            return false;
        }
        m_weight_vector.push_back(weight_vectors[c]);
    }

    /*  dimension of m_weight_vector is feature_dim + 1 ( bias term is not counted as feature_dim ) */
    m_feature_dim = (m_weight_vector[0].rows > m_weight_vector[0].cols? m_weight_vector[0].rows: m_weight_vector[0].cols) - 1;

    /*  check the parameters */
    if(!checkParameter())
    {
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

    for(unsigned int c=0;c<m_weight_vector.size();c++)
    {
        if( m_weight_vector[c].empty() || !m_weight_vector[c].isContinuous() || m_weight_vector[c].type()!=CV_32F)
        {
            cout<<"Weight vector should be continuous and format shoule be CV_32F"<<endl;
            return false;
        }

        if( m_weight_vector[c].cols!=1 && m_weight_vector[c].rows!=1 )
        {
            cout<<"Weight vector should be one column or one row "<<endl;
            return false;
        }
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
    fs<<"weight_vectors"<<m_weight_vector;
    
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
    
    vector<Mat> weight_vectors;
    Size padded_size;
    Size target_size;
    int feature_dim;
    int fhog_binsize;
    int fhog_orientation;

    fs["weight_vectors"]>>weight_vectors;
    fs["padded_size"]>>padded_size;
    fs["fhog_binsize"]>>fhog_binsize;
    fs["fhog_orientation"]>>fhog_orientation;
    fs["target_size"]>>target_size;
    fs["infos"]>>m_info;

    setParameters( fhog_binsize, fhog_orientation, target_size, padded_size, weight_vectors);
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
    if(!form_filter_bank())
    {
        cout<<"Forming the filter bank failed "<<endl;
        return false;
    }
    return true;
}


bool scanner::loadModels( const vector<string> &model_files)
{
    /* load first model */
    if( !loadModel( model_files[0]) )
    {
        cout<<"Can not load the first model ..."<<endl;
        return false;
    }
    cout<<"Load template 0 done "<<endl;

    if(model_files.size() == 1)
        return true;

    /*  load the result */
    for( unsigned int c=1;c<model_files.size(); c++)
    {
       FileStorage fs( model_files[c], FileStorage::READ);
       if( !fs.isOpened())
       {
           cout<<"Can not open the file "<<model_files[c]<<endl;
           return false;
       }
       /* check if they share the same configuration with first one */
       vector<Mat> weight_vectors;
       Size padded_size;
       Size target_size;
       int feature_dim;
       int fhog_binsize;
       int fhog_orientation;

       fs["weight_vectors"]>>weight_vectors;
       fs["padded_size"]>>padded_size;
       fs["fhog_binsize"]>>fhog_binsize;
       fs["fhog_orientation"]>>fhog_orientation;
       fs["target_size"]>>target_size;
       fs["infos"]>>m_info;

       /*  check the models' configuration */
       if( padded_size.width != m_padded_size.width ||
           padded_size.height != m_padded_size.height ||
           fhog_binsize != m_fhog_binsize ||
           fhog_orientation != m_fhog_orientation ||
           target_size.width != m_target_size.width ||
           target_size.height != m_target_size.height )
       {
           cout<<"Models should share the same configuration "<<endl;
           return false;
       }
        
       /* adding the template */
       for( unsigned int i=0;i<weight_vectors.size();i++ )
       {
            m_weight_vector.push_back( weight_vectors[i]);
       }
       fs.release();
       cout<<"Load template "<<c<<" done "<<endl;
    }
    
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
    if(!form_filter_bank())
    {
        cout<<"Forming the filter bank failed "<<endl;
        return false;
    }

    cout<<"Models loading done"<<endl;
    return true;
}



bool scanner::load_weight_to_filters()
{
    m_filters.clear();
    m_row_filters.clear();
    m_col_filters.clear();

    if(m_weight_vector.empty())
	{
		cout<<"m_weight_vector is empty"<<endl;
        return false;
	}

    /*  all the filter in the vector should have the same width and height */
    const int temp_same_width = m_weight_vector[0].cols;
    const int temp_samp_height = m_weight_vector[0].rows;

    for( unsigned int c=1; c<m_weight_vector.size(); c++)
    {
        if( temp_same_width != m_weight_vector[c].cols ||
            temp_samp_height != m_weight_vector[c].rows)
        {
            cout<<"weight vectors should have the same width and height "<<endl;
            return false;
        }
    }

    // 4 texture, m_fhog_orientation insensitiev channel, 2*m_fhog_orientation sensitive channel
    int number_of_channels = m_fhog_orientation*3+4; 
    
    int filter_width  = m_padded_size.width/m_fhog_binsize;
    int filter_height = m_padded_size.height/m_fhog_binsize;

    /*  check another time */
    if(filter_width*filter_height*number_of_channels!= std::max(temp_same_width-1, temp_samp_height-1))
    {
        cout<<"filter's size shoule equals feature's dim or length of m_weight_vector"<<endl;
        return false;
    }

    /* load to the filter matrix */
    m_filters.resize( m_weight_vector.size());
    for( unsigned int template_index=0; template_index<m_weight_vector.size(); template_index++) 
    {
        int counter = 0;
        m_filters[template_index].reserve(number_of_channels);
        for(unsigned int channel_index=0;channel_index<number_of_channels;channel_index++)
        {
            Mat filter = Mat::zeros(filter_height, filter_width, CV_32F);
            /* assign, row major*/
            for(unsigned int r=0;r<filter.rows;r++)
            {
                for(unsigned int c=0;c<filter.cols;c++)
                {
                    filter.at<float>(r,c)=m_weight_vector[template_index].at<float>(counter++,0);
                }
            }
            m_filters[template_index].push_back(filter);
        }

    }

    return true;
}

bool scanner::form_filter_bank(
								const double relative_ratio_to_max 	/*  in : if a filter's singular value less than relative_ratio_to_max*max_singular_value
																					 it will be discarded, DO NOT set this too large*/
								)
{
    m_row_filters.clear();
    m_col_filters.clear();

	if( m_filters.empty() || m_filters[0][0].type()!=CV_32F)
	{
		cout<<"Error, m_filters is empty() or wrong type, run load_weight_to_filters() function first..."<<endl;
		return false;
	}


    /* prepare for the row and col filters */
    m_row_filters.resize( m_filters.size() );
    m_col_filters.resize( m_filters.size() );

    for(unsigned int template_index=0;template_index<m_filters.size();template_index++)
    {
        int n_count = 0;    /* separable filter's count*/
        m_row_filters[template_index].resize(m_filters[template_index].size());
        m_col_filters[template_index].resize(m_filters[template_index].size());

        /* decompose the m_filters, and save the row filter and col filter */
        for(unsigned int channel_index=0;channel_index<m_filters[template_index].size(); channel_index++)
        {
            /*  save the SVD results, w is a column vector, stores the singular values*/
            Mat w,u,vt;
            SVD::compute( m_filters[template_index][channel_index], w, u, vt);
            
            /*  find the max singular value, set threshold value to "crop" */
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
		    		m_row_filters[template_index][channel_index].push_back( row_filer*std::sqrt( w.at<float>(j,0)));
		    		m_col_filters[template_index][channel_index].push_back( col_filter*std::sqrt( w.at<float>(j,0)));
                    n_count++;
		    	}
		    }
        }
        cout<<"adding "<<n_count<<" separable filters for template "<<template_index<<endl;
    }
    return true;
}
