#ifndef DETECT_CHECK
#define DETECT_CHECK
#include <iostream>
#include <vector>
#include <algorithm>
#include <sstream>
#include <ctime>
#include <omp.h>
#include <string>
#include <fstream>

#include "opencv2/contrib/contrib.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/contrib/contrib.hpp"

#include "boost/filesystem.hpp"
#include "boost/lambda/bind.hpp"

using namespace std;
using namespace cv;

namespace bf = boost::filesystem;
namespace bl = boost::lambda;

/* This class is used to check the performance of a detector which should implement
 * the detectMultiScale interface like this :
 */

//        bool detectMultiScale( const Mat &image,                    /* in : image */
//                               vector<Rect> &targets,               /* out: target positions*/
//                               vector<double> &confidence,          /* out: target confidence */
//                               const Size &minSize,                 /* in : min target Size */
//                               const Size &maxSize,                 /* in : max target Size */
//                               double scale_factor = 1.2,           /* in : scale factor */
//                               int stride = 4,                      /* in : detection stride */
//                               double threshold = 0) const;         /* in : detect threshold */
template< typename detectType>
class detect_check
{

    public:
    detect_check()
    {
        m_do_neg_test = true;
        m_minSize = Size(40,40);
        m_maxSize = Size(300,300);
        m_scale_factor = 1.2;
        m_stride = 1;
        m_threshold = 0;
        m_save_images = false;
    }
    /* 
     * ===  FUNCTION  ======================================================================
     *         Name:  set_path
     *  Description:  set the path to positive imgs, groundTruth files, pure neg imgs;
     * =====================================================================================
     */
    bool set_path( const string &positive_imgs,     /*  in : path of the positive images */
                   const string &gts,               /*  in : path of the groundtruth */
                   const string &negative_imgs,     /*  in : path of the pure negative image */
                   bool save_the_result)            /*  in : save the results image or not */
    {
        bf::path pos_img_path( positive_imgs);
        bf::path gt_path(gts);
        bf::path neg_img_path( negative_imgs );

        if( !bf::exists(pos_img_path) || !bf::exists(gt_path))
        {
            cout<<"Path of positive image or path does not exist "<<endl;
            cout<<"Check "<<positive_imgs<<" and "<<gts<<endl;
            return false;
        }
        if( !bf::exists(neg_img_path))
        {
            cout<<"Pure neg images folder does not exist, will skip the neg test "<<endl;
            m_do_neg_test = false;
        }

        m_pos_img_path = positive_imgs;
        m_gt_path = gts;
        m_neg_img_path = negative_imgs;
        if( save_the_result)
        {
            bf::path pos_fn("./pos_fn");
            bf::path pos_fp("./pos_fp");
            /* check if folder exist */ 
            if( !bf::exists(pos_fn))
                bf::create_directory(pos_fn);
            else
            {
                bf::remove_all(pos_fn);
                bf::create_directory(pos_fn);
            }
            
            if( !bf::exists(pos_fp))
                bf::create_directory(pos_fp);
            else
            {
                bf::remove_all(pos_fp);
                bf::create_directory(pos_fp);
            }
            m_save_images = save_the_result;
        }
        return true;
    }


    /* 
     * ===  FUNCTION  ======================================================================
     *         Name:  set_parameter
     *  Description:  the parameters for detector
     * =====================================================================================
     */
    bool set_parameter( const Size &minSize,
                        const Size &maxSize,
                        const double &scale_factor,
                        const int &stride,
                        const double &threshold)
    {
        if( minSize.width > maxSize.width || minSize.height > maxSize.height ||
            minSize.width <= 0 || minSize.height <= 0)
        {
            cout<<"Set the correct size "<<endl;
            return false;
        }
        m_minSize = minSize;
        m_maxSize = maxSize;
        
        if( scale_factor <=1)
        {
            cout<<"Set the right scale factor( > 1)"<<endl;
            return false;
        }
        m_scale_factor = scale_factor;

        if( stride < 0)
        {
            cout<<"Set the right stride "<<endl;
            return false;
        }
        m_stride = stride;
        m_threshold = threshold;

        return true;
    }


    bool generate_roc(  detectType &detector,           /*  in : detector */
                        vector<double> &FPPIs,          /*  out: fppis */
                        vector<double> &hits,            /*  out: hist  */
                        double start_threshold = 1,     /*  in : starter threshold */
                        double end_threshold = -1,       /*  in : end threshold */
                        int num_test_point = 10         /*  in : how many times we test */
                     )
    {
        /* clear the output */
        FPPIs.clear();
        hits.clear();
        vector<double> thres;

        double threshold_step = (end_threshold - start_threshold)/num_test_point;
        for( unsigned int c=0;c<num_test_point;c++)
        {
            double fppi = 0;
            double hit = 0;
            set_parameter(m_minSize, m_maxSize, m_scale_factor, m_stride, start_threshold+c*threshold_step);
            test_detector( detector, hit, fppi );
            FPPIs.push_back( fppi);
            hits.push_back( hit);
            thres.push_back(start_threshold+c*threshold_step);
            cout<<"round "<<c<<" out of "<<num_test_point<<" with hit "<<hit<<" and fppi "<<fppi<<endl;
        }

        /*  save it to file */
        ofstream output_f("./det_roc.txt", ios::trunc);
        if(!output_f.is_open())
        {
            cout<<"Can not open file ./det_roc.txt "<<endl;
            return false;
        }

        for( unsigned int c=0;c<FPPIs.size();c++)
            output_f<<FPPIs[c]<<" ";
        output_f<<endl;
        
        for( unsigned int c=0;c<hits.size();c++)
            output_f<<hits[c]<<" ";
        output_f<<endl;

        for( unsigned int c=0;c<thres.size();c++)
            output_f<<thres[c]<<" ";
        output_f<<endl;

        output_f.close();
        return true;
    }

    
    /* 
     * ===  FUNCTION  ======================================================================
     *         Name:  get_stat_on_missed
     *  Description:  mainly about the size of the missed face
     * =====================================================================================
     */
    bool get_stat_on_missed()
    {
        bf::path pos_fn("./pos_fn");
        if(!bf::exists(pos_fn))
        {
            cout<<"folder pos_fn does not exist "<<endl;
            return false;
        }
        long num_under_40 = 0;
        long num_40_80 = 0;
        long num_80_120 = 0;
        long num_above_120 = 0;

        bf::directory_iterator end_it;
        for( bf::directory_iterator file_iter( m_pos_img_path ); file_iter!=end_it; file_iter++)
	    {
            string pathname = file_iter->path().string();
            string basename = bf::basename( *file_iter);
	    	string extname  = bf::extension( *file_iter);
            /*  both the images and groundtruth file should exist */
	    	if( extname!=".jpg" && extname!=".bmp" && extname!=".png" && extname!=".JPG" && extname!=".BMP" && extname!=".PNG")
	    		continue;
            Mat img = imread( pathname );
            if(img.empty())
                continue;
            int face_size = img.rows;
            if( face_size <= 40)
                num_under_40++;
            else if( 40 < face_size && face_size <=80)
                num_40_80++;
            else if( 80<face_size && face_size<=120)
                num_80_120++;
            else
                num_above_120++;
        }
        ofstream stat_f("stat.txt", ios::trunc);
        stat_f<<"missed face size :"<<endl;
        stat_f<<"0 - 40 \t"<<num_under_40<<endl;
        stat_f<<"40- 80 \t"<<num_40_80<<endl;
        stat_f<<"80- 120 \t"<<num_80_120<<endl;
        stat_f<<"120 - inf \t"<<num_above_120<<endl;

        stat_f.close();
        return true;
    }

    /* 
     * ===  FUNCTION  ======================================================================
     *         Name:  test_detector
     *  Description:  check the performance of the detector
     * =====================================================================================
     */
    bool test_detector( detectType &detector,   /*  in : the detector */
                        double &hit,            /*  out: hit */
                        double &FPPI)           /*  out: false positive per image */
    {
        TickMeter tk;tk.start();
        int Nthreads = omp_get_max_threads();
        if(!check_path())
        {
            return false;
        }

        vector<string> pos_image_path_vector;
        vector<string> gt_path_vector;

        /*  first load the positive image and gts */
        bf::directory_iterator end_it;
        for( bf::directory_iterator file_iter( m_pos_img_path ); file_iter!=end_it; file_iter++)
	    {
            string pathname = file_iter->path().string();
            string basename = bf::basename( *file_iter);
	    	string extname  = bf::extension( *file_iter);
            /*  both the images and groundtruth file should exist */
	    	if( extname!=".jpg" && extname!=".bmp" && extname!=".png" && extname!=".JPG" && extname!=".BMP" && extname!=".PNG")
	    		continue;
            bf::path tmp_gt_path( m_gt_path + basename + ".xml");
            if(!bf::exists(tmp_gt_path))
                continue;
            pos_image_path_vector.push_back( pathname);
            gt_path_vector.push_back( m_gt_path + basename + ".xml");
	    }
        
        /*  Test on positive images */
        int number_of_target = 0;
        int number_of_fn = 0;
        int number_of_wrong = 0;
        
        #pragma omp parallel for num_threads(Nthreads) reduction( +: number_of_fn) reduction( +: number_of_wrong ) reduction(+:number_of_target)
        for( int i=0;i<pos_image_path_vector.size(); i++)
        { 
            // reading groundtruth...
            vector<Rect> target_rects;
            FileStorage fst( gt_path_vector[i], FileStorage::READ | FileStorage::FORMAT_XML);
            fst["boxes"]>>target_rects;
            fst.release();
            number_of_target += target_rects.size();

            // reading image
            Mat test_img = imread( pos_image_path_vector[i]);
            vector<Rect> det_rects;
            vector<double> det_confs;
            detector.detectMultiScale( test_img, det_rects, det_confs, m_minSize, m_maxSize, m_scale_factor, m_stride, m_threshold);

            /* debug show */
            //for ( int c=0;c<det_rects.size() ; c++) {
            //    rectangle( test_img, det_rects[c], Scalar(0,0,255), 3);
            //    cout<<"conf is "<<det_confs[c]<<endl;
            //}
            //cout<<endl;
            //imshow("test", test_img);
            //waitKey(0);

            int matched = 0;
            vector<bool> isMatched_r( target_rects.size(), false);
            vector<bool> isMatched_l( det_rects.size(), false);
            for( int c=0;c<det_rects.size();c++)
            {
                for( int k=0;k<target_rects.size();k++)	
                {
                    if( isSameTarget( det_rects[c], target_rects[k]) && !isMatched_r[k] && !isMatched_l[c])
                    {
                        matched++;
                        isMatched_r[k] = true;
                        isMatched_l[c] = true;
                        break;
                    }
                }
            }

            bf::path t_path( pos_image_path_vector[i]);
            string basename = bf::basename(t_path);
            /*  no need for the critical section when using the reduction */
            //#pragma omp critical
            {
                for(int c=0;c<isMatched_r.size();c++)
                {
                    if( !isMatched_r[c])
                    {
                        number_of_fn++;
                        if(m_save_images)
                        {
                            stringstream ss;ss<<c;string index_string;ss>>index_string;
                            string save_path = "./pos_fn/"+basename+"_"+index_string+".jpg";
                            Mat save_img = cropImage( test_img, target_rects[c] );
                            imwrite( save_path, save_img);
                        }
                    }
                }

                for(int c=0;c<isMatched_l.size();c++)
                {
                    if( !isMatched_l[c])
                    {
                        number_of_wrong++;
                        if(m_save_images)
                        {
                            stringstream ss;ss<<c;string index_string;ss>>index_string;
                            string save_path = "./pos_fp/"+basename+"_"+index_string+".jpg";
                            Mat save_img = cropImage( test_img, det_rects[c]);
                            imwrite( save_path, save_img);
                        }
                    }
                }
            }
        }

        hit  = 1.0*(number_of_target - number_of_fn)/number_of_target;
        FPPI = 1.0*(number_of_wrong)/pos_image_path_vector.size();
        tk.stop();
        cout<<"Done, total scan time "<<tk.getTimeSec()<<" seconds"<<endl;
        return true;
    }

    private:

    /* 
     * ===  FUNCTION  ======================================================================
     *         Name:  isSameTarget
     *  Description:  compute match score of two rects,  if inter(r1,r2)/union(r1,r2)> 0.5 --> matched!
     * =====================================================================================
     */
    bool isSameTarget( Rect r1, Rect r2)
    {
    	Rect intersect = r1 & r2;
    	if(intersect.width * intersect.height < 1)
    		return false;
    	
    	double union_area = r1.width*r1.height + r2.width*r2.height - intersect.width*intersect.height;
    
    	if( intersect.width*intersect.height/union_area < 0.5 )
    		return false;
    
    	return true;
    }


    /* 
     * ===  FUNCTION  ======================================================================
     *         Name:  check_path
     *  Description:  check if the path 
     * =====================================================================================
     */
    bool check_path()
    {
        bf::path pos_img_path( m_pos_img_path);
        bf::path gt_path(m_gt_path);

        if( !bf::exists(pos_img_path) || !bf::exists(gt_path))
        {
            cout<<"Path of positive image or path does not exist "<<endl;
            cout<<"Check "<<m_pos_img_path<<" and "<<m_gt_path<<endl;
            return false;
        }
        return true;
    }

    /*  data section */
    string m_pos_img_path;      /* path for positive images */
    string m_gt_path;           /* path for groundtruth files */
    string m_neg_img_path;      /* path for pure negative images */
    bool m_do_neg_test;         /*  do negative scan or not */

    /*  detector's paramters section */
    Size m_minSize;
    Size m_maxSize;
    double m_threshold;
    double m_scale_factor;
    int m_stride;

    /*  others */
    bool m_save_images;         /*  save the error images or not */
};
#endif
