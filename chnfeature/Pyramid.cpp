#include <iostream>
#include <fstream> 
#include <cmath>
#include <vector>
#include <typeinfo>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/contrib/contrib.hpp"
#include "../misc/misc.hpp"
#include "sseFun.h"
#include "Pyramid.h"

using namespace std;
using namespace cv;

bool feature_Pyramids::chnsPyramid_sse( const Mat &img,                                    //in:  image
										vector<vector<Mat> > &approxPyramid,			   //out: feature channels pyramid
										vector<double> &scales,							   //out: all scales
										vector<double> &scalesh,						   //out: the height scales
										vector<double> &scalesw) const  				   //out: the width scales
{
    if( img.empty() || !img.isContinuous() || img.cols < 8 || img.rows <8)
    {
        cout<<"image is too samll or not Continuous"<<endl;
        return false;
    }

	int shrink =m_opt.shrink;     //down_samples
	int smooth =m_opt.smooth;
	int nApprox=m_opt.nApprox;
	Size pad =m_opt.pad;
	
	/*  check the parameters */
	CV_Assert(nApprox>=0);
	CV_Assert(shrink>=1); 
	CV_Assert(pad.width>=0&&pad.height>=0);

	/*clear the input*/
	scales.clear();
	approxPyramid.clear();
	scalesh.clear();
	scalesw.clear();

	/*get scales*/
	vector<Size> ap_size;
	vector<int> real_scal;
	getscales(img,ap_size,real_scal,scales,scalesh,scalesw);
	Mat img_tmp;	
	
	vector<vector<Mat> > chns_Pyramid;
	int chns_num;
	for (int s_r=0;s_r<(int)real_scal.size();s_r++)
	{
		vector<Mat> chns;
		resize(img,img_tmp,ap_size[real_scal[s_r]]*shrink,0.0,0.0,INTER_AREA);
		computeChannels_sse(img_tmp,chns);
		chns_num=chns.size();
		chns_Pyramid.push_back(chns);
	}

	//compute lambdas
	vector<double> lambdas;
	if (nApprox!=0)
	{
		get_lambdas(chns_Pyramid,lambdas,real_scal,scales);

	}
	//compute based-scales
	vector<int> approx_scal;
	for (unsigned int s_r=0;s_r<scales.size();s_r++)
	{
		int tmp=s_r/(nApprox+1);
		if (s_r-real_scal[tmp]>((nApprox+1)/2))
		{
			approx_scal.push_back(real_scal[tmp+1]);
		}else{
			approx_scal.push_back(real_scal[tmp]);
		}	
	}
	//compute approxPyramid
	double ratio;
	for (int ap_id=0;ap_id<(int)approx_scal.size();ap_id++)
	{
		vector<Mat> approx_chns;
		approx_chns.clear();
		/*memory is consistent*/
		int approx_rows=ap_size[ap_id].height;
		int approx_cols=ap_size[ap_id].width;
		/*the size of pad*/
		int pad_T=pad.height/shrink;
		int pad_R=pad.width/shrink;
		Mat approx=Mat::zeros(10*(approx_rows+2*pad_T),approx_cols+2*pad_R,CV_32FC1);//因为chns_Pyramind是32F
		for(int n_chans=0;n_chans<chns_num;n_chans++)
		{
			Mat py_tmp=Mat::zeros(approx_rows,approx_cols,CV_32FC1);
			Mat py=approx.rowRange(n_chans*(approx_rows+2*pad_T),(n_chans+1)*(approx_rows+2*pad_T));//pad 以后的图像
			int ma=approx_scal[ap_id]/(nApprox+1);
			resize(chns_Pyramid[ma][n_chans],py_tmp,py_tmp.size(),0.0,0.0,INTER_LINEAR);
			if (nApprox!=0)
			{
				ratio=(double)pow(scales[ap_id]/scales[approx_scal[ap_id]],-lambdas[n_chans]);
				py_tmp=py_tmp*ratio;
			}
			//smooth channels, optionally pad and concatenate channels
			convTri(py_tmp,py_tmp,1,1);
			copyMakeBorder(py_tmp,py,pad_T,pad_T,pad_R,pad_R,IPL_BORDER_CONSTANT);
			approx_chns.push_back(py);
		}		
		approxPyramid.push_back(approx_chns);
	}	
	vector<int>().swap(approx_scal);
	vector<double>().swap(lambdas);
	vector<vector<Mat> >().swap(chns_Pyramid);
	vector<int>().swap(real_scal);
	vector<Size>().swap(ap_size) ;
	
	return true;
}

bool feature_Pyramids::fhog( const Mat &input_image,//in : input image ( w x h )
                             Mat &fhog_feature,     //out: output feature ( w/binSize*(3*oritent+5) x h/binSize for fhog, w/binSize*(4*oritent) x h/binSize for hog)
                             vector<Mat> &fea_chns, //out: share the same memory with feature, just a wrapper for operation, each channels -> one orientation
                             int type,              //in :  0 -> fhog, otherwise -> hog
                             int binSize,           //in : binSize ,better be 8
                             int oritent,           //in : oritent ,better be 9 
                             float clip             //in : clip value, better be 0.2
                             ) const
{
    int dimension = 1;          //convert input_image to gray image if needed
    if(input_image.empty())
    {
        cout<<"input_image is empty in function fhog"<<endl;
        return false;
    }
    Mat gray_input;

    if( input_image.channels() == 3)
    {
        cvtColor( input_image, gray_input, CV_BGR2GRAY);
    }
    else if( input_image.channels() == 1)
        gray_input = input_image;
    
    Mat f_input_image;      /*  convert to float image if needed */
    const float *f_input_data;
    if( gray_input.depth() == CV_8U  )
    {
        gray_input.convertTo( f_input_image, CV_32F, 1.0f/255);
        f_input_data = (float*)f_input_image.data;
    }
    else if( gray_input.depth() == CV_64F)
    {
        gray_input.convertTo( f_input_image, CV_32F);
        f_input_data = (float*)f_input_image.data;
    }
    else if( gray_input.depth() == CV_32F)
        f_input_data = (float*)gray_input.data;
    else
    {
        cout<<"unsupported data type in Function fhog"<<endl;
        return false;
    }

    /*  compute the raw magnitue and orientation */
    Mat mag = Mat::zeros( input_image.size(), CV_32F);
    Mat ori = Mat::zeros( input_image.size(), CV_32F);
    gradMag( f_input_data, (float *)(mag.data), (float *)(ori.data), input_image.rows, input_image.cols, dimension, true );
    /* father code the mag and ori due to different feature type */
    if(type == 0)       // fhog
    {
		if( fhog_feature.empty() || 
		fhog_feature.rows != input_image.rows/binSize*( oritent*3+5) ||
		fhog_feature.cols != input_image.cols/binSize)
		{
			fhog_feature = Mat::zeros( input_image.rows/binSize*( oritent*3+5), input_image.cols/binSize, CV_32F );
		}
        ssefhog( (const float*)(mag.data), (const float *)(ori.data), (float *)(fhog_feature.data),input_image.rows,input_image.cols, binSize, oritent ,clip);

        /*  "store" it in fea_chns */
        for(int c=0;c<oritent*3+5;c++)
        {
            Mat chns_temp = fhog_feature.rowRange( input_image.rows/binSize*c, input_image.rows/binSize*(c+1) );
            fea_chns.push_back( chns_temp);
        }
    }
    else            //hog
    {
		if( fhog_feature.empty() || 
		fhog_feature.rows != input_image.rows*oritent*4/binSize ||
		fhog_feature.cols != input_image.cols/binSize)
		{
			fhog_feature = Mat::zeros( input_image.rows*oritent*4/binSize, input_image.cols/binSize, CV_32F );
		}
        ssehog( (const float*)(mag.data), 
                (const float *)(ori.data), 
                (float *)(fhog_feature.data),input_image.rows,input_image.cols, binSize, oritent ,false, clip);

        /*  "store" it in fea_chns */
        for(int c=0;c<oritent*4;c++)
        {
            Mat chns_temp = fhog_feature.rowRange( input_image.rows/binSize*c, input_image.rows/binSize*(c+1) );
            fea_chns.push_back( chns_temp);
        }
    }

    return true;
}


void feature_Pyramids::visualizeHog(const vector<Mat> &chns,         // in : each chns corresponding to one orientation
                                    Mat &glyphImg, 
                                    int glyphSize, 
                                    double range)
{
    int numOrient = chns.size();
    int hogW = chns[0].cols, hogH = chns[0].rows;
    double radius = glyphSize*0.48;
    int thickness = 1+glyphSize*0.02;
    double scale = 255.0/range;

    glyphImg = cv::Mat::zeros(hogH*glyphSize, hogW*glyphSize, CV_8U);

    for (int i=0; i<numOrient; i++)
    {
        cv::Point pt1, pt2, offset;
        double angle = i*CV_PI/numOrient+CV_PI/2;
        const cv::Mat &hog = chns[i];
        pt1.x = glyphSize/2 + std::cos(angle)*radius;
        pt1.y = glyphSize/2 + std::sin(angle)*radius;
        pt2.x = glyphSize/2 - std::cos(angle)*radius;
        pt2.y = glyphSize/2 - std::sin(angle)*radius;

        for (int j=0; j<hogW; j++)
        {
            for (int k=0; k<hogH; k++)
            {
                offset.x = j*glyphSize;
                offset.y = k*glyphSize;
                cv::line(glyphImg, pt1+offset, pt2+offset,
                         scale*hog.at<float>(k, j), thickness);
            }
        }

    }
}





bool feature_Pyramids::computeGradHist(  const Mat &mag,       //in : mag
                                          const Mat &ori,       //in : ori
                                          Mat &Ghist,           //out: g hist
                                          int binSize,          //in : number of bin
                                          int oritent,          //in : number of ori
                                          bool full             //in : ture->0-2pi, false->0-pi
                                          ) const
{
    if( mag.depth() !=CV_32F || ori.depth()!=CV_32F || mag.empty() || ori.empty() )
    {
        cout<<"mag, ori format wrong"<<endl;
        return false;
    }
    if( mag.cols!=ori.cols || mag.rows!=ori.rows)
    {
        cout<<"mag , ori size do not match "<<endl;
        return false;
    }
    if( Ghist.empty())
        Ghist = Mat::zeros( mag.rows/binSize*oritent, mag.cols/binSize, CV_32F );
    else
    {
        if( Ghist.rows != mag.rows/binSize*oritent || Ghist.cols != mag.cols/binSize)
        {
            cout<<"error - > Ghist is pre allocated, but the size doesn't match "<<endl;
        }
    }
    gradHist( (float*)mag.data, (float*)ori.data, (float*)Ghist.data, mag.rows, mag.cols, binSize, oritent, 0, full);

    return true;
}

bool feature_Pyramids::convt_2_luv( const Mat input_image, 
					                 Mat &L_channel,
					                 Mat &U_channel,
					                 Mat &V_channel) const
{
	if( input_image.channels() != 3 || input_image.empty())
		return false;
	/* L U V channel is continunous in memory ~ */
	Mat luv_big = Mat::zeros( input_image.rows*3, input_image.cols, CV_32F);
	L_channel = luv_big.rowRange( 0, input_image.rows);
	U_channel = luv_big.rowRange( input_image.rows, input_image.rows*2);
	V_channel = luv_big.rowRange( input_image.rows*2, input_image.rows*3);
	int number_of_element = input_image.cols * input_image.rows;
	if( input_image.depth() == CV_8U)
		rgb2luv_sse( (const uchar*)(input_image.data), (float*)(luv_big.data), number_of_element, 1.0f/255);
	else if( input_image.depth() == CV_32F)
		rgb2luv_sse( (const float*)(input_image.data), (float*)(luv_big.data), number_of_element, 1.0f);
	else if( input_image.depth() == CV_64F)
		rgb2luv_sse( (const double*)(input_image.data), (float*)(luv_big.data), number_of_element, 1.0);
	else
		return false;
	return true;
}

bool feature_Pyramids::computeGradMag(  const Mat &input_image,   
                                        const Mat &input_image2,
                                        const Mat &input_image3,
                                        Mat &mag,
                                        Mat &ori,
                                        bool full,
                                        int channel) const
{
    int dim = 1;
    if( input_image.depth() != CV_32F || input_image.empty() || channel < 0 || channel>2)
    {
        cout<<"image empty or depth!= CV_32F or channel < 0 || channel > 2"<<endl;
        return false;
    }
    if( !input_image2.empty() || !input_image3.empty() )
    {
        dim = 3;
        /* check the input data */
        float* data_start = (float*)input_image.data;
        float* data2_start = (float*)input_image2.data;
        float* data3_start = (float*)input_image3.data;
        if( data_start+input_image.cols*input_image.rows != data2_start || 
                data_start+2*input_image.cols*input_image.rows !=data3_start)
        {
            cout<<"for color image, the channels should be continuous in memory "<<endl;
        }
    }

    mag = Mat::zeros( input_image.size(), CV_32F);
    ori = Mat::zeros( input_image.size(), CV_32F);
    
    float* in_data = (float*)(input_image.data);
    if( channel!=0)
    {
        /* use specific channel */
        in_data += channel*input_image.cols*input_image.rows;
        dim = 1;
    }

    gradMag( (const float*)(input_image.data), (float *)(mag.data), (float *)(ori.data), input_image.rows, 
                input_image.cols, dim, full );

    Mat smooth_mag;
    int norm_pad = 5;
    convTri( mag, smooth_mag, norm_pad, 1);
    float norm_const = 0.005;

    gradMagNorm( (float*)mag.data, (float*)smooth_mag.data, mag.rows, mag.cols, norm_const);

    return true;
}

 bool feature_Pyramids::computeChannels_sse( const Mat &image,                  // in : input image, BGR 
                                                vector<Mat>& channels) const    //out : 10 channle features, continuous in memory
{
    const int limited_size = 8;
    if( image.channels() != 3 || image.empty() || image.cols <limited_size || image.rows < limited_size)
    {
        cout<<" only works with color imagem which size is larger than 8"<<endl;
        return false;
    }

    /*set para*/
	int nbins=m_opt.nbins;
	int binsize=m_opt.binsize;
	int shrink =m_opt.shrink;
	int smoothSize=m_opt.smooth;

	int channels_addr_rows=(image.rows)/shrink;
	int channels_addr_cols=(image.cols)/shrink;
	Mat channels_addr=Mat::zeros((nbins+4)*channels_addr_rows,channels_addr_cols,CV_32FC1);

    /*  convert to LUV with normalization to [0,1] */
    /*  first crop it to the proper size, use crop instead of resize to keep the nature radio of image */
    int w_cropped_size = image.cols - image.cols%shrink;
    int h_cropped_size = image.rows - image.rows%shrink;
    if( w_cropped_size < limited_size|| h_cropped_size < limited_size)
    {
        cout<<"image is too small "<<endl;
        return false;
    }
    Mat crop_input = image.rowRange( 0, h_cropped_size).colRange(0, w_cropped_size );
    
    /* 1--> convert it to LUV with normalization */
    Mat L,U,V;
    if(!convt_2_luv( crop_input, L, U, V))
    {
        cout<<"error in convt_2_luv function "<<endl;
        return false;
    }

    /*  smooth all the three channel */
    Mat smooth_LUV;                      //LUV togather, 3 times bigger than L
    convTri( L,smooth_LUV, smoothSize, 3);  //L U V is all been smoothed, giving just L ( since they have the same size and continuous in memory)
    
    /*  resize and add to channels */
    for ( int c=0; c<3;c++) 
    {
        Mat c_resized = channels_addr.rowRange( c*channels_addr_rows, (c+1)*channels_addr_rows);
        cv::resize( smooth_LUV.rowRange( c*crop_input.rows, (c+1)*crop_input.rows), c_resized, c_resized.size(), 0.0, 0.0, 1);
        channels.push_back(c_resized);
    }
    
    /* 2--> compute the magnitude and oritentation */
    Mat Mag,Ori;
    computeGradMag(L, U, V, Mag, Ori, false);
    Mat mag_resized = channels_addr.rowRange( 3*channels_addr_rows, 4*channels_addr_rows);
    cv::resize( Mag, mag_resized, mag_resized.size(), 0.0, 0.0, INTER_AREA);
    channels.push_back( mag_resized );

    /*  3--> compute Gradient hist */ 
    /* size will be nbins*channels_addr_rows x channels_addr_cols */
    Mat gghist = channels_addr.rowRange( 4*channels_addr_rows, (4+nbins)*channels_addr_rows);   
    if(!computeGradHist( Mag, Ori, gghist, binsize, nbins, false))
    {
        cout<<"computeGradHist wrong "<<endl;
        return false;
    }
    
    if(gghist.rows/nbins != channels_addr_rows || gghist.cols!= channels_addr_cols)
    {
        cout<<"fatal error, some thing wrong in computeGradHist, size does not match ~"<<endl;
        return false;
    }
    
    /* push it into channels */
    for ( int c=0;c<nbins ; c++)
    {
        Mat ghist_temp = gghist.rowRange( c*channels_addr_rows, (c+1)*channels_addr_rows);
        channels.push_back( ghist_temp );
    }

    /* check the pointer */
    float * add1 = (float*)channels[0].data;
    for( int c=1;c<channels.size();c++)
	{
	    if(!(static_cast<void*>(add1+c*channels[0].cols*channels[0].rows) == static_cast<void*>(channels[c].data)))
            return false;
	}
    return true;
}

bool feature_Pyramids::chnsPyramid_sse(const Mat &img,                      //in : input image
                                     vector<vector<Mat> > &chns_Pyramid,    //out: output features
                                     vector<double> &scales) const          //out: scale of each pyramid
{
    if( img.empty() || !img.isContinuous() || img.cols < 8 || img.rows <8)
    {
        cout<<"image is too samll or not Continuous"<<endl;
        return false;
    }
    
	int shrink =m_opt.shrink;
	int smooth =m_opt.smooth;

	/*get scales*/
	vector<Size> ap_size;
	vector<int> real_scal;
	vector<double> scalesh;
	vector<double> scalesw;
	//clear
	scales.clear();
	chns_Pyramid.clear();

	getscales(img,ap_size,real_scal,scales,scalesh,scalesw);

	Mat img_tmp;
	Mat img_half;
	//compute real 
	for (int s_r=0;s_r<(int)scales.size();s_r++)
	{
		vector<Mat> chns;
		cv::resize(img,img_tmp,ap_size[s_r]*shrink,0.0,0.0,1);
		if(!computeChannels_sse(img_tmp,chns))
        {
            cout<<"error computing computeChannels_sse "<<endl;
            return false;
        }
		for (int c = 0; c < chns.size(); c++)
		{
			convTri(chns[c],chns[c],1,1);
		}
		chns_Pyramid.push_back(chns);
	}
    return true;
}




Mat get_Km(int smooth )
{ 
    Mat dst(1, 2*smooth+1, CV_32FC1);
	for (int c=0;c<=smooth;c++)
	{
		dst.at<float>(0,c)=(float)((c+1)/((smooth+1.0)*(smooth+1.0)));
		dst.at<float>(0,2*smooth-c)=dst.at<float>(0,c);
	}
    return dst;
}


void feature_Pyramids::get_lambdas( vector<vector<Mat> > &chns_Pyramid,  //in:  image feature pyramid
								    vector<double> &lambdas,             //out: lambdas
									vector<int> &real_scal,              //in:  the layer of image pyramid
									vector<double> &scales               //in:  all scales 
									)const         
{
	
	if (lam.empty()) 
	{
		Scalar lam_s;
		Scalar lam_ss;
		CV_Assert(chns_Pyramid.size()>=2);
		if (chns_Pyramid.size()>2)
		{
			//compute lambdas
			double size1,size2,lam_tmp;
			size1 =(double)chns_Pyramid[1][0].rows*chns_Pyramid[1][0].cols;
			size2 =(double)chns_Pyramid[2][0].rows*chns_Pyramid[2][0].cols;
			//compute luv	 
			for (int c=0;c<3;c++)
			{
				lam_s+=sum(chns_Pyramid[1][c]);		
				lam_ss+=sum(chns_Pyramid[2][c]);
			}
			lam_s=lam_s/(size1*3.0);
			lam_ss=lam_ss/(size2*3.0);
			lam_tmp=-cv::log(lam_ss.val[0]/lam_s.val[0])/cv::log(scales[real_scal[2]]/scales[real_scal[1]]);
			for (int c=0;c<3;c++)
			{
				lambdas.push_back(lam_tmp);
			}
			//compute  mag
			lam_s=sum(chns_Pyramid[1][4])/(size1*1.0);
			lam_ss=sum(chns_Pyramid[2][4])/(size2*1.0);
			lambdas.push_back(-cv::log(lam_ss.val[0]/lam_s.val[0])/cv::log(scales[real_scal[2]]/scales[real_scal[1]]));
			//compute grad_hist
			for (int c=4;c<10;c++)
			{
				lam_s+=sum(chns_Pyramid[1][c]);		
				lam_ss+=sum(chns_Pyramid[2][c]);
			}
			lam_s=lam_s/(size1*6.0);
			lam_ss=lam_ss/(size2*6.0);
			lam_tmp=-cv::log(lam_ss.val[0]/lam_s.val[0])/cv::log(scales[real_scal[2]]/scales[real_scal[1]]);
			for (int c=4;c<10;c++)
			{
				lambdas.push_back(lam_tmp);
			}
		}else{
			//compute lambdas
			double size0,size1,lam_tmp;
			size0 =(double)chns_Pyramid[0][0].rows*chns_Pyramid[0][0].cols;
			size1 =(double)chns_Pyramid[1][0].rows*chns_Pyramid[1][0].cols;
			//compute luv	 
			for (int c=0;c<3;c++)
			{
				lam_s+=sum(chns_Pyramid[0][c]);		
				lam_ss+=sum(chns_Pyramid[1][c]);
			}
			lam_s=lam_s/(size0*3.0);
			lam_ss=lam_ss/(size1*3.0);
			lam_tmp=-cv::log(lam_ss.val[0]/lam_s.val[0])/cv::log(scales[real_scal[1]]/scales[real_scal[0]]);
			for (int c=0;c<3;c++)
			{
				lambdas.push_back(lam_tmp);
			}
			//compute  mag
			lam_s=sum(chns_Pyramid[0][4])/(size0*1.0);
			lam_ss=sum(chns_Pyramid[1][4])/(size1*1.0);
			lambdas.push_back(-cv::log(lam_ss.val[0]/lam_s.val[0])/cv::log(scales[real_scal[1]]/scales[real_scal[0]]));
			//compute grad_hist
			for (int c=4;c<10;c++)
			{
				lam_s+=sum(chns_Pyramid[0][c]);		
				lam_ss+=sum(chns_Pyramid[1][c]);
			}
			lam_s=lam_s/(size0*6.0);
			lam_ss=lam_ss/(size1*6.0);
			lam_tmp=-cv::log(lam_ss.val[0]/lam_s.val[0])/cv::log(scales[real_scal[1]]/scales[real_scal[0]]);
			for (int c=4;c<10;c++)
			{
				lambdas.push_back(lam_tmp);
			}
		}
	}else{
		lambdas.resize(10);
		for(int n=0;n<3;n++)
		{
			lambdas[n]=lam[0];
		}
		lambdas[3]=lam[1];
		for(int n=4;n<10;n++)
		{
			lambdas[n]=lam[2];
		}	
	}
}

void feature_Pyramids::convTri( const Mat &src,                          //in:  inputArray
							    Mat &dst,                                //out: outpuTarray
							    const Mat &Km							 //in:  the kernel of Convolution
								) const   //1 opencv version
{
	CV_Assert((!src.empty()) && (!Km.empty()));
	filter2D(src,dst,src.depth(),Km,Point(-1,-1),0,IPL_BORDER_REFLECT);
	filter2D(dst,dst,src.depth(),Km.t(),Point(-1,-1),0,IPL_BORDER_REFLECT); 
	
}	

void feature_Pyramids::convTri( const Mat &src, Mat &dst, int conv_size, int dim) const      //2 sse version, faster
{
	CV_Assert(src.channels()==1 && conv_size > 0);//不支持多通道
    Mat tmp;
    if (dst.empty())
        tmp = Mat::zeros(  src.rows*dim, src.cols , CV_32F );
    else if( static_cast<void*>(dst.data) == static_cast<void*>(src.data))  /* output is the same as input, use tmp */
        tmp = Mat::zeros(  src.rows*dim, src.cols , CV_32F );
    else if( dst.rows!=(src.rows*dim) || dst.cols!=src.cols )               /* pre allocated, but size does not macth */
        tmp = Mat::zeros(  src.rows*dim, src.cols , CV_32F );
    else                                                                    /* use pre allocated memory */
        tmp = dst;

    if( conv_size > 1)
        convTri_sse( (const float*)( src.data), (float *)( tmp.data), src.cols, src.rows, conv_size, dim);
    else
        convTri1( (const float*)( src.data), (float *)( tmp.data), src.rows, src.cols, dim, (float)(2.0) ); //means kernel is [1 2 1]

    dst = tmp;
}

void feature_Pyramids::getscales( const Mat &img,                         //in:  image
								  vector<Size> &ap_size,                  //out: the size of per layer
								  vector<int> &real_scal,                 //out: the ID of layer we really compute
								  vector<double> &scales,                 //out: all scales
								  vector<double> &scalesh,                //out: the height of per layer
								  vector<double> &scalesw                 //out: the width of per layer
								  )const
{
	
	int nPerOct =m_opt.nPerOct;
	int nOctUp =m_opt.nOctUp;
	int shrink =m_opt.shrink;
	int nApprox=m_opt.nApprox;
	Size minDS =m_opt.minDS;

	int nscales=(int)floor(nPerOct*(nOctUp+log(min(img.cols/(minDS.width*1.0),img.rows/(minDS.height*1.0)))/log(2))+1);
	Size ap_tmp_size;
	if (nApprox>=nscales)
	{
		cout<<"the approximated scales must < all scales"<<endl;
	}
	CV_Assert(nApprox<nscales);

	double d0=(double)min(img.rows,img.cols);
	double d1=(double)max(img.rows,img.cols);
	for (double s=0;s<nscales;s++)
	{

		/*adjust ap_size*/
		double sc=pow(2.0,(-s)/nPerOct+nOctUp);
		double s0=(cvRound(d0*sc/shrink)*shrink-0.25*shrink)/d0;
		double s1=(cvRound(d0*sc/shrink)*shrink+0.25*shrink)/d0;
		double ss,es1,es2,a=10,val;
		for(int c=0;c<101;c++)
		{
			ss=(double)((s1-s0)*c/101+s0);
			es1=abs(d0*ss-cvRound(d0*ss/shrink)*shrink);
			es2=abs(d1*ss-cvRound(d1*ss/shrink)*shrink);
			if (max(es1,es2)<a)
			{
				a=max(es1,es2);
				val=ss;
			}
		}
		if (scales.empty())
		{
			/*all scales*/
			scales.push_back(val);
			scalesh.push_back(cvRound(img.rows*val/shrink)*shrink/(img.rows*1.0));
			scalesw.push_back(cvRound(img.cols*val/shrink)*shrink/(img.cols*1.0));
			/*save ap_size*/
			ap_size.push_back(Size(cvRound(((img.cols*val)/shrink)),cvRound(((img.rows*val)/shrink))));

		}else{
			if (val!=scales.back())
			{	
				/*all scales*/
				scales.push_back(val);
				scalesh.push_back(cvRound(img.rows*val/shrink)*shrink/(img.rows*1.0));
				scalesw.push_back(cvRound(img.cols*val/shrink)*shrink/(img.cols*1.0));
				/*save ap_size*/
				ap_size.push_back(Size(cvRound(((img.cols*val)/shrink)),cvRound(((img.rows*val)/shrink))));
			}
		}	
	}
   /*compute real & approx scales*/
	nscales=scales.size();

	for (int s=0;s<nscales;s++)
	{
		/*real scale*/
		if (((int)s%(nApprox+1)==0))
		{
			real_scal.push_back((int)s);
		}
		if ((s==(scales.size()-1)&&(s>real_scal[real_scal.size()-1])&&(s-real_scal[real_scal.size()-1]>(nApprox+1)/2)))
		{
			real_scal.push_back((int)s);
		}
	}
} 

void feature_Pyramids::computeGradient( const Mat &img,                   //in:  image
                                        Mat& grad1,						  //out: Gradient magnitude 0
                                        Mat& grad2,                       //out: Gradient magnitude 1
                                        Mat& qangle1,                     //out: Gradient angle 0
                                        Mat& qangle2,                     //out: Gradient angle 1
                                        Mat& mag_sum_s                    //out: Gradient magnitude
										) const
{

	bool gammaCorrection = false;
    Size paddingTL=Size(0,0);
	Size paddingBR=Size(0,0);
	int nbins=m_opt.nbins;
	//CV_Assert( img.type() == CV_8U || img.type() == CV_8UC3 );
	CV_Assert( img.type() == CV_32F || img.type() == CV_32FC3 );

	Size gradsize(img.cols + paddingTL.width + paddingBR.width,
		img.rows + paddingTL.height + paddingBR.height);

	grad1.create(gradsize, CV_32FC1);  // <magnitude*(1-alpha)
	grad2.create(gradsize, CV_32FC1);  //  magnitude*alpha>
	qangle1.create(gradsize, CV_8UC1); // [0..nbins-1] - quantized gradient orientation
	qangle2.create(gradsize, CV_8UC1); // [0..nbins-1] - quantized gradient orientation

	Size wholeSize;
	Point roiofs;
	img.locateROI(wholeSize, roiofs);

	int x, y;
	int cn = img.channels();

	AutoBuffer<int> mapbuf(gradsize.width + gradsize.height + 4);
	int* xmap = (int*)mapbuf + 1;
	int* ymap = xmap + gradsize.width + 2;

	const int borderType = (int)BORDER_REFLECT_101;
	//! 1D interpolation function: returns coordinate of the "donor" pixel for the specified location p.
	for( x = -1; x < gradsize.width + 1; x++ )
		xmap[x] = borderInterpolate(x - paddingTL.width + roiofs.x,
		wholeSize.width, borderType) - roiofs.x;
	for( y = -1; y < gradsize.height + 1; y++ )
		ymap[y] = borderInterpolate(y - paddingTL.height + roiofs.y,
		wholeSize.height, borderType) - roiofs.y;

	// x- & y- derivatives for the whole row
	int width = gradsize.width;
	AutoBuffer<float> _dbuf(width*4);
	float* dbuf = _dbuf;
	Mat Dx(1, width, CV_32F, dbuf);
	Mat Dy(1, width, CV_32F, dbuf + width);
	Mat Mag(1, width, CV_32F, dbuf + width*2);
	Mat Angle(1, width, CV_32F, dbuf + width*3);

	int _nbins = nbins;
	float angleScale = (float)(_nbins/(CV_PI));//0~2*pi


	for( y = 0; y < gradsize.height; y++ )
	{
		const float* imgPtr  = (float*)(img.data + img.step*ymap[y]);
		const float* prevPtr = (float*)(img.data + img.step*ymap[y-1]);
		const float* nextPtr = (float*)(img.data + img.step*ymap[y+1]);

		float* gradPtr1 = (float*)grad1.ptr(y);
		float* gradPtr2 = (float*)grad2.ptr(y);
		uchar* qanglePtr1 = (uchar*)qangle1.ptr(y);
		uchar* qanglePtr2 = (uchar*)qangle2.ptr(y);

		if( cn == 1 )
		{
			for( x = 0; x < width; x++ )
			{
				int x1 = xmap[x];
				dbuf[x] = (float)(imgPtr[xmap[x+1]] - imgPtr[xmap[x-1]]);
				dbuf[width + x] = (float)(nextPtr[x1] - prevPtr[x1]); //??
			}
		}
		else
		{
			for( x = 0; x < width; x++ )
			{
				int x1 = xmap[x]*3;
				float dx0, dy0, dx, dy, mag0, mag;
				const float* p2 = imgPtr + xmap[x+1]*3;
				const float* p0 = imgPtr + xmap[x-1]*3;

				dx0 = (p2[2] - p0[2]);
				dy0 = (nextPtr[x1+2] - prevPtr[x1+2]);
				mag0 = dx0*dx0 + dy0*dy0;

				dx = (p2[1] - p0[1]);
				dy = (nextPtr[x1+1] - prevPtr[x1+1]);
				mag = dx*dx + dy*dy;

				if( mag0 < mag )
				{
					dx0 = dx;
					dy0 = dy;
					mag0 = mag;
				}

				dx = (p2[0] - p0[0]);
				dy = (nextPtr[x1] - prevPtr[x1]);
				mag = dx*dx + dy*dy;

				if( mag0 < mag )
				{
					dx0 = dx;
					dy0 = dy;
					mag0 = mag;
				}

				dbuf[x] = dx0;
				dbuf[x+width] = dy0;
			}
		}
		cartToPolar( Dx, Dy, Mag, Angle, false );

		for( x = 0; x < width; x++ )
		{
			float mag = dbuf[x+width*2];
			float act_ang = (dbuf[x+width*3] > CV_PI ? dbuf[x+width*3]-CV_PI: dbuf[x+width*3]);
			float angle = act_ang*angleScale;
			
			int hidx = cvFloor(angle);
			angle -= hidx;
			gradPtr1[x] = mag*(1.f - angle);
			gradPtr2[x] = mag*angle;

			if( hidx < 0 )
				hidx += _nbins;
			else if( hidx >= _nbins )
				hidx -= _nbins;
			assert( (unsigned)hidx < (unsigned)_nbins );

			qanglePtr1[x] = (uchar)hidx;
			hidx++;
			hidx &= hidx < _nbins ? -1 : 0;
			qanglePtr2[x] = (uchar)hidx;

		}
	}

  


    Mat grad1_smooth, grad2_smooth;

	double normConst=0.005; 

    /*  1 opencv version */
    //--------------------------------
	//convTri(grad1,grad1_smooth,m_normPad);
	//convTri(grad2,grad2_smooth,m_normPad);
    //--------------------------------

    /*  2 sse version */
    int norm_const = 5;
	convTri(grad1,grad1_smooth,norm_const, 1);
	convTri(grad2,grad2_smooth,norm_const, 1);

	//normalization
    
    /* 1 opencv  version */
    //--------------------------------
    //Mat norm_term = grad1_smooth + grad2_smooth + normConst;
	//grad1 = grad1/( norm_term );
	//grad2 = grad2/( norm_term );
    //--------------------------------

    /* 2 sse version, faster */
    //--------------------------------
    Mat norm_term = grad1_smooth + grad2_smooth;
    float *smooth_term = (float*)(norm_term.data);
    float *ptr_grad1 = (float*)grad1.data;
    float *ptr_grad2 = (float*)grad2.data;
    gradMagNorm( ptr_grad1, smooth_term, grad1.rows, grad1.cols, normConst);
    gradMagNorm( ptr_grad2, smooth_term, grad2.rows, grad2.cols, normConst);
    //--------------------------------
	mag_sum_s=grad1+grad2;
}


void feature_Pyramids::computeChannels( const Mat &image,                 //in:  image
									    vector<Mat>& channels             //out: feature channels
										) const
{
	if (image.empty()||(image.channels()<3))
	{
		cout<<"image is empty or image channels < 3"<<endl;
	}
	CV_Assert((!image.empty())||(image.channels()==3));
	/*set para*/
	int nbins=m_opt.nbins;
	int binsize=m_opt.binsize;
	int shrink =m_opt.shrink;
	int smooth=m_opt.smooth;
	/* compute luv and push */
	Mat_<double> grad;
	Mat_<double> angles;
	Mat src,luv;
	
	//cv::TickMeter tm3;
	//tm3.start();
	int channels_addr_rows=(image.rows)/shrink;
	int channels_addr_cols=(image.cols)/shrink;
	Mat channels_addr=Mat::zeros((nbins+4)*channels_addr_rows,channels_addr_cols,CV_32FC1);

	src = Mat(image.rows, image.cols, CV_32FC3);
	image.convertTo(src, CV_32FC3, 1./255);
	cv::cvtColor(src, luv, CV_RGB2Luv);
	channels.clear();

	vector<Mat> luv_channels;
	luv_channels.resize(3);

	cv::split(luv, luv_channels);
	/*  0<L<100, -134<u<220 -140<v<122  */
	/*  normalize to [0, 1] */
	luv_channels[0] *= 1.0/354;
	convTri(luv_channels[0],luv_channels[0],m_km);
	luv_channels[1] = (luv_channels[1]+134)/(354.0);
	convTri(luv_channels[1],luv_channels[1],m_km);
	luv_channels[2] = (luv_channels[2]+140)/(354.0);
	convTri(luv_channels[2],luv_channels[2],m_km);

	for( int i = 0; i < 3; ++i )
	{
		Mat channels_tmp=channels_addr.rowRange(i*channels_addr_rows,(i+1)*channels_addr_rows);
		cv::resize(luv_channels[i],channels_tmp,channels_tmp.size(),0.0,0.0,1);
		channels.push_back(channels_tmp);
	}

	/*compute gradient*/
	Mat mag_sum=channels_addr.rowRange(3*channels_addr_rows,4*channels_addr_rows);

	Mat luv_norm;
	cv::merge(luv_channels,luv_norm);
	
	Mat mag_sum_s;
	
    Mat mag1,mag2,ori1,ori2;
	computeGradient(luv_norm, mag1, mag2, ori1, ori2, mag_sum_s);//mzx 以上共花费64ms

	cv::resize(mag_sum_s,mag_sum,mag_sum.size(),0.0,0.0,INTER_AREA);
	channels.push_back(mag_sum);

	vector<Mat> bins_mat,bins_mat_tmp;
	int bins_mat_tmp_rows=mag1.rows/binsize;
	for( int s=0;s<nbins;s++){
		Mat channels_tmp=channels_addr.rowRange((s+4)*channels_addr_rows,(s+5)*channels_addr_rows);
		if (binsize==shrink)
		{
			bins_mat_tmp.push_back(channels_tmp);
		}else{
			bins_mat.push_back(channels_tmp);
			bins_mat_tmp.push_back(Mat::zeros(bins_mat_tmp_rows,bins_mat_tmp_rows,CV_32FC1));
		}
	}
	//s*s---the number of the pixels of the spatial bin;
	float sc=binsize;
	/*split*/
#define GH \
	bins_mat_tmp[ori1.at<uchar>(row,col)].at<float>((row)/binsize,(col)/binsize)+=(mag1.at<float>(row,col)*(1.0/sc)*(1.0/sc));\
	bins_mat_tmp[ori2.at<uchar>(row,col)].at<float>((row)/binsize,(col)/binsize)+=(mag2.at<float>(row,col)*(1.0/sc)*(1.0/sc));
	for(int row=0;row<(mag1.rows/binsize*binsize);row++){
		for(int col=0;col<(mag1.cols/binsize*binsize);col++){GH;}}

	/*push*/
	for (int c=0;c < (int)nbins;c++)
	{
		/*resize*/
		if (binsize==shrink)
		{
			channels.push_back(bins_mat_tmp[c]);
		}else{
			cv::resize(bins_mat_tmp[c],bins_mat[c],bins_mat[c].size(),0.0,0.0,INTER_AREA);
			channels.push_back(bins_mat[c]);
		}
		
	}
 }


bool feature_Pyramids:: chnsPyramid( const Mat &img,                     //in:  image
									  vector<vector<Mat> >                //out: feature channels pyramid
									  &approxPyramid,                             //contain:really compute && approx
									  vector<double> &scales,             //out: all scales
									  vector<double> &scalesh,			  //out: the height of per layer
									  vector<double> &scalesw) 			  //out: the width of per layer
									  const
 {
	 if (img.empty()||(img.channels()<3))
	 {
		 cout<<"image is empty or image channels < 3"<<endl;
		 return 1;
	 }
	 CV_Assert((!img.empty())||(img.channels()==3));
	 int shrink =m_opt.shrink;     //down_samples
	 int smooth =m_opt.smooth;
	 int nApprox=m_opt.nApprox;
	 Size pad =m_opt.pad;
	 /*check the para*/
	 CV_Assert(nApprox>=0);
	 CV_Assert(shrink>=1); 
	 CV_Assert(pad.width>=0&&pad.height>=0);//pad.size()>0
	 /*clear the input*/
	 scales.clear();
	 approxPyramid.clear();
	 scalesh.clear();
	 scalesw.clear();
	 /*get scales*/
	 vector<Size> ap_size;
	 vector<int> real_scal;
	 getscales(img,ap_size,real_scal,scales,scalesh,scalesw);
	 Mat img_tmp;
	 /*compute lam*/
	//compute real 
	vector<vector<Mat> > chns_Pyramid;
	int chns_num;
	for (int s_r=0;s_r<(int)real_scal.size();s_r++)
	{
		vector<Mat> chns;
		resize(img,img_tmp,ap_size[real_scal[s_r]]*shrink,0.0,0.0,INTER_AREA);
		computeChannels(img_tmp,chns);
		chns_num=chns.size();
		chns_Pyramid.push_back(chns);
	}
	//compute lambdas
	vector<double> lambdas;
	if (nApprox!=0)
	{
		get_lambdas(chns_Pyramid,lambdas,real_scal,scales);

	}
	//compute based-scales
	vector<int> approx_scal;
	for (int s_r=0;s_r<scales.size();s_r++)
	{
		int tmp=s_r/(nApprox+1);
		if (s_r-real_scal[tmp]>((nApprox+1)/2))
		{
			approx_scal.push_back(real_scal[tmp+1]);
		}else{
			approx_scal.push_back(real_scal[tmp]);
		}	
	}
	//compute approxPyramid
	double ratio;
	for (int ap_id=0;ap_id<(int)approx_scal.size();ap_id++)
	{
		vector<Mat> approx_chns;
		approx_chns.clear();
		/*memory is consistent*/
		int approx_rows=ap_size[ap_id].height;
		int approx_cols=ap_size[ap_id].width;
		/*the size of pad*/
		int pad_T=pad.height/shrink;
		int pad_R=pad.width/shrink;
		Mat approx=Mat::zeros(10*(approx_rows+2*pad_T),approx_cols+2*pad_R,CV_32FC1);//因为chns_Pyramind是32F
		for(int n_chans=0;n_chans<chns_num;n_chans++)
		{
			Mat py_tmp=Mat::zeros(approx_rows,approx_cols,CV_32FC1);
			Mat py=approx.rowRange(n_chans*(approx_rows+2*pad_T),(n_chans+1)*(approx_rows+2*pad_T));//pad 以后的图像
			int ma=approx_scal[ap_id]/(nApprox+1);
			resize(chns_Pyramid[ma][n_chans],py_tmp,py_tmp.size(),0.0,0.0,INTER_LINEAR);
			if (nApprox!=0)
			{
				ratio=(double)pow(scales[ap_id]/scales[approx_scal[ap_id]],-lambdas[n_chans]);
				py_tmp=py_tmp*ratio;
			}
			//smooth channels, optionally pad and concatenate channels
			convTri(py_tmp,py_tmp,m_km);
			copyMakeBorder(py_tmp,py,pad_T,pad_T,pad_R,pad_R,IPL_BORDER_CONSTANT);
			approx_chns.push_back(py);
		}		
		approxPyramid.push_back(approx_chns);
	}	
	vector<int>().swap(approx_scal);
	vector<double>().swap(lambdas);
	vector<vector<Mat> >().swap(chns_Pyramid);
	vector<int>().swap(real_scal);
	vector<Size>().swap(ap_size) ;
	
	return 0;
 }
 

 bool feature_Pyramids:: chnsPyramid( const Mat &img,                       //in: image
									 vector<vector<Mat> > &chns_Pyramid,   //out: feature channels pyramid
									 vector<double> &scales                //out: all scales
									 ) const//nApprox==0时
{
	if (img.empty()||(img.channels()<3))
	{
		cout<<"image is empty or image channels < 3"<<endl;
		return 1;
	}
	CV_Assert((!img.empty())||(img.channels()==3));
	int shrink =m_opt.shrink;
	int smooth =m_opt.smooth;
	/*get scales*/
	Size ap_tmp_size;
	vector<Size> ap_size;
	vector<int> real_scal;
	vector<double> scalesh;
	vector<double> scalesw;
	//clear
	scales.clear();
	chns_Pyramid.clear();
	scalesh.clear();
	scalesw.clear();
	getscales(img,ap_size,real_scal,scales,scalesh,scalesw);
	Mat img_tmp;
	//compute real 
 	for (int s_r=0;s_r<(int)scales.size();s_r++)
	{
		vector<Mat> chns;
		cv::resize(img,img_tmp,ap_size[s_r]*shrink,0.0,0.0,INTER_AREA);
		computeChannels(img_tmp,chns);
		for (int c = 0; c < chns.size(); c++)
		{
			convTri(chns[c],chns[c],m_km);
		}
		chns_Pyramid.push_back(chns);
	}
	vector<Size>().swap(ap_size) ;
	return 0;
}


 
void feature_Pyramids::setParas(const channels_opt &in_para)
{
     m_opt=in_para;
}


bool feature_Pyramids::compute_lambdas(const vector<Mat> &fold)           //in: the vector of image
	
 {	 
	 /*set para*/
	 int nApprox=m_opt.nApprox;
	 m_opt.nApprox=0;
	 int nimages=fold.size();//the number of the images used to be train,must>=2;
	 if (nimages<=1)
	 {
		 cout<<"the number of image used to compute lambdas must > 1"<<endl;
		 return 1 ;
	 }
	 CV_Assert(nimages>1);	
	 /*allocate vector*/
	 vector<double> scal;	
	 vector<vector<Mat> > Pyramid;
	 vector<Mat> nimages_pixel_mean;
	 vector<vector<double> > base_data;
	 base_data.resize(3);
	 /*compute chnsPyramid*/
	 Mat image;
	 for (int m=0;m<nimages;m++)
	 {   
		 image= fold[m];
		CV_Assert(chnsPyramid_sse(image,Pyramid,scal)) ;
		 Mat pixel_mean(scal.size(),3,CV_64FC1);
		 /*compute the mean of the n_type,where n_type=3(color,mag,gradhist)*/
		 for (int n=0;n<Pyramid.size();n++)//比例scales
		 {
			 double *pty=(double*)pixel_mean.ptr(n);
			 double *base=(double*)pixel_mean.ptr(0);
			 double size=Pyramid[n][0].rows*Pyramid[n][0].cols*1.0;
			 Scalar lam_color,lam_mag,lam_hist;
			 for (int p=0;p<3;p++)
			 {
				 lam_color+=sum(Pyramid[n][p]);
			 }		
			 lam_mag=sum(Pyramid[n][3]);
			 for (int p = 0; p < 6; p++)
			 {
				 lam_hist+=sum(Pyramid[n][p+4]);
			 }
			 if(n>0){
				 pty[2]=lam_hist[0]/(size*6.0)/(base[2]);
				 pty[1]=lam_mag[0]/(size*1.0)/ (base[1]);
				 pty[0]=lam_color[0]/(size*3.0)/(base[0]);
			 }else{
				 pty[2]=(lam_hist[0]*1.0)/(size*6.0);
				 pty[1]=(lam_mag[0]*1.0)/(size*1.0);
				 pty[0]=(lam_color[0]*1.0)/(size*3.0);
				 base_data[0].push_back(pty[0]);
				 base_data[1].push_back(pty[1]);
				 base_data[2].push_back(pty[2]);
			 }
			 nimages_pixel_mean.push_back(pixel_mean);
		 }
	 }
	 if (scal.size()<=1)
	 {
		 cout<<"the size of image pyramid used to compute lambdas must > 1"<<endl;
		 return 1 ;
	 }
	 CV_Assert(scal.size()>1);	
	 /*get the lambdas*/
	 //1.compute mus & S
	 double num=0;//the number of uesd image
	 Mat mus=Mat::zeros(scal.size()-1,3,CV_64FC1);//
	 Mat s=Mat::ones(scal.size()-1,2,CV_64FC1);//0~35个log（scale）

	 for (int r=0;r<scal.size()-1;r++)
	 {
		 s.at<double>(r,0)=log(scal[r+1])/log(2);
	 }
	 Mat nimages_sum=Mat::zeros(scal.size(),3,CV_64FC1);
	 /*2.remove the small value when scale==1*/
	 vector<Mat> nimages_scale0_meanpixel;
	 double  maxdata0= *max_element( base_data[0].begin(),base_data[0].end());
	 double  maxdata1= *max_element( base_data[1].begin(), base_data[1].end());
	 double  maxdata2= *max_element( base_data[2].begin(), base_data[2].end());
	 for (int n=0;n<nimages;n++)
	 {
		 if (base_data[0][n]<maxdata0/50.0 
			 ||base_data[1][n]<maxdata1/50.0 
			 ||base_data[2][n]<maxdata2/50.0) 
			 break;
		 num++;
		 nimages_sum+=nimages_pixel_mean[n];
	 }		
	 log(nimages_sum/num,nimages_sum);	
	 nimages_sum=nimages_sum/log(2);	
	 mus=nimages_sum.rowRange(1,scal.size());	
	 /*3.get lam*/
	 lam.clear();
	 lam.resize(3);
	 for (int n=0;n<3;n++)
	 {
		 Mat mus_omega=mus.colRange(n,n+1);
		 Mat lam_omgea=(s.t()*s).inv()*(s.t())*mus_omega;
		 double a=-lam_omgea.at<double>(0,0);
		 lam[n]=a;			
	 }
	 /*4.print lam*/
	 for (int n=0;n<3;n++)
	 {
		 cout<<"lam:"<<lam[n]<<endl;
	 }
	 /*5.recover the para*/
	  m_opt.nApprox=nApprox;
	  if (m_opt.nApprox!=nApprox)
	  {
		  cout<<"compute lambdas was finished,but para never recover"<<endl;
		  return 1;
	  }
 }
 
 

const channels_opt& feature_Pyramids::getParas() const
{
	 return m_opt;
}
feature_Pyramids::feature_Pyramids()
{
    int norm_pad_size = 5;
	m_normPad = get_Km(norm_pad_size);
	m_opt = channels_opt();
    m_km = get_Km( m_opt.smooth );
}
feature_Pyramids::~feature_Pyramids()
{
}

