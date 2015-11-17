#ifndef SHAPE_PREDICTOR_HPP
#define SHAPE_PREDICTOR_HPP

#include <vector>
#include <limits>
#include <sstream>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#define SHAPE_LENGTH 136
typedef cv::Matx<float,SHAPE_LENGTH,1> shape_type;

using namespace std;
using namespace cv;


namespace impl
{
	/* store the node's information*/
	struct split_feature
	{
		unsigned long idx1;
		unsigned long idx2;
		float threshold;
	};

	/*!
		convert int to string
	!*/
	inline string _to_string( int x)
	{
		stringstream ss;ss<<x;string str;ss>>str;
		return str;
	}

	/*a tree is just a std::vector<impl::split_feature>.  We use this function to navigate the tree nodes*/
	/*!
		- returns the index of the left child of the binary tree node idx
	!*/
	inline unsigned long left_child( unsigned long idx){return idx*2+1;}

	/*!
		- returns the index of the left child of the binary tree node idx
	!*/
	inline unsigned long right_child(unsigned long idx){return idx*2+2;}

	struct regression_tree
	{
		vector<split_feature> splits;
		vector<shape_type> leaf_values;

		/*
			- runs through the tree and returns the vector at the leaf we end up in.
		*/
		inline const shape_type& operator()(  const vector<float> &feature_pixel_values) const
		{
			unsigned long i=0;
			while( i < splits.size())
			{
				if( feature_pixel_values[splits[i].idx1] - feature_pixel_values[splits[i].idx2] > splits[i].threshold)
					i = left_child(i);
				else
					i = right_child(i);
			}
			return leaf_values[i-splits.size()];
		}
	};

	/*---------  helper functions --------*/
	/*!
		- returns the idx-th point from the shape vector.
	!*/
	inline Vec2f location( const shape_type &shape, unsigned long idx)
	{
		Vec2f pt;
		pt[0] = shape(idx*2,0);
		pt[1] = shape(idx*2+1,0);
		return pt;
	}

	/*!
		swap two Matx
	!*/
	inline void swap_shape( shape_type &shape1, shape_type &shape2)
	{
		shape_type temp_shape( shape1.val);
		shape1 = shape2;
		shape2 = temp_shape;
	}
		
	/*! 
		- find the nearest part of the shape to this pixel
	!*/
	inline unsigned long nearest_shape_point( const shape_type &shape, const Vec2f &pt)
	{
		double best_dist = std::numeric_limits<double>::infinity();
		const unsigned long num_shape_parts = shape.rows/2;
		unsigned long best_idx = 0;
		Vec2f shape_pt;
		for( unsigned long j=0;j<num_shape_parts;j++)
		{
			shape_pt[0] = shape(j*2,0);
			shape_pt[1] = shape(j*2+1,0);
			const double dist = cv::norm( shape_pt - pt);
			if(dist<best_dist)
			{
				best_dist = dist;
				best_idx  = j;
			}
		}
		return best_idx;
	}

	/*!
		- transform the pixel_coordinates into shape relative encodings
	!*/
	inline void create_shape_relative_encoding(
		const shape_type &shape,	/* in : input shape*/
		const vector<Vec2f> &pixel_coordinates,		/* in : pixel_coordinates*/
		vector<unsigned long> &anchor_idx,          /* out: nearest landmark's index*/
		vector<Vec2f> &deltas)						/* out: nearest landmark's relative distance, (delta_x, delta_y)' */
	{
		anchor_idx.resize(pixel_coordinates.size());
		deltas.resize(pixel_coordinates.size());

		for( unsigned long i=0;i<pixel_coordinates.size();i++)
		{
			anchor_idx[i] = nearest_shape_point( shape, pixel_coordinates[i]);
			deltas[i] = pixel_coordinates[i] - location(shape, anchor_idx[i]);
		}
	}

	/*!
		- find the transform Matrix(m and b) between the two shape
			y = m*x + b, y->transfromed vector, x-> original vector
	!*/
	inline void find_tfrom_between_shapes(  const Mat &from_shape,		/* in : from shape*/
											const Mat &to_shape,		/* in : to shape*/
											Matx<float, 2, 2> &tran_m,	/* out: m   to_sample = m*from_sample + b  */
											Matx<float, 2, 1> &tran_b)	/* out: b     */
	{
		assert( from_shape.rows == to_shape.rows && (from_shape.rows%2)==0);
		Mat mean_from, mean_to;
		Mat cov = Mat::zeros(2,2,CV_32F);
		double sigma_from = 0, sigma_to = 0;

		/* compute the mean and cov*/
		const Mat from_shape_points = from_shape.reshape( 0, from_shape.rows/2);
		const Mat to_shape_points = to_shape.reshape(0, to_shape.rows/2);
		cv::reduce( from_shape_points, mean_from, 0, CV_REDUCE_AVG);
		cv::reduce( to_shape_points, mean_to, 0, CV_REDUCE_AVG);
		
		double temp_dis;
		for( long i=0;i<from_shape_points.rows;i++)
		{
			temp_dis = cv::norm( mean_from, from_shape_points.row(i), NORM_L2);
			sigma_from += temp_dis*temp_dis;
			temp_dis = cv::norm( mean_to, to_shape_points.row(i), NORM_L2);
			sigma_to += temp_dis*temp_dis;
			cov += (to_shape_points.row(i).t() - mean_to.t())*( from_shape_points.row(i) - mean_from);
		}
		sigma_from = sigma_from/to_shape_points.rows;
		sigma_to   = sigma_to/to_shape_points.rows;
		cov        = cov/to_shape_points.rows;

		/*  compute the affine matrix*/
		Mat u = Mat::zeros(2,2,CV_32F);
		Mat vt = Mat::zeros(2,2,CV_32F);
		Mat d = Mat::zeros(2,2,CV_32F);
		Mat r = Mat::zeros(2,2,CV_32F);
		Mat s = Mat::eye(2,2,CV_32F);

		SVD::compute( cov, d, u, vt);
		if( cv::determinant(cov) < 0)
		{
			if( d.at<float>(1,1) < d.at<float>(0,0))
				s.at<float>(1,1) = -1;
			else
				s.at<float>(0,0) = -1;
		}
		r = u*s*vt; 
		double c=1;
	
		if (sigma_from !=0)
			c = 1.0/sigma_from*cv::trace( Mat::diag(d)*s).val[0];
		Mat b = Mat::zeros(2,1,CV_32F);
		b = (mean_to.t() - c*r*mean_from.t());

		Mat m = c*r;

		tran_m = Matx<float,2,2>((float*)(m.clone().ptr()));
		tran_b = Matx<float,2,1>((float*)(b.clone().ptr()));
	}

	/*!
		compute the transform that maps (0,0) to rect.tl_corner() and (1,1) to rect.br_corner()
		y = m*x +b 
	!*/
	inline void unnormalizing_tform(	const Rect &rect,		/* in : original rect*/
										Matx<float,2,2> &m,		/* out: transform matrix*/
										Matx<float,2,1> &b)		/* out: bias item*/
	{
		Mat from_shape = (Mat_<float>(6,1)<<0,0,1,0,1,1);
		Mat to_shape = (Mat_<float>(6,1)<<rect.x, rect.y, rect.x+rect.width-1,rect.y, rect.x+rect.width-1, rect.y+rect.height-1);
		find_tfrom_between_shapes( from_shape, to_shape, m, b);
	}

	/*! 
		compute the transform that maps rect.tl_corner() to (0,0) and rect.br_corner() to (1,1)
	!*/
	inline void normalizing_tform(	const Rect &rect,		/* in : original rect*/
									Matx<float,2,2> &m,		/* out: transform matrix*/
									Matx<float,2,1> &b)		/* out: bias item*/
	{
		Mat from_shape = (Mat_<float>(6,1)<<rect.x, rect.y, rect.x+rect.width-1,rect.y, rect.x+rect.width-1, rect.y+rect.height-1);
		Mat to_shape = (Mat_<float>(6,1)<<0,0,1,0,1,1);
		find_tfrom_between_shapes( from_shape, to_shape, m, b);
	}

	/*!
		compute the m*vector+b;
	!*/
	inline Vec2f transfrom_vector( const Vec2f &vector,
									const Matx<float,2,2> &m,
									const Matx<float,2,1> &b = Matx<float,2,1>::zeros())
	{
		Vec2f tran_vector;
		tran_vector[0] = m(0,0)*vector[0] +m(0,1)*vector[1] + b(0,0);
		tran_vector[1] = m(1,0)*vector[0] +m(1,1)*vector[1] + b(1,0);
		return tran_vector;
	}

	/*!
		if the image contains the point
	!*/
	inline bool in_image(	const Mat &img,
							const Vec2f &pt)
	{
		return (pt[0] >=0 && pt[0] <= img.cols-1 && 
				pt[1]>=0 && pt[1] <= img.rows-1);
	}
	
	/*!
		extract the pixel values 
		for all valid i:
		- #feature_pixel_values[i] == the value of the pixel in img that
		corresponds to the pixel identified by reference_pixel_anchor_idx[i]
		and reference_pixel_deltas[i] when the pixel is located relative to
		current_shape rather than reference_shape.
	!*/
	inline void extract_feature_pixel_values(	const Mat &img,			
										const Rect &rect,
										const shape_type &current_shape,
										const shape_type &reference_shape,
										const vector<unsigned long> &reference_pixel_anchor_idx,
										const vector<Vec2f> &reference_pixel_deltas,
										vector<float> &feature_pixel_values)
	{
		Matx<float,2,2> trans_m,pt_affine_m;
		Matx<float,2,1> trans_b, pt_affine_b;
		find_tfrom_between_shapes( Mat(reference_shape), Mat(current_shape), trans_m, trans_b);
		unnormalizing_tform( rect, pt_affine_m, pt_affine_b);

		feature_pixel_values.resize(reference_pixel_deltas.size());
		for( unsigned long i=0;i<feature_pixel_values.size();i++)
		{
			// Compute the point in the current shape corresponding to the i-th pixel and
			// then map it from the normalized shape space into pixel space.
			Vec2f pt = transfrom_vector(	transfrom_vector(reference_pixel_deltas[i], trans_m)+location(current_shape,reference_pixel_anchor_idx[i]),
										pt_affine_m,
										pt_affine_b);

			if (in_image(img, pt))
				feature_pixel_values[i] = img.at<uchar>( cvRound(pt[1]), cvRound(pt[0]));
			else
				feature_pixel_values[i] = 0;
		}
	}
}; // end of impl

/*-----------------------------------------------------------------*/

class shape_predictor
{
public:
	shape_predictor();

	/*!
	for all valid i:
	- all the index values in forests[i] are less than pixel_coordinates[i].size()
	- for all valid i and j: 
	- forests[i][j].leaf_values.size() is a power of 2.
	(i.e. we require a tree with all the levels fully filled out.
	- forests[i][j].leaf_values.size() == forests[i][j].splits.size()+1
	(i.e. there need to be the right number of leaves given the number of splits in the tree)
	!*/
	shape_predictor(const shape_type &init_shape, 
					const vector<vector<impl::regression_tree> > &_forests,
					const vector<vector<Vec2f > >& _pixel_coordinates);
	~shape_predictor();

	/*!
		return the number of points in the shape
	!*/
	unsigned long num_parts() const;

	/*!
		runs the tree predictor on the detection rect inside img and returns a shape
	!*/
	shape_type operator()( const Mat &img, const Rect rect) const;


	/*!
		save the model 
	!*/
	bool save_model( const string &model_path) const;

	/*!
		load the model
	!*/
	bool load_model( const string &model_path);


	/*!
		get eyes center
	!*/
	static void get_eye_center( const shape_type &shape,	/* in */
										Point &eye1,		/* out */
										Point &eye2 )		/* out */
	{
		eye1.x = (shape(36*2,0)+shape(37*2,0)+shape(38*2,0)+shape(39*2,0)+shape(40*2,0)+shape(41*2,0))/4;
		eye1.y = (shape(36*2+1,0)+shape(37*2+1,0)+shape(38*2+1,0)+shape(39*2+1,0)+shape(40*2+1,0)+shape(41*2+1,0))/4;

		eye2.x = (shape(42*2,0)+shape(43*2,0)+shape(44*2,0)+shape(45*2,0)+shape(46*2,0)+shape(47*2,0))/4;
		eye2.y = (shape(42*2+1,0)+shape(43*2+1,0)+shape(44*2+1,0)+shape(45*2+1,0)+shape(46*2+1,0)+shape(47*2+1,0))/4;
	}

	/*!
		rotate the image according to the eye point
	!*/
	static void rotate_image(	Point p1,				/* in : left eye*/
								Point p2,				/* in : right eye*/
								const Mat &face_region, /* in : image*/
								int desired_width,		/* in : desired face image width*/
								Mat &warp_face) 		/* out: output aligned face image*/
	{
		cv::Point2f eyescenter;
		eyescenter.x = (p1.x+p2.x)*0.5f;
		eyescenter.y = (p1.y+p2.y)*0.5f;

		double dy = p2.y - p1.y;
		double dx = p2.x - p1.x;

		double len = sqrt(dx*dx+dy*dy);
		double angle =  atan2(dy,dx)*180.0/CV_PI;

        // desmond
		const double DESIRED_LEFT_EYE_X = 0.3;     // align 细致的参数
		const double DESIRED_LEFT_EYE_Y = 0.35;
        
        // yy
		//const double DESIRED_LEFT_EYE_X = 0.32;     // align 细致的参数
		//const double DESIRED_LEFT_EYE_Y = 0.3;

		//const double DESIRED_LEFT_EYE_X = 0.4;     //  pre align 的参数
		//const double DESIRED_LEFT_EYE_Y = 0.42;

		const double DESIRED_RIGHT_EYE_X=1.0f-DESIRED_LEFT_EYE_X; 

		int DESIRED_FACE_WIDTH=desired_width;
		int DESIRED_FACE_HEIGHT= DESIRED_FACE_WIDTH;  

		double desiredLen=(DESIRED_RIGHT_EYE_X-DESIRED_LEFT_EYE_X);//目标图像两个眼睛直接的比例距离,乘以WIDTH=70即得到距离  
		double scale=desiredLen*DESIRED_FACE_WIDTH/len;//通过目的图像两个眼睛距离除以原图像两个眼睛的距离，得到旋转矩阵的尺度因子.  


		cv::Mat rot_mat = cv::getRotationMatrix2D(eyescenter, angle, scale);//绕原图像两眼连线中心点旋转，旋转角度为angle，缩放尺度为scale  

		double ex=DESIRED_FACE_WIDTH* 0.5f - eyescenter.x;//获取x方向的平移因子,即目标两眼连线中心点的x坐标—原图像两眼连线中心点x坐标  
		double ey = DESIRED_FACE_HEIGHT*DESIRED_LEFT_EYE_Y-eyescenter.y;//获取x方向的平移因子,即目标两眼连线中心点的x坐标—原图像两眼连线中心点x坐标  
		rot_mat.at<double>(0, 2) += ex;//将上述结果加到旋转矩阵中控制x平移的位置  
		rot_mat.at<double>(1, 2) += ey;//将上述结果加到旋转矩阵中控制y平移的位置  

		warp_face = cv::Mat(DESIRED_FACE_WIDTH, DESIRED_FACE_WIDTH,face_region.type());  
		cv::warpAffine(face_region, warp_face, rot_mat, warp_face.size());  
	}


	/*!
		rotate the image according to the mouth and eyebrow center
	!*/
	static void rotate_image2(	Point eyebrow_center,	/* in : eyeborw center */
								Point mouth_center,		/* in : mouth center*/
								const Mat &face_region, /* in : image*/
								int desired_width,		/* in : desired face image width*/
								Mat &warp_face) 		/* out: output aligned face image*/
	{
        double dy = eyebrow_center.x - mouth_center.x;
        double dx = eyebrow_center.y - mouth_center.y;

		double len = sqrt(dx*dx+dy*dy);
		double angle =  -1*atan2(-1*dy,dx)*180.0/CV_PI +180;

        /* 256 --> 0.28, 0.23*/
        /* 268 --> 0.2899, 0.2421 */

		const double ratio_to_top   = 0.4;     // eyebrow_center to the top of the image
		const double ratio_of_middle = 0.35;   // ratio of the distance between eyebrow_center to mouth_center
        
		double scale =  desired_width*ratio_of_middle / len;

		cv::Mat rot_mat = cv::getRotationMatrix2D(eyebrow_center, angle, scale);//绕原图像两眼连线中心点旋转，旋转角度为angle，缩放尺度为scale  

		double ex=desired_width* 0.5f - eyebrow_center.x;//获取x方向的平移因子,即目标两眼连线中心点的x坐标—原图像两眼连线中心点x坐标  
		double ey =  desired_width*ratio_to_top -eyebrow_center.y;//获取x方向的平移因子,即目标两眼连线中心点的x坐标—原图像两眼连线中心点x坐标  
		rot_mat.at<double>(0, 2) += ex;//将上述结果加到旋转矩阵中控制x平移的位置  
		rot_mat.at<double>(1, 2) += ey;//将上述结果加到旋转矩阵中控制y平移的位置  

		warp_face = cv::Mat(desired_width, desired_width ,face_region.type());  
		cv::warpAffine(face_region, warp_face, rot_mat, warp_face.size());  
	}


	/*!
		align the face using eye center
	*/
	static void align_face( const shape_type &shape, 
							const Mat &input_image,
							int desired_width,
							Mat &aligned_face)
	{
		Point left_eye, right_eye;
		get_eye_center( shape, left_eye, right_eye);
		rotate_image(left_eye, right_eye,input_image, desired_width, aligned_face);
	}


	/*!
		align the face using eyebrow center and mouth center
	*/
	static void align_face2( const shape_type &shape, 
							 const Mat &input_image,
							 int desired_width,
							 Mat &aligned_face)
	{
        Point mouth_center;
        Point eyebrow_center;

        mouth_center.x = (shape(50*2,0) + shape(51*2,0) + shape(52*2,0))/3;
        mouth_center.y = (shape(50*2+1,0) + shape(51*2+1,0) + shape(52*2+1,0))/3;
        eyebrow_center.x = shape(27*2,0);
        eyebrow_center.y = shape(27*2+1,0);
        
		rotate_image2(eyebrow_center, mouth_center,input_image, desired_width, aligned_face);
	}

	/*!
		draw the shape and rect on a image, help to debug
	!*/
	static Mat draw_shape( 
		const Mat &img, 
		const vector<Rect> &rects, 
		const vector<shape_type> &shapes,
		Scalar color = Scalar( 255, 255, 255)
		)
	{
		Mat draw = img.clone();
		
		for (unsigned long i=0;i<rects.size();i++)
		{
			cv::rectangle( draw, rects[i], color, 3);
			for (unsigned long j=0;j<shapes[i].rows/2;j++)
			{
				Point pt( (int)shapes[i](j*2,0), (int)shapes[i](2*j+1,0));
				cv::circle( draw, pt, 3, color);
			}
		}
		return draw;
	}

private:
	shape_type m_initial_shape;
	vector<vector<impl::regression_tree> > m_forests;
	vector<vector<unsigned long> > m_anchor_idx; 
	vector<vector<Vec2f > > m_deltas;
};

class shape_predictor_trainer
{
	/*!
	This thing really only works with unsigned char or rgb_pixel images (since we assume the threshold 
	should be in the range [-128,128]).
	!*/
public:
	shape_predictor_trainer();
	~shape_predictor_trainer();

	/*---------------------------- set and get functions ------------------------------*/
	unsigned long get_cascade_depth () const { return m_cascade_depth; }
	void set_cascade_depth ( unsigned long depth )
	{
		assert(depth>0);
		m_cascade_depth = depth;
	}

	unsigned long get_tree_depth () const { return m_tree_depth; }
	void set_tree_depth (unsigned long depth)
	{
		assert(depth>0);
		m_tree_depth = depth;
	}

	unsigned long get_num_trees_per_cascade_level () const { return m_num_trees_per_cascade_level; }
	void set_num_trees_per_cascade_level ( unsigned long num )
	{
		assert(num>0);
		m_num_trees_per_cascade_level = num;
	}

	double get_nu () const { return m_nu; } 
	void set_nu (double nu)
	{
		assert(nu>0);
		m_nu = nu;
	}

	unsigned long get_oversampling_amount () const { return m_oversampling_amount; }
	void set_oversampling_amount ( unsigned long amount)
	{
		assert(amount>0);
		m_oversampling_amount = amount;
	}

	unsigned long get_feature_pool_size () const { return m_feature_pool_size; }
	void set_feature_pool_size ( unsigned long size) 
	{
		assert(size>0);
		m_feature_pool_size = size;
	}

	double get_lambda () const { return m_lambda; }
	void set_lambda (double lambda)
	{
		assert(lambda>0);
		m_lambda = lambda;
	}

	unsigned long get_num_test_splits () const { return m_num_test_splits; }
	void set_num_test_splits (unsigned long num)
	{
		assert(num>0);
		m_num_test_splits = num;
	}

	double get_feature_pool_region_padding () const { return m_feature_pool_region_padding; }
	void set_feature_pool_region_padding ( double padding )
	{
		m_feature_pool_region_padding = padding;
	}

	void set_rand_seed( unsigned long seed)
	{
		m_rng.state = seed;
	}
	/*---------------------------- set and get functions  end ------------------------------*/

	/*! interface and helper function for training! */
	shape_predictor train( 
		const vector<Mat> &images,				/* in : input images*/
		const vector<vector<Rect> > &rects,		/* in : rect for the target */
		const vector<vector<shape_type> > &shapes		/* in : shape for the corresponding rect*/
		);

	/*! test the predictor, return the mean error !*/
	double test_shape_predictor(
		const shape_predictor &sp,
		const vector<Mat> & images,
		const vector<vector<Rect> > rects,
		const vector<vector<shape_type> > shapes
		) const;

private:
	/*---------------------------------static helper--------------------------------------*/
	/* convert the original shape to normalized shape */
	static shape_type object_to_shape( const Rect &rect,			/* in : target rect*/
								const shape_type &original_shape)	/* in : original shape */
	{
		shape_type normalized_shape = shape_type::zeros();
		Matx<float,2,2> tran_m;
		Matx<float,2,1>tran_b;
		impl::normalizing_tform( rect, tran_m, tran_b);

		Vec2f temp_vector;
		for( long i=0;i<original_shape.rows/2;i++)
		{
			temp_vector[0] = original_shape(i*2,0);
			temp_vector[1] = original_shape(i*2+1,0);
			temp_vector = impl::transfrom_vector( temp_vector, tran_m, tran_b);
			normalized_shape(i*2,0) = temp_vector[0];
			normalized_shape(i*2+1,0) = temp_vector[1];
		}
		return normalized_shape;
	}

	/*------------------------------------inner struct ------------------------------------*/
	struct training_sample
	{
		/*!
        - feature_pixel_values.size() == get_feature_pool_size()
        - feature_pixel_values[j] == the value of the j-th feature pool
		  pixel when you look it up relative to the shape in current_shape.
        !*/
		unsigned long image_idx;
		Rect rect;						//the position of the object in the image_idx-th image.  All shape coordinates are coded relative to this rectangle.
		shape_type target_shape;		//The truth shape.  Stays constant during the whole training process.
		shape_type current_shape;		// current shape
		vector<float> feature_pixel_values;

		void swap( training_sample &item)
		{
			std::swap( image_idx, item.image_idx);
			std::swap(rect, item.rect);
			feature_pixel_values.swap(item.feature_pixel_values);

			/* swap two Matx*/
			impl::swap_shape( target_shape, item.target_shape);
			impl::swap_shape( current_shape, item.current_shape);
		}

	};

	/*------------------------------------ real workers ------------------------------------*/
	impl::regression_tree make_regression_tree (
		vector<training_sample>& samples,			/* in&out: training samples, may change the current shape*/
		const vector<Vec2f>& pixel_coordinates		/* in    ; pixel coordinates, for feature selection */
		) const;

	/*!
		randomly generate a feature for split
	!*/
	impl::split_feature randomly_generate_split_feature (
		const vector<Vec2f>& pixel_coordinates	
		) const;

	/*!
		generate a bunch of random splits and test them and return the best one
	!*/
	impl::split_feature generate_split (
		const vector<training_sample>& samples,	/* in : inpute samples*/
		unsigned long begin,					/* in : start of the feature selection range */
		unsigned long end,						/* in : end of the feature selection range*/
		const vector<Vec2f>& pixel_coordinates,	/* in : pixel candicates*/
		const shape_type& sum,					/* in : sum of the shap */
		shape_type& left_sum,					/* out : shape sum in the left node*/
		shape_type& right_sum					/* out : shape sum in the right node*/
		) const;

	/*!
		partition sample use split, return the mid point index, 
		put samples which > threshold in front of mid point
	!*/
	unsigned long partition_samples (
		const impl::split_feature& split,
		std::vector<training_sample>& samples,		/* in &out : will be change to [ left_nodes , right_nodes]*/
		unsigned long begin,						/* in : start index of sample */
		unsigned long end							/* in : end index of samples */
		) const;

	/*! 
	   generate training examples, use ground truth as target_shape, 
	   randomly sample other shapes including mean shape as start point
	   !*/
	shape_type populate_training_sample_shapes(
		const vector<vector<Rect> > &object_rects,	/* in : rects of target*/
		const vector<vector<shape_type> > &object_shapes,	/* in : shape of target, ensure object_shape has the same size as object_rect */
		vector<training_sample>& samples			/* out: samples*/
		) const;

	void randomly_sample_pixel_coordinates (
		vector<Vec2f>& pixel_coordinates,
		float min_x,
		float max_x,
		float min_y,
		float max_y
		) const;

	vector<vector<Vec2f > > randomly_sample_pixel_coordinates (
		const shape_type &initial_shape 
		) const;

	unsigned long m_cascade_depth;
	unsigned long m_tree_depth;
	unsigned long m_num_trees_per_cascade_level;
	double m_nu;
	unsigned long m_oversampling_amount;
	unsigned long m_feature_pool_size;
	double m_lambda;
	unsigned long m_num_test_splits;
	double m_feature_pool_region_padding;
	bool m_verbose;
	mutable cv::RNG m_rng;
};


#endif // !SHAPE_PREDICTOR_HPP
