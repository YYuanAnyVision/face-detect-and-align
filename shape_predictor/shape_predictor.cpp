#include <iostream>
#include <deque>
#include <time.h>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "shape_predictor.hpp"

using namespace std;

shape_predictor::shape_predictor()
{
}

shape_predictor::~shape_predictor()
{
}

/*!
	save the model 
!*/
bool shape_predictor::save_model( const string &model_path) const
{
	FileStorage fs( model_path, FileStorage::WRITE);
	if ( !fs.isOpened())
	{
		cout<<"Can not open the model "<<model_path<<endl;
		return false;
	}

	//1 write the initial_shape
	fs<<"m_initial_shape"<<Mat(m_initial_shape);

	//2 write deltas
	fs<<"m_deltas";
	fs<<"{";
	for (unsigned long i=0;i<m_deltas.size();i++)
	{
		fs<<"m_deltas_"+impl::_to_string(i);
		fs<<m_deltas[i];
	}
	fs<<"}";

	//3 write m_anchor_idx
	fs<<"m_anchor_idx";
	fs<<"{";
	for (unsigned long i=0;i<m_anchor_idx.size();i++)
	{
		fs<<"m_anchor_idx_"+impl::_to_string(i);
		fs<<"[";
		for (unsigned long j=0;j<m_anchor_idx[i].size();j++)
		{
			// since Opencv does not overwrite the write() function for unsigned long, 
			// but int should also be safe and enough here ...
			fs<<(int)m_anchor_idx[i][j];
		}
		fs<<"]";
	}
	fs<<"}";

	//4 write the m_forest
	fs<<"m_forest";
	fs<<"{";
	for (unsigned long i=0;i<m_forests.size();i++)
	{
		fs<<"stage_"+impl::_to_string(i);
		fs<<"{";
		for (unsigned long j=0;j<m_forests[i].size();j++)
		{
			fs<<"tree_"+impl::_to_string(j);
			fs<<"{";
			{
				// idx1
				fs<<"idx1";
				fs<<"[";
				for (unsigned long iter=0;iter<m_forests[i][j].splits.size();iter++)
				{
					fs<<(int)m_forests[i][j].splits[iter].idx1;
				}
				fs<<"]";

				// idx2
				fs<<"idx2";
				fs<<"[";
				for (unsigned long iter=0;iter<m_forests[i][j].splits.size();iter++)
				{
					fs<<(int)m_forests[i][j].splits[iter].idx2;
				}
				fs<<"]";
			
				// threshold
				fs<<"threshold";
				fs<<"[";
				for (unsigned long iter=0;iter<m_forests[i][j].splits.size();iter++)
				{
					fs<<m_forests[i][j].splits[iter].threshold;
				}
				fs<<"]";

				// leaf values
				fs<<"leaf";
				fs<<"[";
				for (unsigned long iter=0;iter<m_forests[i][j].leaf_values.size();iter++)
				{
					fs<<Mat(m_forests[i][j].leaf_values[iter]);
				}
				fs<<"]";
			}
			fs<<"}";
		}
		fs<<"}";
	}
	fs<<"}";
	cout<<"Model saved"<<endl;
	return true;
}

/*!
	load the model
!*/
bool shape_predictor::load_model( const string &model_path)
{
	FileStorage fs( model_path, FileStorage::READ);
	if ( !fs.isOpened())
	{
		cout<<"Can not open the model "<<model_path<<endl;
		return false;
	}

	//1 read the initial_shape
	Mat initial_shape;
	fs["m_initial_shape"]>>initial_shape;
	m_initial_shape = shape_type((float*)initial_shape.ptr());

	//2 read deltas
	m_deltas.clear();
	FileNode deltas_node = fs["m_deltas"];
	if ( deltas_node.empty())
	{
		cout<<"Error reading deltas nodes"<<endl;
		return false;
	}
	for( FileNodeIterator deltas_iter=deltas_node.begin(); deltas_iter!=deltas_node.end();deltas_iter++)
	{
		vector<Vec2f> t_delta;
		*deltas_iter>>t_delta;
		m_deltas.push_back(t_delta);
	}

	// 3 read m_anchor_idx
	m_anchor_idx.clear();
	FileNode anchor_node = fs["m_anchor_idx"];
	if ( anchor_node.empty())
	{
		cout<<"Error reading anchor nodes"<<endl;
		return false;
	}
	for( FileNodeIterator anchor_iter=anchor_node.begin(); anchor_iter!=anchor_node.end();anchor_iter++)
	{
		vector<int> t_anchor;
		*anchor_iter>>t_anchor;
		vector<unsigned long> long_delta;
		for (unsigned long i = 0; i < t_anchor.size(); i++)
		{
			long_delta.push_back( (unsigned long)(t_anchor[i]));
		}
		m_anchor_idx.push_back(long_delta);
	}

	// 4 load the forest
	m_forests.clear();
	FileNode forest_node = fs["m_forest"];
	if(forest_node.empty())
	{
		cout<<"Can not open the forest node"<<endl;
		return false;
	}
	for ( FileNodeIterator forest_iter=forest_node.begin(); forest_iter!=forest_node.end();forest_iter++)
	{
		if ((*forest_iter).empty())
		{
			cout<<"Forest Node empty "<<endl;
			return false;
		}
		vector<impl::regression_tree> t_trees;
		for (FileNodeIterator tree_iter=(*forest_iter).begin(); tree_iter!=(*forest_iter).end();tree_iter++)
		{
			// idx1 
			vector<int> t_idx1;
			FileNode idx1_node = (*tree_iter)["idx1"];
			idx1_node>>t_idx1;

			// idx2 
			vector<int> t_idx2;
			FileNode idx2_node = (*tree_iter)["idx2"];
			idx2_node>>t_idx2;
			
			// threshold
			vector<float> t_threshold;
			FileNode thres_node = (*tree_iter)["threshold"];
			thres_node>>t_threshold;

			// leaf values
			vector<Mat> t_leafs;
			FileNode leaf_node = (*tree_iter)["leaf"];
			leaf_node>>t_leafs;

			if ( idx1_node.empty() || idx2_node.empty() || thres_node.empty() || leaf_node.empty())
			{
				cout<<"Can not read tree "<<endl;
				return false;
			}

			impl::regression_tree full_tree;
			full_tree.splits.resize( t_idx1.size() );
			full_tree.leaf_values.resize( t_leafs.size());
			for (unsigned long t=0;t<t_threshold.size();t++)
			{
				full_tree.splits[t].idx1 = (unsigned long)t_idx1[t];
				full_tree.splits[t].idx2 = (unsigned long)t_idx2[t];
				full_tree.splits[t].threshold = t_threshold[t];
			}

			for (unsigned long t=0;t<t_leafs.size();t++)
			{
				full_tree.leaf_values[t] = shape_type( (float*)(t_leafs[t].ptr()));
			}
			t_trees.push_back(full_tree);
		}
		m_forests.push_back( t_trees);
	}
	cout<<"Model loaded"<<endl;
	return true;
}

shape_predictor::shape_predictor(const shape_type &init_shape, 
								 const vector<vector<impl::regression_tree> > &_forests,
								 const vector<vector<Vec2f > >& _pixel_coordinates):m_initial_shape(init_shape), m_forests(_forests)
{
	m_anchor_idx.resize(_pixel_coordinates.size());
	m_deltas.resize(_pixel_coordinates.size());

	// Each cascade uses a different set of pixels for its features.  We compute
	// their representations relative to the initial shape now and save it.
	for (unsigned long i = 0; i < _pixel_coordinates.size(); ++i)
		impl::create_shape_relative_encoding(m_initial_shape, _pixel_coordinates[i], m_anchor_idx[i], m_deltas[i]);
}

unsigned long shape_predictor::num_parts() const
{
	return m_initial_shape.rows/2;
}

shape_type shape_predictor::operator()( const Mat &img, const Rect rect) const
{
	using namespace impl;


    Mat for_process_img;
    if( 1 != img.channels() )
    {
        cv::cvtColor( img, for_process_img, CV_BGR2GRAY);
    }
    else
    {
        for_process_img = img;
    }

	/* copy from initial shape */
	shape_type current_shape(m_initial_shape.val);

	vector<float> feature_pixel_values;

	for( unsigned long iter=0;iter<m_forests.size();iter++)
	{
		extract_feature_pixel_values( for_process_img, rect, current_shape, m_initial_shape, m_anchor_idx[iter], m_deltas[iter], feature_pixel_values);
		// evaluate all the trees at this level of the cascade.
		for (unsigned long i = 0; i < m_forests[iter].size(); ++i)
		{
			current_shape += m_forests[iter][i](feature_pixel_values);
		}
	}

	// convert the current_shape into a full_object_detection
	Matx<float,2,2> tran_m;
	Matx<float,2,1> tran_b;
	unnormalizing_tform(rect, tran_m, tran_b);
	shape_type final_shape = shape_type::zeros();
	for( unsigned long i=0;i<num_parts();i++)
	{
		Vec2f pt = location( current_shape, i);
		pt = transfrom_vector( pt, tran_m, tran_b);
		final_shape(2*i,0) = cvRound(pt[0]);
		final_shape(2*i+1,0) = cvRound(pt[1]);
	}
	return final_shape;
}

shape_predictor_trainer::shape_predictor_trainer():m_rng(0)
{
	/* default value for trainer*/
	m_cascade_depth = 10;
	m_tree_depth = 4;
	m_num_trees_per_cascade_level = 500;
	m_nu = 0.1;
	m_oversampling_amount = 20;
	m_feature_pool_size = 400;
	m_lambda = 0.1;
	m_num_test_splits = 20;
	m_feature_pool_region_padding = 0;
	m_verbose = false;
}

void shape_predictor_trainer::randomly_sample_pixel_coordinates (
	vector<Vec2f>& pixel_coordinates,
	float min_x,
	float max_x,
	float min_y,
	float max_y
) const
{
	pixel_coordinates.resize(get_feature_pool_size());
	for( unsigned long i=0;i<get_feature_pool_size();i++)
	{
		pixel_coordinates[i] = Vec2f();
		pixel_coordinates[i][0] = m_rng.uniform(min_x, max_x);
		pixel_coordinates[i][1] = m_rng.uniform(min_y, max_y);
	}
}

vector<vector<Vec2f > > shape_predictor_trainer::randomly_sample_pixel_coordinates (
	const shape_type& initial_shape
	) const
{
	const double padding = get_feature_pool_region_padding();
	// Figure figure out the bounds on the object shapes.  We will sample uniformly from this box.
	double min_x, max_x, min_y, max_y;
	Mat reshape_cor =Mat(initial_shape);
	reshape_cor = reshape_cor.reshape(0, reshape_cor.rows/2);

	minMaxLoc( reshape_cor.col(0), &min_x, &max_x);
	minMaxLoc( reshape_cor.col(1), &min_y, &max_y);
	min_x -= padding;
	min_y -= padding;
	max_x += padding;
	max_y += padding;

	vector<vector<Vec2f> > pixel_coordinates;
	pixel_coordinates.resize(get_cascade_depth());
	for(unsigned long i=0;i<get_cascade_depth();i++)
	{
		randomly_sample_pixel_coordinates( pixel_coordinates[i], float(min_x), float(max_x), float(min_y), float(max_y) );
	}
	return pixel_coordinates;
}

shape_type shape_predictor_trainer::populate_training_sample_shapes(
	const vector<vector<Rect> > &object_rects,
	const vector<vector<shape_type> > &object_shapes,
	vector<training_sample>& samples
	) const
{
	assert( object_shapes.size() == object_shapes.size());

	samples.clear();
	shape_type total_shape = shape_type::zeros();
	long count = 0;
	// fisrt fill out the target shapes
	for( unsigned long i=0;i<object_rects.size();i++)
	{
		assert( object_rects[i].size() == object_shapes[i].size());
		for (unsigned long j=0;j<object_rects[i].size();j++)
		{
			training_sample sample;
			sample.image_idx = i;
			sample.rect = object_rects[i][j];
			sample.target_shape = object_to_shape( object_rects[i][j], object_shapes[i][j]);
			for( unsigned long iter=0;iter<get_oversampling_amount();iter++)
				samples.push_back(sample);
			total_shape += sample.target_shape;
			count++;
		}
	}

	shape_type mean_shape( total_shape, 1.0/count, Matx_ScaleOp());

	// now go pick random initial shapes
	for (unsigned long i = 0; i < samples.size(); ++i)
	{
		if ((i%get_oversampling_amount()) == 0)
		{
			// The mean shape is what we really use as an initial shape so always
			// include it in the training set as an example starting shape.
			samples[i].current_shape = mean_shape;
		}
		else
		{
			// Pick a random convex combination of two of the target shapes and use
			// that as the initial shape for this sample.

			const unsigned long rand_idx = m_rng.operator unsigned int()%samples.size();
			const unsigned long rand_idx2 = m_rng.operator unsigned int()%samples.size();
			const double alpha = m_rng.operator double();
			samples[i].current_shape = alpha*samples[rand_idx].target_shape + (1-alpha)*samples[rand_idx2].target_shape;
		}
	}
	return mean_shape;
}

unsigned long shape_predictor_trainer::partition_samples (
	const impl::split_feature& split,
	std::vector<training_sample>& samples,
	unsigned long begin,
	unsigned long end
	) const
{
	// splits samples based on split (sorta like in quick sort) and returns the mid
	// point.  make sure you return the mid in a way compatible with how we walk
	// through the tree.

	unsigned long i = begin;
	for (unsigned long j = begin; j < end; ++j)
	{
		if (samples[j].feature_pixel_values[split.idx1] - samples[j].feature_pixel_values[split.idx2] > split.threshold)
		{
			samples[i].swap(samples[j]);
			++i;
		}
	}
	return i;
}

impl::split_feature shape_predictor_trainer::randomly_generate_split_feature (
	const vector<Vec2f>& pixel_coordinates	
	) const
{
	const double lambda = get_lambda();
	impl::split_feature feat;
	double accept_prob;

	do 
	{
		feat.idx1 = m_rng.operator unsigned int() % get_feature_pool_size();
		feat.idx2 = m_rng.operator unsigned int() % get_feature_pool_size();
		const double dist = cv::norm( pixel_coordinates[feat.idx1]- pixel_coordinates[feat.idx2]);
		accept_prob = std::exp(-1*dist/lambda);
	} while(feat.idx1 == feat.idx2 || !(accept_prob > m_rng.operator double()));
	feat.threshold = (m_rng.operator float()*256 -128)/2.0f;
	
	return feat;
}

impl::split_feature shape_predictor_trainer::generate_split (
	const vector<training_sample>& samples,
	unsigned long begin,
	unsigned long end,
	const vector<Vec2f>& pixel_coordinates,
	const shape_type &sum,
	shape_type& left_sum,
	shape_type& right_sum 
	) const
{
	assert( samples.size() !=0 && "Training samples should not be empty");
	// generate a bunch of random splits and test them and return the best one.
	const unsigned long num_test_splits = get_num_test_splits();  

	// sample the random features we test in this function
	std::vector<impl::split_feature> feats;
	feats.reserve(num_test_splits);
	for (unsigned long i = 0; i < num_test_splits; ++i)
	{
		feats.push_back(randomly_generate_split_feature(pixel_coordinates));
	}

	std::vector<shape_type> left_sums(num_test_splits);
	for (unsigned long i=0; i<left_sums.size();i++)
		left_sums[i] =shape_type::zeros();

	std::vector<unsigned long> left_cnt(num_test_splits,0);

	// now compute the sums of vectors that go left for each feature
	shape_type temp;
	for( unsigned long j=begin; j < end; j++)
	{
		temp = samples[j].target_shape - samples[j].current_shape;
		for (unsigned long i=0;i<num_test_splits;i++)
		{
			if(samples[j].feature_pixel_values[feats[i].idx1] - samples[j].feature_pixel_values[feats[i].idx2] > feats[i].threshold)
			{
				left_sums[i] += temp;
				++left_cnt[i];
			}
		}
	}
	
	// now figure out which feature is the best
	double best_score = -1;
	unsigned long best_feat = 0;
	for (unsigned long i = 0; i < num_test_splits; ++i)
	{
		// check how well the feature splits the space.
		double score = 0;
		unsigned long right_cnt = end-begin-left_cnt[i];
		if (left_cnt[i] != 0 && right_cnt != 0)
		{
			temp = sum - left_sums[i];
			score = (left_sums[i].dot(left_sums[i]))/left_cnt[i] + (temp.dot(temp))/right_cnt;
			if (score > best_score)
			{
				best_score = score;
				best_feat = i;
			}
		}
	}


	/* store the value and feature*/
	impl::swap_shape( left_sums[best_feat], left_sum);
	if (left_cnt[best_feat] !=0 )
	{
		right_sum = sum - left_sum;
	}
	else
	{
		right_sum = sum;
		left_sum = shape_type::zeros();
	}

	return feats[best_feat];
}

impl::regression_tree shape_predictor_trainer::make_regression_tree (
	vector<training_sample>& samples,			/* in&out: training samples, may change the current shape*/
	const vector<Vec2f>& pixel_coordinates		/* in    : pixel coordinates, for feature selection */
	) const
{
	
	assert( samples.size()!=0 && "Size of samples should be zero");
	using namespace impl;
	deque<pair<unsigned long, unsigned long> > parts;
	parts.push_back( make_pair(0,samples.size()));

	impl::regression_tree tree;

	// walk the tree in breadth first order
	const unsigned long num_split_nodes = static_cast<unsigned long>(std::pow(2.0, (double)get_tree_depth())-1);
	std::vector<shape_type> sums(num_split_nodes*2+1);
	for (unsigned long i=0;i<sums.size();i++)
		sums[i] = shape_type::zeros();

	for (unsigned long i = 0; i < samples.size(); ++i)
		 sums[0] += samples[i].target_shape - samples[i].current_shape;

	for (unsigned long i = 0; i < num_split_nodes; ++i) 
	{
		std::pair<unsigned long,unsigned long> range = parts.front();
		parts.pop_front();
		
		const impl::split_feature split = generate_split( samples, range.first, range.second,
			pixel_coordinates, sums[i], sums[left_child(i)], sums[right_child(i)]);

		tree.splits.push_back(split);

		const unsigned long mid = partition_samples( split, samples, range.first, range.second);
		parts.push_back( std::make_pair(range.first, mid));
		parts.push_back( std::make_pair(mid, range.second));

	}
	
	// Now all the parts contain the ranges for the leaves so we can use them to
	// compute the average leaf values.
	tree.leaf_values.resize( parts.size());
	for (unsigned long i=0;i<parts.size(); i++)
	{
		if (parts[i].second != parts[i].first)
			tree.leaf_values[i] = shape_type( sums[num_split_nodes+i],get_nu()/(parts[i].second - parts[i].first), Matx_ScaleOp());
		else
			tree.leaf_values[i] =shape_type::zeros();

		// now adjust the current shape based on these predictions
		for (unsigned long j = parts[i].first; j < parts[i].second; ++j)
			samples[j].current_shape += tree.leaf_values[i];
	}
	

	return tree;
}

shape_predictor shape_predictor_trainer::train( 
	const vector<Mat> &images,				/* in : input images*/
	const vector<vector<Rect> > &rects,		/* in : rect for the target */
	const vector<vector<shape_type> > &shapes		/* in : shape for the corresponding rect*/
	)
{
	using namespace impl;
	assert( images.size() == rects.size() && images.size() == shapes.size());

	// make sure the objects agree on the number of parts and that there is at
	// least one full_object_detection. 
	unsigned long num_parts=0;
	for (unsigned long i=0;i<rects.size();i++)
	{
		assert( rects[i].size() == shapes[i].size());
		for (unsigned long j=0;j<rects[i].size();j++)
		{
			if (num_parts == 0)
				num_parts = shapes[i][j].rows/2;
			else
			{
				assert( num_parts == shapes[i][j].rows/2 && shapes[i][j].cols == 1);
			}
		}
		assert( num_parts !=0 );
	}

	std::vector<training_sample> samples;
	const shape_type initial_shape = populate_training_sample_shapes( rects, shapes, samples);
 	const vector<vector<Vec2f> > pixel_coordinates = randomly_sample_pixel_coordinates(initial_shape);

	std::cout << "Fitting trees..." << std::endl;
	// Now start doing the actual training by filling in the forests
	std::vector<std::vector<impl::regression_tree> > forests(get_cascade_depth());
	for (unsigned long cascade = 0; cascade < get_cascade_depth(); ++cascade)
	{
		cout<<"cascade No : "<<cascade<<endl;
		// Each cascade uses a different set of pixels for its features.  We compute
		// their representations relative to the initial shape first.
		std::vector<unsigned long> anchor_idx; 
		std::vector<Vec2f> deltas;
		create_shape_relative_encoding(initial_shape, pixel_coordinates[cascade], anchor_idx, deltas);

		// First compute the feature_pixel_values for each training sample at this level of the cascade.
		for (unsigned long i = 0; i < samples.size(); ++i)
		{
			extract_feature_pixel_values(images[samples[i].image_idx], samples[i].rect,
				samples[i].current_shape, initial_shape, anchor_idx,
				deltas, samples[i].feature_pixel_values);
		}

		// Now start building the trees at this cascade level.
		for (unsigned long i = 0; i < get_num_trees_per_cascade_level(); ++i)
		{
			forests[cascade].push_back(make_regression_tree(samples, pixel_coordinates[cascade]));
			if ( i % 100 == 0)
			{
				cout<<"Fitting Tree No "<<i<<endl;
			}
		}
	}
	cout<<"Training Complete .."<<endl;
	return shape_predictor(initial_shape, forests, pixel_coordinates);

}

/*! test the predictor, return the mean error !*/
double shape_predictor_trainer::test_shape_predictor(
	const shape_predictor &sp,
	const vector<Mat> & images,
	const vector<vector<Rect> > rects,
	const vector<vector<shape_type> > shapes
	) const
{
	assert( rects.size() == shapes.size() && "rects and shapes should have same size");

	double sum_of_error = 0;
	long number_of_test = 0;

	for (unsigned long i=0;i<rects.size();i++)
	{
		for (unsigned long j=0;j<rects[i].size();j++)
		{
			shape_type predicted_shape = sp( images[i], rects[i][j]);
			sum_of_error += cv::norm( predicted_shape, shapes[i][j]);
			number_of_test++;
		}
	}
	return sum_of_error/number_of_test;
}

shape_predictor_trainer::~shape_predictor_trainer()
{

}
