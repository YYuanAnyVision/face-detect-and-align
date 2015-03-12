#include <iostream>
#include <cmath>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/contrib/contrib.hpp"
#include "opencv_warpper_libsvm.h"
#include "svm.h"

using namespace std;
using namespace cv;

opencv_warpper_libsvm::opencv_warpper_libsvm()
{
    /*  set the default svm train parameters, see doc of libsvm  */
    m_svm_para.svm_type = C_SVC;
    m_svm_para.kernel_type = LINEAR;
    m_svm_para.degree = 3;
    m_svm_para.gamma = 0;	// should be set to 1/num_features
    m_svm_para.coef0 = 0;   // need tuning
    m_svm_para.nu = 0.5;
    m_svm_para.cache_size = 400;
    m_svm_para.C = 0.1;
    m_svm_para.eps = 1e-3;
    m_svm_para.p = 0.1;
    m_svm_para.shrinking = 1;
    m_svm_para.probability = 0;

    /* no change for samples during training */
    m_svm_para.nr_weight = 0;
    m_svm_para.weight_label = NULL;
    m_svm_para.weight = NULL;
    
    m_feature_dim = 0;
    m_weight_vector_for_linearsvm = NULL;
}

opencv_warpper_libsvm::~opencv_warpper_libsvm()
{
    if( m_model )
        svm_free_and_destroy_model( &m_model );
    if( m_weight_vector_for_linearsvm)
        delete [] m_weight_vector_for_linearsvm;
}

svm_parameter opencv_warpper_libsvm::getSvmParameters()
{
    return m_svm_para;
}

bool opencv_warpper_libsvm::train(  const Mat &positive_data,       //in : positive data, row samples, number_of_samples x feature_dim
                                    const Mat &negative_data,       //in : negative data, row samples, number_of_samples x feature_dim
                                    const string &path_to_save)     //in : where to save the model file
{
    cout<<"Start training process "<<endl;
    /* check the data */
    cout<<"Check the data "<<endl;
    if( positive_data.cols != negative_data.cols ||
        positive_data.empty() || negative_data.empty()||
        !positive_data.isContinuous() || !negative_data.isContinuous())
    {
        cout<<"Input data should be continunous in momory, CV_32F format, row sample "<<endl;
        return false;
    }

    /*  save the feature dim */
    m_feature_dim = positive_data.cols;

    /*-----------------------------------------------------------------------------
     *  prepare the training data in libsvm format
     *-----------------------------------------------------------------------------*/
    cout<<"Forming the training data in libsvm format"<<endl;
    svm_node **svm_train_positive_data;
    svm_node **svm_train_negative_data;

    fromMatToLibsvmNode( positive_data, svm_train_positive_data);
    fromMatToLibsvmNode( negative_data, svm_train_negative_data);

    /*  put negative and positive data together */
    int number_of_train_sample = positive_data.rows + negative_data.rows;
    svm_node ** svm_train_data = new svm_node*[ number_of_train_sample] ;
    double *train_label = new double[ number_of_train_sample];

    /*positive samples*/
    for ( unsigned int c=0;c<positive_data.rows ;c++ ) 
    {
        svm_train_data[c] = svm_train_positive_data[c];
		svm_train_positive_data[c] = NULL;
        train_label[c] = 1;
    }

    /* negative samples */
    for( unsigned int c=0;c<negative_data.rows;c++)
    {
        svm_train_data[c+positive_data.rows] = svm_train_negative_data[c];
		svm_train_negative_data[c] = NULL;
        train_label[c + positive_data.rows] = 0;
    }

    /* form the libsvm format training data: svm_problem */
    m_svm_training_data.l = number_of_train_sample;
    m_svm_training_data.x = svm_train_data;
    m_svm_training_data.y = train_label;
    
    /* check the svm_parameter */
    const char *error_message = svm_check_parameter( &m_svm_training_data, &m_svm_para);
    if( error_message )
    {
        cout<<"Parameters check failed, error message "<<error_message<<endl;
        return false;
    }

    /*  start  training ... */
    cout<<"Optimising ..."<<endl;
    m_model = svm_train( &m_svm_training_data, &m_svm_para );

    if( svm_save_model( path_to_save.c_str(), m_model)!= 0 )
    {
        cout<<"Can not save the model to file "<<path_to_save<<endl;
        return false;
    }
    cout<<"Saved .."<<endl;
    /*  convert to weight vector if feasiable */
    int type = svm_get_svm_type(m_model);
	cout<<"returned type is "<<type<<endl;
    if(LINEAR == svm_get_svm_type(m_model))
    {
        cout<<"Extract the linear weight vector for acceleration"<<endl;
        extract_weight_vector();
    }

    
    /*-----------------------------------------------------------------------------
     *  debug code
     *-----------------------------------------------------------------------------*/
        //svm_cross_validation( &m_svm_training_data, &m_svm_para, 10, target);
        //
        /*  output the weight vector */
        //cout<<"weight vector is "<<endl;
        //for( int c=0;c<m_feature_dim+1;c++)
        //    cout<<m_weight_vector_for_linearsvm[c]<<" ";
        //cout<<endl;

//        Mat pos_pre,neg_pre;
//        TickMeter tk;tk.start();
//        predict_linear( positive_data, pos_pre);
//        predict_linear( negative_data, neg_pre);
//        tk.stop();
//        cout<<"Time avg(linear ) is "<<tk.getTimeMilli()/number_of_train_sample<<endl;
        
//        tk.reset();tk.start();
//        int number_of_corr = 0;
//        for( int c=0;c<number_of_train_sample;c++)
//        {
//            double svm_predict_value;
//            svm_predict_values( m_model, svm_train_data[c], &svm_predict_value);
//            double linear_value  = 0;
//            if( c < positive_data.rows )
//                linear_value = pos_pre.at<float>(c,0);
//            else
//                linear_value = neg_pre.at<float>(c-positive_data.rows, 0);
//            cout<<"predict value is "<<svm_predict_value<<", gt is "<<train_label[c]<<" linear value is "<<linear_value<<endl;
//        }
//        tk.stop();
//        cout<<"Time avg(svm ) is "<<tk.getTimeMilli()/number_of_train_sample<<endl;
    /*-----------------------------------------------------------------------------
     *  debug code
     *-----------------------------------------------------------------------------*/

    /*  clean the memory */
    if( train_label )
        delete [] train_label;
	train_label = NULL;
    for( int c=0;c<number_of_train_sample;c++)
    {
        if( svm_train_data[c])
		{
            delete [] svm_train_data[c];
			svm_train_data[c] = NULL;
		}
    }
	delete [] svm_train_data;
	svm_train_data = NULL;
    cout<<"Finished ..."<<endl;
    return true;
}
bool opencv_warpper_libsvm::setSvmParameters( const svm_parameter &paras) // in : svm paras
{
    m_svm_para = paras;
}

bool opencv_warpper_libsvm::fromMatToLibsvmNode(   const Mat &inputData,             // in : number_of_samples x feature_dim
                                                    svm_node **&nodesData ) const    //out : 
{
    int number_of_samples = inputData.rows;
    int feature_dim       = inputData.cols;

    if( inputData.empty() || inputData.type()!=CV_32F)
    {
        cout<<"inputData invalid, only support CV_32F "<<endl;
        return false;
    }
    /*  allocate the memory for nodesData, remember to delete it after training */
    nodesData = new svm_node*[ number_of_samples ];

    for ( unsigned int sample_index=0; sample_index < number_of_samples ; sample_index++) 
    {
        nodesData[sample_index] = new svm_node[ feature_dim + 1];   //last node should have index = -1 indicates ending
        const float *input_sample_ptr = static_cast<const float*>( inputData.ptr<float>(sample_index) );

        int feature_count = 0;
        for( unsigned int fea_index = 0; fea_index < feature_dim; fea_index++)
        {
            if( input_sample_ptr[fea_index] == 0 )          // skip when the value is exact zero, libsvm use sparse representation, default is 0
                continue;
            nodesData[sample_index][feature_count].index = fea_index+1;             // !! svm_node index start with 1
            nodesData[sample_index][feature_count].value = input_sample_ptr[fea_index];
            feature_count++;
        }
        /* set -1 to the rest */
        while( feature_count < feature_dim+1 )
            nodesData[sample_index][feature_count++].index = -1;

    }
    return true;
}

bool opencv_warpper_libsvm::predict( const Mat &input_data,         // in : input feature, row format, CV_32F, number_of_sample x feature_dim
                                      Mat &predict_value) const     //out : CV_32F, number_of_samples x 1
{
	/* check the model */
	if( !m_model )
	{
		cout<<"Model file not ready"<<endl;
		return false;
	}
	if( input_data.type() != CV_32F || input_data.empty() || !input_data.isContinuous() || input_data.cols != m_feature_dim )
	{
		cout<<"Check the input data, should be CV_32F, row format , continuous is memory and feature dimension is "<<m_feature_dim<<endl;
		return false;
	}

	/*  use the linear weight directly */
    if(LINEAR == svm_get_svm_type(m_model) && m_weight_vector_for_linearsvm)
	{
		cout<<"using linear"<<endl;
		return predict_linear( input_data, predict_value);
	}
	else
	{
		cout<<"using svm"<<endl;
		/* use libsvm predict function */
        return predict_general( input_data, predict_value);
	}

    return true;   
}

bool opencv_warpper_libsvm::predict_general( const Mat &input_data,    // in : input feature, row format, CV_32F, number_of_sample x feature_dim
                                            Mat &predict_value) const // out: CV_32F, number_of_samples x 1  
{
	/* check the model */
	if( !m_model )
	{
		cout<<"Model file not ready"<<endl;
		return false;
	}

	if( input_data.type() != CV_32F || input_data.empty() || !input_data.isContinuous() || input_data.cols != m_feature_dim )
	{
		cout<<"Check the input data, should be CV_32F, row format , continuous is memory and feature dimension is "<<m_feature_dim<<endl;
		return false;
	}

	svm_node **test_data;
	fromMatToLibsvmNode( input_data, test_data );
	predict_value = Mat::zeros( input_data.rows, 1 , CV_32F);
	for( unsigned int c=0;c<input_data.rows; c++)
	{
        cout<<"predicting sample "<<c<<endl;
		double predict_score;
		svm_predict_values( m_model, test_data[c], &predict_score);
		predict_value.at<float>(c,0) = (float)predict_score;
	}
    
    cout<<"clear the memory"<<endl;
    /*  delete test_data */
    for( int c=0;c<input_data.rows;c++)
    {
        if( test_data[c])
		{
            delete [] test_data[c];
			test_data[c] = NULL;
		}
    }
    delete test_data;
    test_data = NULL;

    return true;
}



bool opencv_warpper_libsvm::extract_weight_vector()
{
    /* check the model */
    if( !m_model )
    {
        cout<<"Model file is empty "<<endl;
        return false;
    }
    if(LINEAR != svm_get_svm_type(m_model))
    {
        cout<<"Only works for linear svm model "<<endl;
        return false;
    }
        
    m_weight_vector_for_linearsvm = new float[ m_feature_dim+1 ]; // one more dimension for constant
    for( int c=0;c<m_feature_dim;c++)
        m_weight_vector_for_linearsvm[c] = 0.0;
    m_weight_vector_for_linearsvm[ m_feature_dim ] = -1*(*m_model->rho);

    /* used var */
    double alpha = 0;           //  w = y_0*a_0*x_0 +  y_1*a_1*x_1 + ... 
    svm_node *sample_node;
    
    for( unsigned int sample_index = 0; sample_index < m_model->l; sample_index++)
    {
        sample_node = m_model->SV[sample_index];
        //TODO implement  multi-class type ?
        alpha = m_model->sv_coef[0][sample_index];           // C class svm model has C-1 classifier, 0 means this is a binary classifier
        for( unsigned int fea_index = 0; fea_index < m_feature_dim; fea_index++)
        {
            if( sample_node[fea_index].index < 0 )
                break;
            //libsvm feature index starts from 1, check libsvm doc
            m_weight_vector_for_linearsvm[ sample_node[fea_index].index-1] += alpha*sample_node[fea_index].value;         
        }
    }

}


bool opencv_warpper_libsvm::predict_linear( const Mat &input_data,      // in : input feature, row format, CV_32F, number_of_sample x feature_dim
                                             Mat &predict_value) const  // out: out: CV_32F, number_of_samples x 1  
{
     /* check the model */
    if( !m_model )
    {
        cout<<"Model file is empty "<<endl;
        return false;
    }
    if(LINEAR != svm_get_svm_type(m_model))
    {
        cout<<"Only works for linear svm model "<<endl;
        return false;
    }
    if( input_data.type() != CV_32F || !input_data.isContinuous() || input_data.empty())
    {
        cout<<"Input data invalid "<<endl;
        return false;
    }

    /*  output value */
    predict_value = Mat::zeros( input_data.rows, 1, CV_32F);  

    for( unsigned int num_index=0;num_index<input_data.rows;num_index++)
    {
        const float *feature_vector = static_cast<const float*>(input_data.ptr<float>(num_index));
        predict_value.at<float>( num_index, 0) = get_feature_weight_dot_value( feature_vector );
    }
    return true;
}


float opencv_warpper_libsvm::get_feature_weight_dot_value( const float *feature ) const
{
    if(!m_weight_vector_for_linearsvm)
    {
        cout<<"m_weight_vector_for_linearsvm not ready "<<endl;
        return 0.0;
    }
    float sum_value = 0.0;
    for( unsigned int f_index = 0; f_index < m_feature_dim; f_index++)
    {
        sum_value += feature[f_index]*m_weight_vector_for_linearsvm[f_index];
    }
    sum_value += 1*m_weight_vector_for_linearsvm[ m_feature_dim ];        //plus the bias value
    return sum_value;
}


Mat opencv_warpper_libsvm::get_weight_vector() const
{
    if( !m_weight_vector_for_linearsvm )
        return Mat::zeros(0,0,CV_32F);
    Mat weight_mat = Mat::zeros( m_feature_dim + 1, 1, CV_32F ); // +1 for bias term
    
    for ( unsigned int c=0; c < m_feature_dim; c++) {
        weight_mat.at<float>(c,0) = m_weight_vector_for_linearsvm[c];
    }
    weight_mat.at<float>(m_feature_dim,0) = m_weight_vector_for_linearsvm[m_feature_dim];       // plus the bias term
    return weight_mat;
}
