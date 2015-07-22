#include <iostream>
#include <stdlib.h>
#include <sstream>
#include <stdio.h>
#include <string>
#include <omp.h>

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/contrib/contrib.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include "boost/filesystem.hpp"
#include "boost/lambda/bind.hpp"

#include "../scanner/scanner.h"
#include "shape_predictor.hpp"


using namespace std;
using namespace cv;

namespace bf = boost::filesystem;


bool read_in_list( const string &list_file, /* in */
                   vector<string> &people1, /* out */
                   vector<string> &people2, /* out */
                   vector<int> &index1,     /* out */
                   vector<int> &index2)     /* out */
{
    people1.clear();
    people2.clear();
    index1.clear();
    index2.clear();

    FILE *fp = fopen( list_file.c_str(), "r");
    if( fp == NULL)
    {
        cout<<"Can not open file "<<list_file<<endl;
        return false;
    }

    int r;
    int idx1,idx2;
    char name1[200];
    char name2[200];
    int counter =1;

    while(1)
    {
        if((counter/300)%2==0)
        {
            /* positive sample */
            r = fscanf( fp, "%s %d %d\n", &name1, &idx1, &idx2);
            if( r == EOF)
                break;
            people1.push_back( name1 );
            people2.push_back( name1 );
            index1.push_back( idx1);
            index2.push_back( idx2);

        }
        else
        {
            /* negative samples */
            r = fscanf( fp, "%s %d %s %d\n", &name1, &idx1, &name2, &idx2);
            if( r == EOF)
                break;
            people1.push_back( name1 );
            people2.push_back( name2 );
            index1.push_back( idx1);
            index2.push_back( idx2);
        }
        counter++;
    }

    return true;
}


int main( int argc, char** argv)
{
    string lfw_root = "/media/yuanyang/disk1/data/face_database/lfw/lfw/lfw/";
    string lfw_file_list = "/media/yuanyang/disk1/data/face_database/lfw/pairs.txt";
    string save_pos_folder1 = "./pos/1/";
    string save_pos_folder2 = "./pos/2/";
    string save_neg_folder1 = "./neg/1/";
    string save_neg_folder2 = "./neg/2/";

    vector<string> people1,people2;
    vector<int> name1,name2;
    
    read_in_list( lfw_file_list, people1, people2, name1, name2);
    for(int c=0; c<people1.size(); c++)
    {
        cout<<people1[c]<<" "<<name1[c]<<" "<<people2[c]<<" "<<name2[c]<<endl;
    }

	/* Load face detector*/
	scanner fhog_sc;
	if(!fhog_sc.loadModel("super_pack_lfw.xml"))
	{
		cout<<"Can not load face detector .."<<endl;
		return 1;
	}

	/* Load shape predictor */
	shape_predictor sp;
	if(!sp.load_model("model.xml"))
	{
		cout<<"Can not load shape predictor"<<endl;
		return 2;
	}

    int Nthreads = omp_get_max_threads();
    #pragma omp parallel for num_threads(Nthreads) 
    for( unsigned long i=0;i<people1.size();i++)
    {
        cout<<"processing image pair "<<i<<endl;
        char idx1_str[10],idx2_str[10];
        sprintf( idx1_str, "%04d", name1[i]);
        sprintf( idx2_str, "%04d", name2[i]);

        string image_name1 = lfw_root+people1[i]+"/"+people1[i]+"_"+string(idx1_str)+".jpg";
        string image_name2 = lfw_root+people2[i]+"/"+people2[i]+"_"+string(idx2_str)+".jpg";

        Mat img1 = imread( image_name1, CV_LOAD_IMAGE_GRAYSCALE);
        Mat img2 = imread( image_name2, CV_LOAD_IMAGE_GRAYSCALE);

        vector<Rect> face1, face2;
        vector<double> conf1,conf2;

        fhog_sc.detectMultiScale( img1, face1, conf1, Size(40,40), Size(300,300), 1.2, 1, 0);
        fhog_sc.detectMultiScale( img2, face2, conf2, Size(40,40), Size(300,300), 1.2, 1, 0);
        
        if( face1.empty() || face2.empty() )
            continue;

        int biggest_idx = 0;
        for( unsigned long i=0;i<face1.size();i++)
        {
            if( face1[i].width > face1[biggest_idx].width)
                biggest_idx = i;
        }
        
        shape_type shape1 = sp( img1, face1[biggest_idx]);
        Mat rotate_face1;
        shape_predictor::align_face( shape1, img1, 128, rotate_face1);

        biggest_idx = 0;
        for( unsigned long i=0;i<face2.size();i++)
        {
            if( face2[i].width > face2[biggest_idx].width)
                biggest_idx = i;
        }
        
        shape_type shape2 = sp( img2, face2[biggest_idx]);
        Mat rotate_face2;
        shape_predictor::align_face( shape2, img2, 128, rotate_face2);

        char counter_index[50];
        sprintf( counter_index, "%d", i);

        if( (i/300)%2==0 )
        {
            imwrite( save_pos_folder1 +string(counter_index)+".jpg", rotate_face1);
            imwrite( save_pos_folder2 +string(counter_index)+".jpg", rotate_face2);
        }
        else
        {
            imwrite( save_neg_folder1 +string(counter_index)+".jpg", rotate_face1);
            imwrite( save_neg_folder2 +string(counter_index)+".jpg", rotate_face2);
        }
        
    }
   
	return 0;
}
