#include <iostream>
#include <stdlib.h>
#include <sstream>
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

string get_folder_name( const string &fullpath)
{
	size_t pos1 = fullpath.find_last_of("/");
	string sub_str = fullpath.substr(0,pos1);
	size_t pos2 = sub_str.find_last_of("/");

	return fullpath.substr(pos2+1,pos1-pos2-1);
}


void process_folder(  const string &folder_path, 
                      const string &where_to_save_images,
                      scanner &fhog_sc, 
                      shape_predictor &sp)
{
	/*processing sub folder*/
	bf::directory_iterator end_it2;
	for( bf::directory_iterator file_iter( folder_path ); file_iter!=end_it2; file_iter++)
	{
		string pathname = file_iter->path().string();
		string basename = bf::basename( *file_iter);
		string extname  = bf::extension( *file_iter);

        if( extname != ".jpg" && extname != ".png" && extname != ".bmp" && extname != ".tif" &&
            extname != ".JPG" && extname != ".PNG" && extname != ".BMP")
            continue;

        /*  make subfolder for savint the cropped face image */
	    string folder_name = get_folder_name( pathname );
        string save_subfold = where_to_save_images+folder_name;
        if( !bf::exists(save_subfold))
        {
            if(!bf::create_directory( save_subfold))
            {
                cout<<"Can not make folder "<<save_subfold<<endl;
                return;
            }
        }

        /* read and processing image */
        //cout<<"processing image "<<pathname<<endl;
        Mat input_img = imread( pathname);
        
        // resize the image if it is too large
        if( input_img.rows > 1000 )
            cv::resize( input_img, input_img, Size(0,0), 1000.0/input_img.rows, 1000.0/input_img.rows);


        if( input_img.empty() )
        {
            cout<<"Can not open image "<<pathname<<endl;
            continue;
        }

        vector<Rect> faces;
        vector<double> confs;
        fhog_sc.detectMultiScale( input_img, faces, confs, Size(40,40), Size(800,800), 1.1, 1, 0.15);

        if( faces.size() > 1)
            continue;

        /* save the first found face */
        for(int index = 0; index < faces.size(); index++)
        {
            /* crop */
            shape_type shape = sp( input_img, faces[index]);
            Mat rotate_face;
            shape_predictor::align_face( shape, input_img, 268, rotate_face);

            /*  draw */
            vector<shape_type> shapes;
            shapes.push_back( shape);

            vector<Rect> faces2;
            faces2.push_back( faces[index]);
            
            Mat draw = shape_predictor::draw_shape( input_img, faces2, shapes);
            //imshow("rotate", rotate_face);
            //imshow("draw", draw);
            //waitKey(0);
            
            if( rotate_face.empty())
            {
                cout<<"Error, rotated image empty" <<endl;
                return;
            }

            stringstream ss; ss<<index; string idx_s; ss>>idx_s;
            string save_path = save_subfold+ "/" + basename +"_"+idx_s+".jpg";
            imwrite( save_path , rotate_face );
        }
	}
}

int main( int argc, char** argv)
{
	//string original_image_folder = "/home/yuanyang/Data/disosi/";
	//string where_to_save_images =  "/home/yuanyang/Data/disosi_crop/";

	string original_image_folder = "/home/yuanyang/Data/cele_output/";
	string where_to_save_images =  "/home/yuanyang/Data/cele_align/";

	//string original_image_folder = "/media/yuanyang/disk1/data/face_database/pci_staff/resultss/9083/for_test/";
	//string where_to_save_images =  "/home/yuanyang/Data/pci_staff/";

	/* Load face detector*/
	scanner fhog_sc;
	if(!fhog_sc.loadModel("f_rl_rr_face.xml"))
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

	/* ------------------------ iterator the folder -----------------------------*/
	if ( !bf::is_directory(original_image_folder) || !bf::exists(original_image_folder))
	{
		cout<<original_image_folder<<" is not a folder "<<endl;
		return -1;
	}

	if ( !bf::exists( where_to_save_images))
	{
		if(!bf::create_directory( where_to_save_images))
		{
			cout<<"Can not create folder "<<where_to_save_images<<endl;
			return -2;
		}
	}

    vector<string> sub_folder_pathes;
	bf::directory_iterator end_it;
	for( bf::directory_iterator folder_iter( original_image_folder); folder_iter!=end_it; folder_iter++)
	{
		if( !bf::is_directory(*folder_iter))
			continue;
        sub_folder_pathes.push_back( folder_iter->path().string() );
	}
    
    int Nthreads = omp_get_max_threads();
    #pragma omp parallel for num_threads(Nthreads) 
    for( long i=0;i<sub_folder_pathes.size();i++)
    {
        cout<<"processing folder "<<i<<" "<<sub_folder_pathes[i]<<endl;
        process_folder( sub_folder_pathes[i], where_to_save_images, fhog_sc, sp );
    }

	return 0;
}
