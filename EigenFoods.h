#include "CImg.h"
#include <ctime>
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <stdlib.h>
#include <utility>
#include <string>
#include <vector>
#include <Sift.h>
#include <sys/types.h>
#include <dirent.h>
#include <list>
#include <map>
#include <numeric>

using namespace cimg_library;
using namespace std;

typedef map<string, CImgList<double> > GreyDataset;

class EigenFoods : public Classifier
{

private:

	string class_map_file_name, feature_file_name, eigen_vec_file_name, eigen_val_file_name;
	string svm_train_file_name, svm_test_file_name, svm_model_file_name, svm_prediction_file_name;
	int size;
	int k;
	CImg<double> eigenVec;
	map<int, string> class_num_map;

	//method to convert to grey image.

	CImg<double> getGreyScale(CImg<double> &img)
	{

		CImg<double> grey(img.width(), img.height(), 1,1,0);

		for(int i = 0; i<img.width(); i++)
		{
			for(int j = 0; j<img.height(); j++)
			{
				double red = img(i,j,0,0);
				double green = img(i,j,0,1);
				double blue = img(i,j,0,2);

				double avg = (red + green + blue)/3;

				grey(i,j,0,0) = avg; 

			}
		}	

		//grey.save("grey.png");
		return grey;
	}

	void convertToGrey(const Dataset &filenames, GreyDataset &gd)
	{
		

		for(Dataset::const_iterator c_iter = filenames.begin(); c_iter != filenames.end(); ++c_iter)
		{
			CImgList<double> grey_image_list;

			for(int i = 0; i < c_iter->second.size(); i++)
			{
				CImg<double> img(c_iter->second[i].c_str());
				img.resize(size, size, 1, 3);
				CImg<double> gray_image = getGreyScale(img);
				
				gray_image.unroll('x').transpose();

				grey_image_list.push_back(gray_image);
			
			}

			//gd.push_back(c_iter->first, grey_image_list);
			gd.insert( std::pair<string, CImgList<double> >(c_iter->first,grey_image_list) );
					
		}
				
	}

	//Create feature matrix

	CImg<double> createFeatureMatrix(GreyDataset &gd)
	{
		CImgList<double> appended_image_list;

		for(GreyDataset::const_iterator c_iter = gd.begin(); c_iter != gd.end(); ++c_iter)
		{
			CImgList<double> grey_image_list;

			grey_image_list = c_iter->second;

			CImg<double> temp = grey_image_list.get_append('x');

			appended_image_list.push_back(temp);
					
		}

		return appended_image_list.get_append('x');

	}

	//get mean feature vector.

	CImg<double> getMeanFeatureVector(CImg<double> matrix)
	{
		CImg<double> mean_vector(1,matrix.height(),1,1,0);

		

		for(int i = 0; i<matrix.width(); i++)
		{
			for(int j = 0; j<matrix.height(); j++)
			{
				mean_vector(0,j) =  mean_vector(0,j) + matrix(i,j);
			}
		}

		for(int j = 0; j<matrix.height(); j++)
		{
			mean_vector(0,j) =  mean_vector(0,j)/matrix.width();
		}

		return mean_vector;


	}


	//subtract from the mean vectors

	CImg<double> getMatrixA(const CImg<double> &matrix, const CImg<double> mean_vector)
	{
		CImg<double> A(matrix.width(), matrix.height(), 1,1,0);

		for(int i = 0; i<matrix.width(); i++)
		{
			for(int j = 0; j<matrix.height(); j++)
			{
				A(i,j) = matrix(i,j) - mean_vector(0,j);
			}
		}

		return A;

	}

	CImg<double> getTopKVectors(const CImg<double> &ev)
	{
		CImg<double> top_k_vecs(k,ev.height(),1,1,0);

		for(int i = 0; i<k; i++)
		{
			for(int j = 0; j<ev.height(); j++)
			{
				top_k_vecs(i,j) = ev(i,j);
			}
		}

		return top_k_vecs;
	}

	void write(const CImg<double> &ev, string feature_file_name)
	{
		
		ofstream ofs;
		ofs.open(feature_file_name.c_str());
		ofs << ev.height() << " " << ev.width() << endl;

		for(int i = 0; i<ev.height(); i++)
		{
			for(int j = 0; j<ev.width(); j++)
			{
				ofs << ev(j,i) << " ";
			}

			ofs << endl;
		}

		ofs.close();
	}

	void printMatrix(const CImg<double> matrix)
	{
		for(int i = 0; i<matrix.height(); i++)
		{
			for(int j = 0; j<matrix.width();j++)
			{
				cout << matrix(j,i) << " ";
			}

			cout << endl;
		}
	}

	void roll_image(const CImgList<double> &img_list)
	{

		string dir_name = "rolled_images/";
		string ext = ".png";
		int p = 0;

		cimglist_for(img_list,img)
		{
			std::stringstream ss;
			CImg<double> temp = img_list[img];
			//temp.normalize(0,255);

			CImg<double> rolled_image(size,size,1,1);

			int k = 0;
			

			for(int i = 0; i < temp.height();)
			{
				for(int j = 0; j < size; j++)
				{
					rolled_image(j,k,0,0) = temp(0,i,0,0);

					i++;
				}

				k++;
			}

			ss << p;
			rolled_image.normalize(0,255);

			//printMatrix(rolled_image);

			rolled_image.save((dir_name + ss.str() + ext).c_str());

			p++;

		}
	}


	void write_train_to_file(CImg<double> &train, const Dataset &filenames)
	{
		ofstream ofs,ofs1;
		ofs.open(svm_train_file_name.c_str());
		ofs1.open(class_map_file_name.c_str());
		int class_index = 1;
		int h = 0;
		//train.normalize(-100,100);

		for(Dataset::const_iterator c_iter = filenames.begin(); c_iter != filenames.end(); ++c_iter)
		{
			
			ofs1 << c_iter->first << " " << class_index << endl;

			cout << c_iter->first << "::" << class_index << endl;

			for(int i = 0; i < c_iter->second.size(); i++)
			{
				
				ofs << find(class_list.begin(), class_list.end(), c_iter->first) - class_list.begin() + 1 << " " ;

				for(int j = 0; j<train.width(); j++)
				{
					ofs << j + 1 << ":" << train(j,h,0,0) << " ";
				}
				h++;
				ofs << endl;
			}

					class_index++;
		}

		ofs.close();
		ofs1.close();
		string cmd = "./svm_multiclass_learn -c 0.01 ";
		cmd += svm_train_file_name;
		cmd += " ";
		cmd += svm_model_file_name;

		system(cmd.c_str());

	}

	CImg<double> get_features(const string &filename)
	{
		CImg<double> test_file(filename.c_str());

		CImg<double> grey = getGreyScale(test_file);

		grey.resize(size,size,1,1);

		return grey.unroll('x').transpose();
	}

	vector<string> files_in_dir(const string &directory, bool prepend_directory = false)
	{
		vector<string> file_list;
		DIR *dir = opendir(directory.c_str());
		if(!dir)
			throw std::string("Can't find directory " + directory);
	  
		struct dirent *dirent;
		while ((dirent = readdir(dir))) 
			if(dirent->d_name[0] != '.')
		file_list.push_back((prepend_directory?(directory+"/"):"")+dirent->d_name);

		closedir(dir);
		return file_list;
	}

public:

	EigenFoods(const vector<string> &_class_list):Classifier(_class_list)
	{
		class_map_file_name = "eigen_class_map.txt";
		svm_train_file_name = "eigen_svm_train.txt";
		svm_test_file_name = "eigen_svm_test.txt";
		svm_model_file_name = "eigen_svm_model";
		svm_prediction_file_name = "eigen_svm_predict.dat";
		eigen_vec_file_name = "eigen_vec.txt";
		eigen_val_file_name = "eigen_val.txt";
		size = 40;
		k = 1250;
	}

	virtual void train(const Dataset &filenames)
	{
		cout <<"Training for eigen foods" << endl;

		CImg<double> feature_matrix;
		CImg<double> mean_vector;
		CImg<double> A, A_t;
		CImg<double> cov;
		CImg<double> eigen_val;
		CImg<double> eigen_vec;
		CImg<double> top_k_vecs;
		CImg<double> train_data;
		

		GreyDataset gd;

		cout <<"Converting to grey" << endl;

		convertToGrey(filenames, gd);

		cout <<"getting feature matrix" << endl;
		feature_matrix = createFeatureMatrix(gd);

cout <<"computing mean vector" << endl;
		mean_vector = getMeanFeatureVector(feature_matrix);

		cout <<"Getting A matrix " << endl;
		A = getMatrixA(feature_matrix, mean_vector);

		cout <<"A transpose() " << endl;
		A_t = A.get_transpose();

		cout <<"Cov A" << endl;
		cov = A * A_t;

		cout <<"Eigen value:" << endl;
		cov.symmetric_eigen(eigen_val,eigen_vec);

		cout <<"top k" << endl;
		top_k_vecs = getTopKVectors(eigen_vec);

		cout <<"writing top k" << endl;
		write(top_k_vecs, eigen_vec_file_name);

		cout <<"writing eigen val and eigen vec" << endl;
		write(eigen_val, eigen_val_file_name);

		cout <<"split top k" << endl;
		CImgList<double> image_list = top_k_vecs.get_split('x',top_k_vecs.width());

		cout << image_list.size() << endl;

		roll_image(image_list);

		train_data = top_k_vecs.transpose() * feature_matrix;

		write_train_to_file(train_data.transpose(), filenames);




	}

	virtual string classify(const string &filename)
	{
		cout << "in classify" << endl;
	}

	virtual string classify(const string &filename, const string &label)
	{
		CImg<double> test_features = get_features(filename);

		cout << eigenVec.width() << ", " << eigenVec.height() << endl;

		cout << test_features.width() << ", " << test_features.height() << endl;

		CImg<double> t = eigenVec * test_features;



		t.transpose();

		//t.normalize(-100,100);
		
		ofstream ofs;

		ofs.open(svm_test_file_name.c_str());
		//ofs << "0";

		ofs << find(class_list.begin(), class_list.end(), label) - class_list.begin() + 1 << ' ' ;

		for(int j = 0; j<t.width(); j++)
		{
			ofs << j + 1 << ":" << t(j,0) << ' ';
		}
			
		ofs << endl;
		ofs.close();

		string cmd = "./svm_multiclass_classify ";
		cmd += svm_test_file_name;
		cmd += " ";
		cmd += svm_model_file_name;
		cmd += " ";
		cmd += svm_prediction_file_name;

		system(cmd.c_str());

		ifstream ifs(svm_prediction_file_name.c_str());
		int num;

		ifs >> num;

		ifs.close();

		return class_num_map[num];	
		
	}

	virtual void load_model()
	{
		ifstream ifs2(class_map_file_name.c_str());
		string class_name;
		int class_num;

		while(ifs2.good())
		{
			ifs2 >> class_name >> class_num;

			cout << class_name << ":::" << class_num << endl;

			class_num_map[class_num] = class_name;
		}

		ifs2.close();

		int h, w;
		ifstream ifs(eigen_vec_file_name.c_str());

		ifs >> h >> w;

		CImg<double> img(w,h,1,1,0);

		for(int i = 0; i<h; i++)
		{
			for(int j = 0; j<w; j++)
			{
				double val;
				ifs >> val;
				img(j,i) = val;
			}
		}

		eigenVec = img;
		eigenVec.transpose();

		ifs.close();
	}

};