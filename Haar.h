typedef struct region
{
	int x, y;
	int w, h;
} region;

class Haar : public Classifier
{
private:
	// list of all features. Each feature consist of pair of < list of black region and list of white regions >
	list< pair< list<region>, list<region> > > features;
	string class_map_file_name, feature_file_name;
	string svm_train_file_name, svm_test_file_name, svm_model_file_name, svm_prediction_file_name;
	int size;  // subsampled image resolution
	int n_features; // number of Haar like features
	map<int, string> class_num_map; // assign a class name to each number

	void compute_features()
	{
		for (int i = 0; i < n_features; ++i)
		{			
			int width = 5 + rand() % (size - 6);
			int height = 5 + rand() % (size - 6);
			int start_x = 1 + rand() % (size - width - 1);
			int start_y = 1 + rand() % (size - height - 1);

			int f = rand() % 10; // There are 10 types of feature

			list<region> black, white;
			
			if (f == 0 || f == 1)
			{		
				width = (width / 2) * 2;
				region region_white, region_black;
				region_white.x = (f == 0? start_x : start_x + width / 2);
				region_black.x = (f == 0? start_x + width / 2 : start_x);
				region_white.y = region_black.y = start_y;
				region_white.w = region_black.w = width / 2;
				region_white.h = region_black.h = height;	
				black.push_back(region_black);
				white.push_back(region_white);
			}
			else if (f == 2 || f == 3)
			{					
				height = (height / 2) * 2;
				region region_white, region_black;
				region_white.x = region_black.x = start_x;
				region_white.y = (f == 2? start_y : start_y + height / 2);
				region_black.y = (f == 2? start_y + height / 2 : start_y);
				region_white.w = region_black.w = width;
				region_white.h = region_black.h = height / 2;
				black.push_back(region_black);
				white.push_back(region_white);
			}
			else if (f == 4 || f == 5)
			{		
				width = (width / 3) * 3;		
				region region_white_1, region_black_1, region_white_2, region_black_2;
				region_white_1.x = (f == 4? start_x + width / 3 : start_x);
				region_black_1.x = (f == 4? start_x : start_x + width / 3);
				region_white_2.x = (f == 4? start_x + width : start_x + (2 * width) / 3);
				region_black_2.x = (f == 4? start_x + (2 * width) / 3 : start_x + width);
				region_white_1.y = region_black_1.y = region_white_2.y = region_black_2.y = start_y;
				region_white_1.w = region_black_1.w = region_white_2.w = region_black_2.w = width / 3;
				region_white_1.h = region_black_1.h = region_white_2.h = region_black_2.h = height;
				black.push_back(region_black_1);				
				white.push_back(region_white_1);
				if (f == 4) black.push_back(region_black_2);
				if (f == 5) white.push_back(region_white_2);
			}
			else if (f == 6 || f == 7)
			{				
				height = (height / 3) * 3;
				region region_white_1, region_black_1, region_white_2, region_black_2;				
				region_white_1.x = region_black_1.x = region_white_2.x = region_black_2.x = start_x;
				region_white_1.y = (f == 6? start_y + height / 3 : start_y);
				region_black_1.y = (f == 6? start_y : start_y + height / 3);
				region_white_2.y = (f == 6? start_y + height : start_y + (2 * height) / 3);
				region_black_2.y = (f == 6? start_y + (2 * height) / 3 : start_y + height);
				region_white_1.w = region_black_1.w = region_white_2.w = region_black_2.w = width;
				region_white_1.h = region_black_1.h = region_white_2.h = region_black_2.h = height / 3;
				black.push_back(region_black_1);				
				white.push_back(region_white_1);
				if (f == 6) black.push_back(region_black_2);
				if (f == 7) white.push_back(region_white_2);
			}
			else if (f == 8 || f == 9)
			{		
				width = (width / 2) * 2;
				height = (height / 2) * 2;
				region region_white_1, region_black_1, region_white_2, region_black_2;
				region_white_1.x = region_black_2.x = (f == 8? start_x + width / 2 : start_x);
				region_black_1.x = region_white_2.x = (f == 8? start_x : start_x + width / 2);
				region_black_1.y = region_white_1.y = (f == 8? start_y : start_y + height / 2);
				region_white_2.y = region_black_2.y = (f == 8? start_y + height / 2 : start_y);				
				region_white_1.w = region_black_1.w = region_white_2.w = region_black_2.w = width / 2;
				region_white_1.h = region_black_1.h = region_white_2.h = region_black_2.h = height / 2;
				black.push_back(region_black_1);				
				white.push_back(region_white_1);
				black.push_back(region_black_2);
				white.push_back(region_white_2);
			}
			else
			{
				cout << "Unknown feature type: " << f << endl;
			}		

			features.push_back(make_pair(black, white));
		}	
	}

	void normalize_features(vector<int> &feature_values)
	{
		double min, max;
		/*
		min = max = feature_values[0];
		for (int i = 0; i < feature_values.size(); ++i)
		{
			if (feature_values[i] < min)
				min = feature_values[i];
			else if (feature_values[i] > max)
				max = feature_values[i];
		}*/
		min = -1 * size * size * 255;
		max = -min;
		for (int i = 0; i < feature_values.size(); ++i)
			feature_values[i] = 200.0*(((feature_values[i] - min)*1.0) / (max - min))-100.0;
	}

	void write_features_to_file()
	{
		ofstream ofs;
		ofs.open(feature_file_name.c_str());
		ofs << features.size();
		for(list< pair< list<region>, list<region> > >::iterator it1 = features.begin(); it1 != features.end(); ++it1)
		{
			ofs << " " << (*it1).first.size();
			for (list<region>::iterator it2 = (*it1).first.begin(); it2 != (*it1).first.end(); ++it2)			
				ofs << " " << it2->x << " " << it2->y << " " << it2->w << " " << it2->h;
			ofs << " " << (*it1).second.size();
			for (list<region>::iterator it2 = (*it1).second.begin(); it2 != (*it1).second.end(); ++it2)			
				ofs << " " << it2->x << " " << it2->y << " " << it2->w << " " << it2->h;
		}
		ofs << endl;
		ofs.close();
	}

	void read_features_from_file()
	{
		ifstream ifs;
		ifs.open(feature_file_name.c_str());
		ifs >> n_features;		

		for (int i = 0; i < n_features; ++i)
		{
			list<region> black, white;
			int list_size;

			ifs >> list_size;
			for (int j = 0; j < list_size; ++j)
			{
				region region_black;
				ifs >> region_black.x >>  region_black.y >>  region_black.w >>  region_black.h;
				black.push_back(region_black);
			}

			ifs >> list_size;
			for (int j = 0; j < list_size; ++j)
			{
				region region_white;
				ifs >> region_white.x >>  region_white.y >>  region_white.w >>  region_white.h;
				white.push_back(region_white);
			}

			features.push_back(make_pair(black, white));
		}
		ifs.close();
	}

	void integral_image(CImg<int> &img)
	{
		for (int y = 0; y < img.height(); ++y)
		{
			for (int x = 0; x < img.width(); ++x)
			{
				if (x == 0)
				{
					if (y == 0)
						img(x, y, 0, 0) = img(x, y, 0, 0);
					else
						img(x, y, 0, 0) = img(x, y-1, 0, 0) + img(x, y, 0, 0);
				}
				else if (y == 0)
					img(x, y, 0, 0) = img(x-1, y, 0, 0) + img(x, y, 0, 0);
				else
					img(x, y, 0, 0) = img(x-1, y, 0, 0) + img(x, y-1, 0, 0) - img(x-1, y-1, 0, 0) + img(x, y, 0, 0);
			}
		}
	}

	void print(CImg<int> &img)
	{
		for (int y = 0; y < img.height(); ++y)
		{
			for (int x = 0; x < img.width(); ++x)
			{
				printf("%ld ", int(img(x, y, 0, 0)));
			}
			cout << endl;
		}
		cout << endl;
	}

	int get_feature_value_on_image(CImg<int> &img, pair< list<region>, list<region> > &feature)
	{
		int value = 0;

		// Loop through the black regions, add them		
		for (list<region>::iterator it = feature.first.begin(); it != feature.first.end(); ++it)
		{			
			value += img(it->x+it->w-1, it->y+it->h-1, 0, 0) + img(it->x-1, it->y-1, 0, 0)
					- img(it->x-1, it->y+it->h-1, 0, 0) - img(it->x+it->w-1, it->y-1, 0, 0);
		}

		// Loop through the white regions, subtract them		
		for (list<region>::iterator it = feature.second.begin(); it != feature.second.end(); ++it)
		{			
			value -= img(it->x+it->w-1, it->y+it->h-1, 0, 0) + img(it->x-1, it->y-1, 0, 0)
					- img(it->x-1, it->y+it->h-1, 0, 0) - img(it->x+it->w-1, it->y-1, 0, 0);
		}

		return value;
	}

public:

	Haar(const vector<string> &_class_list) : Classifier(_class_list)
	{
		class_map_file_name = "haar_class_map.dat";
		svm_train_file_name = "haar_svm_train.dat";
		svm_test_file_name = "haar_svm_test.dat";
		svm_model_file_name = "haar_svm_model.dat";
		svm_prediction_file_name = "haar_svm_predict.dat";
		feature_file_name = "haar_features.dat";
		size = 60;
		n_features = 1000; 
	}
	
	virtual void train(const Dataset &filenames) 
	{		
		cout << "Creating " << n_features << " Haar like features" << endl;
		compute_features();		

		write_features_to_file();		
		
		ofstream ofs1, ofs2;
		ofs1.open(class_map_file_name.c_str());
		ofs2.open(svm_train_file_name.c_str());

		int class_index = 1;

		// Loop through all the image classes
		for(Dataset::const_iterator c_iter = filenames.begin(); c_iter != filenames.end(); ++c_iter)
		{
			cout << "Processing feature vector for " << c_iter->first << endl;
			ofs1 << c_iter->first << " " << class_index << endl;
					
			// Loop through all the images of the selected class
			for(int i = 0; i < c_iter->second.size(); i++)
			{
				CImg<int> img(c_iter->second[i].c_str());
				img.resize(size, size, 1, 3);
				CImg<int> gray_image = img.get_RGBtoYCbCr().get_channel(0);
				
				vector<int> feature_values;

				// integral of image						
				integral_image(gray_image);	
				
				// Loop through all the features
				for(list< pair< list<region>, list<region> > >::iterator it = features.begin(); it != features.end(); ++it)
					feature_values.push_back(get_feature_value_on_image(gray_image, *it));				

				normalize_features(feature_values);

				ofs2 << class_index;
				
				int feature_index = 1;
				// Loop through all the features
				for(int j = 0; j < feature_values.size(); ++j)
				{													
					ofs2 << " " << feature_index << ":" << feature_values[j];;
					feature_index++;
				}
				ofs2 << endl;
			}					
			class_index++;			
		}
		ofs1.close();
		ofs2.close();

		// Train SVM
		//string cmd = "./svm_multiclass_learn -c 0.000001 ";
		string cmd = "./svm_multiclass_learn -c 50 ";
		cmd += svm_train_file_name;
		cmd += " ";
		cmd += svm_model_file_name;

		system(cmd.c_str());
	}

	virtual string classify(const string &filename)
	{
		CImg<int> img(filename.c_str());
		img.resize(size, size, 1, 3);
		CImg<int> gray_image = img.get_RGBtoYCbCr().get_channel(0);

		// integral of image						
		integral_image(gray_image);			

		ofstream ofs;
		ofs.open(svm_test_file_name.c_str());
		ofs << "2";
		
		vector<int> feature_values;

		// Loop through all the features
		for(list< pair< list<region>, list<region> > >::iterator it = features.begin(); it != features.end(); ++it)
			feature_values.push_back(get_feature_value_on_image(gray_image, *it));

		normalize_features(feature_values);

		int feature_index = 1;
		// Loop through all the features
		for(int i = 0; i < feature_values.size(); ++i)
		{			
			ofs << " " << feature_index << ":" << feature_values[i];
			feature_index++;
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

		return class_num_map[num];		
	}

	virtual void load_model()
	{
		string class_name;
		int class_num;
		ifstream ifs_cnm(class_map_file_name.c_str());
		while (ifs_cnm.good())
		{
			ifs_cnm >> class_name >> class_num;
			class_num_map[class_num] = class_name;
		}
		
		// Read features from file and store on the features data structure
		read_features_from_file();
	}
};
