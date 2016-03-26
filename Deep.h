#define OVERFEAT_PATH "/overfeat/bin/linux_64/overfeat"

class Deep : public Classifier
{
private:		
	string class_map_file_name, feature_file_name;
	string svm_train_file_name, svm_test_file_name, svm_model_file_name, svm_prediction_file_name;
	int size;  // subsampled image resolution	
	map<int, string> class_num_map; // assign a class name to each number

public:

	Deep(const vector<string> &_class_list) : Classifier(_class_list)
	{
		class_map_file_name = "deep_class_map.dat";
		svm_train_file_name = "deep_svm_train.dat";
		svm_test_file_name = "deep_svm_test.dat";
		svm_model_file_name = "deep_svm_model.dat";
		svm_prediction_file_name = "deep_svm_predict.dat";
		feature_file_name = "overfeat_features.dat";
		size = 256;		
	}
	
	virtual void train(const Dataset &filenames) 
	{	
		cout << "SVM model files are present in the repository, you can directly test without training." << endl;
		cout << "Do you still want to train (might take around 25-30 minutes)? (y/n): ";
		string response;
		cin >> response;
		if (response[0] != 'y' && response[0] != 'Y')
			return;		

		ofstream ofs1, ofs2;		
		ofs1.open(class_map_file_name.c_str());
		ofs2.open(svm_train_file_name.c_str());

		int class_index = 1;

		// Loop through all the image classes
		for(Dataset::const_iterator c_iter = filenames.begin(); c_iter != filenames.end(); ++c_iter)
		{
			string class_name = c_iter->first;
			int class_train_size = c_iter->second.size();
			cout << "Processing deep feature vector for " << class_name << endl;
			ofs1 << class_name << " " << class_index << endl;

			// Get features from overfeat			
			string cmd = ".";
			cmd += OVERFEAT_PATH;
			cmd += " -L 21";
					
			// Loop through all the images of the selected class
			for(int i = 0; i < class_train_size; i++)
			{
				string file_name = c_iter->second[i];
				CImg<int> img(file_name.c_str());
				img.resize(size, size, 1, 3);
				ostringstream oss;
				oss << "temp_resized_" << i << "." << file_name.substr(file_name.length()-3);
				file_name = oss.str();
				img.save(file_name.c_str());
								
				cmd += " " + file_name;
			}
			cmd += " > ";
			cmd += feature_file_name;

			system(cmd.c_str());							

			ifstream ifs(feature_file_name.c_str());
			if (ifs.is_open() == false)
			{
				cout << "Failed to read file: " << feature_file_name << endl;
				exit(0);
			}

			int nf, w, h;
			double value;
			for (int i = 0; i < class_train_size; ++i)
			{
				ofs2 << class_index;
				ifs >> nf >> h >> w;
				for (int j = 0; j < nf * w * h; ++j)
				{
					ifs >> value;
					ofs2 << " " << (j+1) << ":" << value;
				}
				ofs2 << endl;
			}
			ifs.close();					
			class_index++;			
		}
		ofs1.close();
		ofs2.close();		
		
		// Train SVM		
		string cmd = "./svm_multiclass_learn -c 1000 ";
		cmd += svm_train_file_name;
		cmd += " ";
		cmd += svm_model_file_name;

		system(cmd.c_str());		
	}

	virtual string classify(const string &filename)
	{
		CImg<int> img(filename.c_str());
		img.resize(size, size, 1, 3);
		string file_name = "temp_resized_test." + filename.substr(filename.length()-3);	
		img.save(file_name.c_str());

		// Get features from overfeat				
		string cmd = ".";
		cmd += OVERFEAT_PATH;
		cmd += " -L 21 ";
		cmd += file_name;
		cmd += " > ";
		cmd += feature_file_name;

		system(cmd.c_str());

		ifstream ifs_feat(feature_file_name.c_str());
		if (ifs_feat.is_open() == false)
		{
			cout << "Failed to read file: " << feature_file_name << endl;
			exit(0);
		}

		ofstream ofs;
		ofs.open(svm_test_file_name.c_str());
		ofs << "2";

		int feature_index = 1;		
		int nf, w, h;
		double value;
				
		ifs_feat >> nf >> h >> w;
		// Loop through all the features
		for (int i = 0; i < nf * w * h; ++i)
		{			
			ifs_feat >> value;
			ofs << " " << feature_index << ":" << value;
			feature_index++;
		}
		ifs_feat.close();

		ofs << endl;
		ofs.close();

		cmd = "./svm_multiclass_classify -v 0 ";
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
		
		ifstream ifs_model(svm_model_file_name.c_str());
		if (ifs_model.is_open() == false)
		{
			cout << "Failed to read file: " << svm_model_file_name << endl;
			exit(0);
		}
		ifs_model.close();

		ifstream ifs_cnm(class_map_file_name.c_str());
		if (ifs_cnm.is_open() == false)
		{
			cout << "Failed to read file: " << class_map_file_name << endl;
			exit(0);
		}
		while (ifs_cnm.good())
		{
			ifs_cnm >> class_name >> class_num;
			class_num_map[class_num] = class_name;
		}
	}
};
