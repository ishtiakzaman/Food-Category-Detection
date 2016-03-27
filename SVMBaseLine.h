#include <iostream>
#include <fstream>

class SVMBaseLine : public Classifier
{
	protected:
		static const int size=40;  // subsampled image resolution
		char tmp_data_file[20];
		char tmp_model_file[20];
		char tmp_output_file[20];
		char class_num_map_file[20];

		map<string, CImg<double> > models; // trained models
		map<int, string> class_num_map; // assign a num to each class

	public:
		SVMBaseLine(const vector<string> &_class_list) : Classifier(_class_list) 
		{
			strcpy(tmp_data_file, "svm.dat");
			strcpy(tmp_model_file, "svm.model");
			strcpy(tmp_output_file, "svm.out");
			strcpy(class_num_map_file, "classnum.txt");
		}

		// Nearest neighbor training. All this does is read in all the images, resize
		// them to a common size, convert to greyscale, and dump them as vectors to a file
		virtual void train(const Dataset &filenames) 
		{
			ofstream ofs(tmp_data_file);
			ofstream ofs_cnm(class_num_map_file);
			int num = 1;
			for(Dataset::const_iterator c_iter=filenames.begin(); c_iter != filenames.end(); ++c_iter)
			{
				cout << "Processing " << c_iter->first << "(" << num << ")" << endl;

				class_num_map[num] = c_iter->first;

				ofs_cnm << num << " " << c_iter->first << endl;

				CImg<double> class_vectors(size*size*3, filenames.size(), 1);

				// convert each image to be a row of this "model" image
				for(int i=0; i<c_iter->second.size(); i++)
				{
					class_vectors = class_vectors.draw_image(0, i, 0, 0, extract_features(c_iter->second[i].c_str()));
				}

				class_vectors.save_png(("nn_model." + c_iter->first + ".png").c_str());

				export_data(ofs, class_vectors, num);
				num ++;
			}
			ofs.flush();

			// Train SVM
			string cmd = "./svm_multiclass_learn -c 1.0 ";
			cmd += tmp_data_file;
			cmd += " ";
			cmd += tmp_model_file;

			system(cmd.c_str());
		}

		virtual string classify(const string &filename, const string &label)
		{
			CImg<double> test_image = extract_features(filename);

			// figure nearest neighbor
			pair<string, double> best("", 10e100);
			double this_cost;
			//for(int c=0; c<class_list.size(); c++)
			//	for(int row=0; row<models[ class_list[c] ].height(); row++)
			//		if((this_cost = (test_image - models[ class_list[c] ].get_row(row)).magnitude()) < best.second)
			//			best = make_pair(class_list[c], this_cost);

			ofstream ofs(tmp_data_file);
			export_data(ofs, test_image, 2);

			string cmd = "./svm_multiclass_classify ";
			cmd += tmp_data_file;
			cmd += " ";
			cmd += tmp_model_file;
			cmd += " ";
			cmd += tmp_output_file;

			system(cmd.c_str());

			ifstream ifs(tmp_output_file);
			int num;

			ifs >> num;

			return class_num_map[num];
		}

		virtual void load_model()
		{
			string class_name;
			int class_num;
			ifstream ifs_cnm(class_num_map_file);
			while (ifs_cnm.good())
			{
				ifs_cnm >> class_num >> class_name;
				class_num_map[class_num] = class_name;
			}
			return;
		}
	protected:
		// extract features from an image, which in this case just involves resampling and 
		// rearranging into a vector of pixel data.
		CImg<double> extract_features(const string &filename)
		{
			return (CImg<double>(filename.c_str())).resize(size,size,1,3).unroll('x');
		}

		void export_data(ostream &os, const CImg<double> &features, int classnum)
		{
			for (int i = 0; i < features.height(); i++)
			{
				os << classnum << " ";
				for (int j = 0; j < features.width(); j++)
				{
					os << j+1 << ":" << features(j, i) << " ";
				}
				os << endl;
			}
		}
};
