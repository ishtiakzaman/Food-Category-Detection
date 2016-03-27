#include <iostream>
#include <fstream>

#include <CImg.h>
#include <Sift.h>
//#include <Kmeans.h>

#include <opencv2/core/core.hpp>

using namespace cv;

class BagofWords : public Classifier
{
	protected:
		static const int size=40;  // subsampled image resolution
		const char *tmp_data_file;
		const char *tmp_model_file;
		const char *tmp_output_file;
		const char *class_num_map_file;
		const char *class_centers_file;
		const char *histogram_file;

		static const int num_centers = 20;
		CImg<double> centers;

		map<string, CImg<double> > models; // trained models
		map<int, string> class_num_map; // assign a num to each class

	public:
		BagofWords(const vector<string> &_class_list) :
			Classifier(_class_list), centers(128, num_centers)
		{
			tmp_data_file = "svm.dat";
			tmp_model_file = "bow_svm.model";
			tmp_output_file = "svm.out";
			class_num_map_file = "classnum.txt";
			class_centers_file = "kmeans_centers.png";
			histogram_file = "histogram.png";
		}


		// data entry
		struct FeatureDataEntry
		{
			int label;
			vector<SiftDescriptor> descriptors;
		};

		// Nearest neighbor training. All this does is read in all the images, resize
		// them to a common size, convert to greyscale, and dump them as vectors to a file
		virtual void train(const Dataset &filenames)
		{
			ofstream ofs(tmp_data_file);
			ofstream ofs_cnm(class_num_map_file);
			int num = 1;

			srand(time(NULL));

			vector<FeatureDataEntry> siftset;

			int num_features = 0;
			CImg<double> all_features(128, 0, 1, 1);

			system("./mk_cache_dir.sh");

			//Do Sift
			for(Dataset::const_iterator c_iter=filenames.begin(); c_iter != filenames.end(); ++c_iter)
			{
				cout << "Processing " << c_iter->first << "(" << num << ")" << endl;

				class_num_map[num] = c_iter->first;

				ofs_cnm << num << " " << c_iter->first << endl;

				CImg<double> class_vectors(size*size*3, filenames.size(), 1);

				for(int i=0; i<c_iter->second.size(); i++)
				{
					cout << i << endl;

					// Sift
					siftset.push_back(FeatureDataEntry());
					FeatureDataEntry &entry = siftset.back();
					entry.label = num;

					string sift_file = "siftcache/"+c_iter->second[i]+".sift";
					fstream fs_sift;
					if ( access( sift_file.c_str(), R_OK ) != -1 )
					{
						fs_sift.open(sift_file.c_str(), fstream::in);
						// Read sift from file
						while (fs_sift.good())
						{
							float x[128];
							for (int j = 0; j < 128; j++)
							{
								fs_sift >> x[j];
							}
							entry.descriptors.push_back(
									SiftDescriptor(0, 0, 0, 0, x));
						}
						fs_sift.close();
						continue;
					}

					//ofstream ofs_sift(sift_file);
					
					CImg<double> image(c_iter->second[i].c_str());
					CImg<double> gray = image.get_RGBtoHSI().get_channel(0);
					entry.descriptors = Sift::compute_sift(gray);

					fs_sift.open(sift_file.c_str(), fstream::out);
					for (int k = 0; k < entry.descriptors.size(); k++)
					{
						for (int j = 0; j < 128; j++)
						{
							fs_sift << entry.descriptors[k].descriptor[j] << "\t";
						}
						fs_sift << endl;
					}
					fs_sift.close();

					num_features += entry.descriptors.size();
				}

				num ++;
			}


			if ( access( "all_sift.cimg", R_OK ) != -1 )
			{
				all_features = CImg<double>("all_sift.cimg");
				cout << "loaded previous feature points." << endl;
				goto KMEANS;
			}

			{// For jump across initialization problem
				all_features.resize(128, num_features, 1, 1);
				int row_index = 0;
				for (int i = 0; i < siftset.size(); i++)
				{
					FeatureDataEntry &entry = siftset[i];
					// Collect features
					for(int j=0; j<entry.descriptors.size(); j++)
					{
						img_set_row(all_features, row_index,
								entry.descriptors[j].descriptor,
								entry.descriptors[j].descriptor.size());
						row_index++;
					}
				}
			}

			all_features.save("all_sift.cimg");

			KMEANS:
			if ( access( class_centers_file, R_OK ) != -1 )
			{
				centers = CImg<double>(class_centers_file);
				cout << "loaded previous centers." << endl;
				goto EXTRACT_FEATURE;
			}

			
			//kmeans(all_features, num_centers, centers);//my own kmeans algorithm

			{// For GOTO
				//to opencv mat
				Mat datamat(all_features.height(), all_features.width(), CV_32F);
				img2mat(datamat, all_features);
				Mat predicts;
				Mat mat_centers;
				TermCriteria crit;
				crit.epsilon = 0.01;
				//crit.
				//kmeans(datamat, num_centers, predicts, crit, 3, KMEANS_PP_CENTERS, mat_centers);
				kmeans(datamat, num_centers, predicts, crit, 3, KMEANS_RANDOM_CENTERS, mat_centers);
				centers = mat2img(mat_centers);
				centers.save(class_centers_file);
			}

			EXTRACT_FEATURE:
			CImg<double> class_vectors;
			if ( access( histogram_file, R_OK ) != -1 )
			{
				class_vectors = CImg<double>(histogram_file);
				cout << "loaded previous historgram." << endl;
				goto EXPORT_DATA;
			}
			class_vectors = extract_features(siftset, centers);
			class_vectors.print();
			class_vectors.save(histogram_file);

			EXPORT_DATA:
			for (int i = 0; i < class_vectors.height(); i++)
			{
				export_data(ofs, class_vectors.get_row(i), siftset[i].label);
			}

			ofs.flush();

			// Train SVM
			string cmd = "./svm_multiclass_learn -c 1 ";
			cmd += tmp_data_file;
			cmd += " ";
			cmd += tmp_model_file;

			system(cmd.c_str());
		}

		virtual string classify(const string &filename)
		{
			// load image
			CImg<double> image(filename.c_str());

			// SIFT
			CImg<double> gray = image.get_RGBtoHSI().get_channel(0);
			vector<SiftDescriptor> descriptors = Sift::compute_sift(gray);

			// extract features
			CImg<double> test_image = extract_features(descriptors, centers);

			// figure nearest neighbor
			ofstream ofs(tmp_data_file);
			export_data(ofs, test_image, 2);

			string cmd = "./svm_multiclass_classify ";
			cmd += tmp_data_file;
			cmd += " ";
			cmd += tmp_model_file;
			cmd += " ";
			cmd += tmp_output_file;
			cmd += " >/dev/null";

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

			centers = CImg<double>(class_centers_file);
			return;
		}

	protected:
		// extract features from an image, which in this case just involves resampling and
		// rearranging into a vector of pixel data.
		CImg<double> extract_features(
				const vector<SiftDescriptor> &descriptors,
				const CImg<double> &centers)
		{
			CImg<double> histo(num_centers, 1);
			histo = 0.0;

			vector<int> predicts(descriptors.size());

			CImg<double> features(128, descriptors.size());
			features = 0.0;

			// Collect features
			for(int i=0; i<descriptors.size(); i++)
			{
				img_set_row(features, i,
						descriptors[i].descriptor,
						descriptors[i].descriptor.size());
			}

			get_prediction(features, centers, predicts);

			for (int j = 0; j < predicts.size(); j++)
			{
				histo(predicts[j], 0)++;
			}
			for (int j = 0; j < histo.width(); j++)
			{
				histo(j, 0) = histo(j, 0) / predicts.size() * 255;
			}
			//for (int j = 0; j < histo.width(); j++)
			//{
			//	cout << j << ":" << histo[j] << endl;
			//}
			return histo;
		}
		CImg<double> extract_features(const vector<FeatureDataEntry> &dataset, const CImg<double> &centers)
		{
			CImg<double> histo(num_centers, dataset.size());
			histo = 0.0;

			for (int i = 0; i < dataset.size(); i++)
			{
				int total = 0;
				const vector<SiftDescriptor> &descriptors =
					dataset[i].descriptors;

				CImg<double> features(128, descriptors.size());
				features = 0.0;

				// Collect features
				for(int j=0; j<descriptors.size(); j++)
				{
					img_set_row(features, j,
							descriptors[j].descriptor,
							descriptors[j].descriptor.size());
				}

				vector<int> predicts;
				predicts.resize(features.height());
				get_prediction(features, centers, predicts);

				for (int j = 0; j < predicts.size(); j++)
				{
					histo(predicts[j], i)++;
				}
				for (int j = 0; j < histo.width(); j++)
				{
					histo(j, i) = histo(j, i) / predicts.size() * 255;
				}
				//for (int j = 0; j < histo.width(); j++)
				//{
				//	cout << j << ":" << histo(j,i) << endl;
				//}
			}
			return histo;
		}

		bool get_prediction(const CImg<double> &data, const CImg<double> &centers,
				vector<int> &predicts)
		{
			bool flag = false;
			double min = 1e9;
			for (int i = 0; i < data.height(); i++)
			{
				int index = -1;
				double min = 1e9;
				CImg<double> row = data.get_row(i);
				for (int j = 0; j < centers.height(); j++)
				{
					double dis = get_distance(row, centers.get_row(j));
					if (dis < min)
					{
						min = dis;
						index = j;
					}
				}
				if (predicts[i] != index)
				{
					predicts[i] = index;
					flag = true;
				}
			}
			return flag;
		}

		double get_distance(const CImg<double> &v1, const CImg<double> &v2)
		{
			CImg<double> diff = v1 - v2;
			return sqrt(diff.mul(diff).sum());
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

		template<typename ArrayT>
		void img_set_row(CImg<double> &img, int row, const ArrayT &data, int n)
		{
			for (int i = 0; i < n; i++)
			{
				img(i, row, 0, 0) = data[i];
			}
		}

		void image_assign_channel(CImg<double> &img, const CImg<double> &other, int channel)
		{
			for (int i = 0; i < img.height(); i++)
			{
				for (int j = 0; j < img.width(); j++)
				{
					img(j,i,0,channel) = other(j,i);
				}
			}
			return;
		}

		void img2mat(Mat &m, const CImg<double> &img)
		{
			CImg<float> fimg(img);
			for (int i = 0; i < img.height(); i++)
			{
				for (int j = 0; j < img.width(); j++)
				{
					m.at<float>(i,j) = img(j,i);
					//fimg(j,i) = img(j,i);
				}
				//cout << i << endl;
			}
			//memcpy(m.data, fimg.data(), img.height() * img.width() * sizeof(float));
			//Mat m(img.height(), img.width(), CV_32F);
			//return m;
		}

		CImg<double> mat2img(const Mat& m)
		{
			CImg<double> img(m.cols, m.rows);
			for (int i = 0; i < img.height(); i++)
			{
				for (int j = 0; j < img.width(); j++)
				{
					img(j,i) = m.at<float>(i,j);
				}
			}
			return img;
		}
};
