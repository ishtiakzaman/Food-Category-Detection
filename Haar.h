class Haar : public Classifier
{

public:

	Haar(const vector<string> &_class_list) : Classifier(_class_list) {}
	
	virtual void train(const Dataset &filenames) 
	{
		cout << "Pre Processing for Haar like features: " << (*filenames.begin()).second[0] << endl;

		CImg<double>((*filenames.begin()).second[0].c_str()).resize(size, size, 1, 3).save_png("test.png");
		return;

		for(Dataset::const_iterator c_iter = filenames.begin(); c_iter != filenames.end(); ++c_iter)
		{
			cout << "Processing " << c_iter->first << endl;
			CImg<double> class_vectors(size*size*3, filenames.size(), 1);

			// convert each image to be a row of this "model" image
			for(int i = 0; i < c_iter->second.size(); i++)
				class_vectors = class_vectors.draw_image(0, i, 0, 0, extract_features(c_iter->second[i].c_str()));

			class_vectors.save_png(("nn_model." + c_iter->first + ".png").c_str());
		}
	}

	virtual string classify(const string &filename)
	{
		CImg<double> test_image = extract_features(filename);

		// figure nearest neighbor
		pair<string, double> best("", 10e100);
		double this_cost;
		for(int c=0; c<class_list.size(); c++)
			for(int row=0; row<models[ class_list[c] ].height(); row++)
				if((this_cost = (test_image - models[ class_list[c] ].get_row(row)).magnitude()) < best.second)
					best = make_pair(class_list[c], this_cost);

		return best.first;
	}

	virtual void load_model()
	{
		for(int c=0; c < class_list.size(); c++)
		models[class_list[c] ] = (CImg<double>(("nn_model." + class_list[c] + ".png").c_str()));
	}

protected:

	// extract features from an image, which in this case just involves resampling and 
	// rearranging into a vector of pixel data.
	CImg<double> extract_features(const string &filename)
	{
		return (CImg<double>(filename.c_str())).resize(size,size,1,3).unroll('x');
	}

	static const int size = 40;  // subsampled image resolution
	map<string, CImg<double> > models; // trained models
};
