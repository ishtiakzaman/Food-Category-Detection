/*
 * Kmeans.h
 *
 *  Created on: Mar 19, 2016
 *      Author: shawn
 */

#ifndef KMEANS_H_
#define KMEANS_H_


#include <CImg.h>

namespace std
{

//const char* (*translatefuncp)()

template< typename T >
ostream & operator << ( ostream & out, const vector< T > & v )
{
	typedef typename vector< T >::const_iterator iterator;
	out << "{";
	for ( iterator iter = v.begin(); iter != v.end(); iter++ )
	{
		if ((iter - v.begin()) % 10 == 0)
		{
			out << endl;
		}
		out<<*iter<<",";
	}
	out << endl << "}";
	return cout << endl;
}

template< typename T >
ostream & operator << ( ostream & out, const list< T > & v )
{
	//for(vector<T>::iterator iter = v.front() ; iter!= v.end() ; iter++)
	typedef typename list< T >::const_iterator iterator;
	out << "{";
	for ( iterator iter = v.begin(); iter != v.end(); iter++ )
	{
		if ((iter - v.begin()) % 10 == 0)
		{
			out << endl;
		}
		out<<*iter<<",";
	}
	out << endl << "}";
//	iterator iter = v.begin();
//	out<<*iter++;
//	while( iter != v.end() )
//	{
//		out<<" "<<*iter++;
//	}
	return out;
}

}


CImg<double> image_col_max(const CImg<double> &data)
{
	CImg<double> max(data.width(), 1, 1, 1);
	max = -1e9;
	for (int j = 0; j < data.width(); j++)
	{
		//int max = -9999999999;
		for (int i = 0; i < data.height(); i++)
		{
			if (data(j,i) > max(j))
			{
				max(j) = data(j,i);
			}
		}
	}
	return max;
}

CImg<double> image_col_min(const CImg<double> &data)
{
	CImg<double> min(data.width(), 1, 1, 1);
	min = 1e9;
	for (int j = 0; j < data.width(); j++)
	{
		for (int i = 0; i < data.height(); i++)
		{
			if (data(j,i) < min(j))
			{
				min(j) = data(j,i);
			}
		}
	}
	return min;
}

void image_row_add(CImg<double> &img, int i, const CImg<double> &row)
{
	for (int j = 0; j < img.width(); j++)
	{
		img(j, i) += row(j);
	}
}

void image_row_add(CImg<double> &img, int i, double val)
{
	for (int j = 0; j < img.width(); j++)
	{
		img(j, i) += val;
	}
}

bool get_prediction(const CImg<double> &data, const CImg<double> &centers,
		vector<int> &predicts);
void get_centers(const CImg<double> &data, CImg<double> &centers,
		const vector<int> &predicts);

double kmeans(const CImg<double> &data, int k, CImg<double> &centers)
{
	// get val range
	CImg<double> maxs = image_col_max(data);
	CImg<double> mins = image_col_min(data);

	double max = data.max();
	double min = data.min();

	// gen random k centers
	centers.rand(min, max);
//	// random select k points
//	CImg<int> rndind(k);
//	rndind.rand(0, data.height());
//	centers = 0.0;
//	for (int i = 0; i < k; i++)
//	{
//		image_row_add(centers, i, data.get_row(rndind(i)));
//	}

	// center index array
	vector<int> predicts(data.height());
	//predicts.resize(data.height());

	bool flag;
	flag = get_prediction(data, centers, predicts);

	int iteration = 0;

	while (flag)
	{
		cout << "iteration : " << ++iteration << endl;
		// update centers
		get_centers(data, centers, predicts);
		centers.print();

		flag = get_prediction(data, centers, predicts);
		//cout << predicts << endl;
	}

	return 0;
}

double get_distance(const CImg<double> &v1, const CImg<double> &v2)
{
	CImg<double> diff = v1 - v2;
	return diff.mul(diff).sum();
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

void get_centers(const CImg<double> &data, CImg<double> &centers,
		const vector<int> &predicts)
{
	//CImg<double> means(1, centers.height());
	CImg<double> counts(centers.width(), centers.height());
	CImg<double> new_centers(centers);
	new_centers = 0.0;
	counts = 0.0;

	double min = data.min();
	double max = data.max();

	for (int i = 0; i < data.height(); i++)
	{
		int index = predicts[i];
		image_row_add(new_centers, index, data.get_row(i));
		image_row_add(counts, index, 1);
	}

	for (int i = 0; i < counts.height(); i++)
	{
		if (counts(0, i) == 0)
		{
			image_row_add(counts, i, 1);
			image_row_add(new_centers, i,
					CImg<double>(centers.width(),1).rand(min, max));
			cout << "random center " << i << endl;
		}
	}

	new_centers.div(counts);
	centers = new_centers;
}



#endif /* KMEANS_H_ */
