#define _CRT_SECURE_NO_WARNINGS
#include <cstdio>
#include <iostream>
#include <algorithm>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include <opencv2/imgcodecs.hpp>

using namespace cv;
using namespace std;

// Min region: for init split and image process
struct min_r
{
	int position[2];	// top left corner of the region
	int size[2];		// height and width
	min_r * next;
};

/*A series of region
For merged region
Store in link list
Also has the feature array of this series of region
*/
typedef struct r
{
	min_r* composition;	// static array ptr
	int size = 1;	// area
	float color[25];			// 25 bin
	float texture[3][8][10];	// 240 bin
}r;

/*Record similarity
Have the pointer towards r, to access the relative data and calc. similarity
*/
struct s
{
	int similarity;
	r* ri;
	r* rj;
};

/*Store image
A simple image store struct
*/
typedef struct Image
{
	vector<vector<vector<uchar>>> data;
	int rows;
	int cols;
}image;

/*Convert 
cv::Mat and std::vector
In order to calculate the similarity, using data type uchar to store the image data.
There is a blank border in four directions,
So the first pixel of each row and col starts at 1
*/
vector<vector<vector<uchar>>> mat2vector(Mat image_in)
{
	uchar * data_ptr = (uchar*)image_in.data;

	vector<vector<vector<uchar>>> array(3, vector<vector<uchar>>(image_in.rows + 2, vector<uchar>(image_in.cols + 2)));

	// set for edge to empty
	for (int c = 0; c < 3; c++)
		for (int i = 0; i < image_in.rows + 2; i++)
			array[c][i][0] = array[c][i][image_in.cols + 1] = 0;
	for (int c = 0; c < 3; c++)
		for (int i = 0; i < image_in.cols; i++)
			array[c][0][i] = array[c][image_in.rows + 1][i] = 0;

	for (int c = 0; c < 3; c++)
		for (int i = 0; i < image_in.rows; i++)
			for (int j = 0; j < image_in.cols; j++)
				array[c][i + 1][j + 1] = data_ptr[c + 3 * i + 3 * image_in.rows*j];

	return array;
}

Mat vector2mat(vector<vector<vector<uchar>>> image_array)
{
	int rows = image_array[0].size() - 2;
	int cols = image_array[0][0].size() - 2;
	Mat img(rows, cols, CV_8UC3);
	uchar * data_ptr = (uchar*)img.data;

	for (int c = 0; c < 3; c++)
	{
		for (int i = 0; i < rows; i++)
		{
			for (int j = 0; j < cols; j++)
			{
				data_ptr[c + 3 * i + 3 * rows * j] = image_array[c][i + 1][j + 1];
			}
		}
	}
	return img;
}

/*Split image
Split a image into many small regions.
Given a size of small region
*/
min_r * split_image(Image img, int height = 3, int width = 3)
{
	int h_split_num = ceil(img.rows / (float)height);
	int w_split_num = ceil(img.cols / (float)width);
	
	min_r * split_head = new min_r;
	min_r ** split = &split_head;

	for (int i = 0; i < h_split_num; i++)
	{
		for (int j = 0; j < w_split_num; j++)
		{
			(*split)->position[0] = i * height + 1;
			(*split)->position[1] = j * width + 1;
			(*split)->size[0] = (i * height + height < img.rows) ? height : (img.rows - i * height);
			(*split)->size[1] = (j * width + width < img.cols) ? width : (img.cols - j * width);
			(*split)->next = new min_r;
			split = &((*split)->next);
		}
	}

	(*split) = NULL;

	return split_head;
}

/*Region Packging
Divide and encapsulate link list of min_region into single r
without feature 
*/
vector<r> RegionPackgingPipline(min_r *minregion_linklist)
{
	vector<r> region_list;
	r temp;
	min_r * next;
	min_r * current = minregion_linklist;
	while (current != NULL)
	{
		//preprocess
		next = current->next;
		current->next = NULL;

		temp.composition = current;
		temp.size = current->size[0] * current->size[1];
		region_list.push_back(temp);
		current = next;
	}
	return region_list;
}

/*calc color feature
input: a region with only one min_region
store the feature into InitRegionList
*/
void color_feature(r * InitRegionList, Image * img)
{
	// region position, size
	min_r *region = InitRegionList->composition;
	
	// initialize color feature
	for (int bin = 0; bin < 25; bin++)
		InitRegionList->color[bin] = 0;
	int bin;

	for (int row = 0; row < region->size[0]; row++)
	{
		for (int col = 0; col < region->size[1]; col++)
		{
			for (int channel = 0; channel < 3; channel++)
			{
				bin = img->data[channel][region->position[0] + row][region->position[1] + col] * 25 / 256;
				InitRegionList->color[bin]+=1.0/(InitRegionList->size);
			}
		}
	}
	return;
}

/*Texture feature
*/
void texture_feature(r * InitRegionList, Image * img)
{

}

int main(void) {
	Mat image_in = imread("1.jpg");
	Mat image_out;
	vector<vector<vector<uchar>>> image_array;
	
	printf("width£º%d height: %d\n", image_in.rows, image_in.cols);
	
	image_array = mat2vector(image_in);
	Image init_image = { image_array,image_in.rows,image_in.cols };

	
	//// mat2vector and vector2mat test case
	//// cvt mat to 2 dim uchar array
	//image_array = mat2vector(image_in);
	//for (int i = 0; i < image_in.rows; i++)
	//{
	//	image_array[0][i][10] = image_array[0][i][11] = image_array[0][i][12] = 0;
	//	image_array[1][i][10] = image_array[1][i][11] = image_array[1][i][12] = 0;
	//	image_array[2][i][10] = image_array[2][i][11] = image_array[2][i][12] = 127;
	//}
	//// cvt uchar array to mat
	//image_out = vector2mat(image_array);
	//imshow("Imgae in", image_in);
	//imshow("image out", image_out);
	//waitKey(5);
	

	min_r * split_min_region = split_image(init_image,3,3);

	
	//// split region test case
	//min_r * current = split_min_region;
	//int count = 0;
	//while (current != NULL)
	//{
	//	printf("%d: %d %d %d %d\n", count, current->position[0], current->position[1], current->size[0], current->size[1]);
	//	count++;
	//	current = current->next;
	//}
	
	vector<r> split_region = RegionPackgingPipline(split_min_region);

	for (int i = 0; i < split_region.size(); i++)
		color_feature(&(split_region[i]), &init_image);

	//// color feature test case
	//for (int i = 0; i < split_region.size(); i++)
	//{
	//	float sum = 0.0;
	//	for (int ii = 0; ii < 25; ii++)
	//	{
	//		sum += split_region[i].color[ii];
	//	}
	//	if (sum - 3 > 0.001 || sum - 3 < -0.001)
	//	{
	//		printf("%f", sum);
	//		return -1;
	//	}
	//}



	return 0;
}