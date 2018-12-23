#include <opencv2/opencv.hpp>
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
using namespace std;

int darray[1000][1000];

#define CHUNK_SIZE 20
#define NUM_THREADS 2

void convertImageTo2DArray(string fileName) 
{
   cv::Mat img = cv::imread(fileName);

	for(int i=0;i<img.cols;i++)
	{
		for(int j=0;j<img.rows;j++)
		{
			darray[i][j]=img.at<int>(i,j);
        }
    }
}

template <size_t rows, size_t cols>
void printMatrix(int (&Z)[rows][cols])
{
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            printf("%2d ", Z[i][j]);
        }
        printf("\n");
    }
}

template <size_t rows, size_t cols>
void initializeMatrix(int (&Z)[rows][cols])
{
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            Z[i][j] = 0;
        }
    }
}

double secventialRotate(string path, string new_path, int direction)
{
	cv::Vec3b rotated_pixel;
	cv::Vec3b original_pixel;
	cv::Mat img = cv::imread(path, cv::IMREAD_COLOR);

	// Create mat with alpha channel
	// 8UC3 helps you access the RGB data
	cv::Mat rotated_img(img.cols, img.rows, CV_8UC3);

	int cols = img.cols;
	int rows = img.rows;

	double start = omp_get_wtime();
	for (int i = 0; i < rows; i++)
	{
		for (int j = 0; j < cols; j++)
		{

			original_pixel = img.at<cv::Vec3b>(i, j);

			if (direction == 1)
				rotated_img.at<cv::Vec3b>(j, rows - i - 1) = original_pixel;
			else
				rotated_img.at<cv::Vec3b>(cols - j - 1, i) = original_pixel;
		}
	}

	double stop = omp_get_wtime();

	cv::imwrite(new_path, rotated_img);

	return stop - start;
}

double firstForRotate(string path, string new_path, int direction)
{
	cv::Vec3b rotated_pixel;
	cv::Vec3b original_pixel;
	cv::Mat img = cv::imread(path, cv::IMREAD_COLOR);

	// Create mat with alpha channel
	// 8UC3 helps you access the RGB data
	cv::Mat rotated_img(img.cols, img.rows, CV_8UC3);

	int cols = img.cols;
	int rows = img.rows;
	int i, j, tid, nthreads;

	double start = omp_get_wtime();
	#pragma omp parallel shared(img, nthreads) private(i, j, tid)
	{

 		tid = omp_get_thread_num();

        if (tid == 0)
        {
            nthreads = omp_get_num_threads();
        }

		#pragma omp for schedule (static, CHUNK_SIZE)
		for (i = 0; i < rows; i++) 
		{
			for (j = 0; j < cols; j++) 
			{
				original_pixel = img.at<cv::Vec3b>(i, j);

				if (direction == 1)
					rotated_img.at<cv::Vec3b>(j, rows - i - 1) = original_pixel;
				else
					rotated_img.at<cv::Vec3b>(cols - j - 1, i) = original_pixel;
			}
		}
	}

	double stop = omp_get_wtime();

	cv::imwrite(new_path, rotated_img);

	return stop - start;
}

double secondForRotate(string path, string new_path, int direction)
{
	cv::Vec3b rotated_pixel;
	cv::Vec3b original_pixel;
	cv::Mat img = cv::imread(path, cv::IMREAD_COLOR);

	// Create mat with alpha channel
	// 8UC3 helps you access the RGB data
	cv::Mat rotated_img(img.cols, img.rows, CV_8UC3);

	int cols = img.cols;
	int rows = img.rows;

	double start = omp_get_wtime();
	for (int i = 0; i < rows; i++)
	{
		#pragma omp parallel num_threads(NUM_THREADS) 
		{
			#pragma omp for schedule (dynamic, CHUNK_SIZE)
			for (int j = 0; j < cols; j++)
			{
				original_pixel = img.at<cv::Vec3b>(i, j);

				if (direction == 1)
					rotated_img.at<cv::Vec3b>(j, rows - i - 1) = original_pixel;
				else
					rotated_img.at<cv::Vec3b>(cols - j - 1, i) = original_pixel;
			}
		}
	}

	double stop = omp_get_wtime();

	cv::imwrite(new_path, rotated_img);

	return stop - start;
}

double allForRotate(string path, string new_path, int direction)
{
	cv::Vec3b rotated_pixel;
	cv::Vec3b original_pixel;
	cv::Mat img = cv::imread(path, cv::IMREAD_COLOR);

	// Create mat with alpha channel
	// 8UC3 helps you access the RGB data
	cv::Mat rotated_img(img.cols, img.rows, CV_8UC3);

	int cols = img.cols;
	int rows = img.rows;

	double start = omp_get_wtime();
	#pragma omp parallel num_threads(NUM_THREADS) 
	{
		#pragma omp for schedule (dynamic, CHUNK_SIZE) collapse(2)
		for (int i = 0; i < rows; i++)
		{
			for (int j = 0; j < cols; j++)
			{

				original_pixel = img.at<cv::Vec3b>(i, j);

				if (direction == 1)
					rotated_img.at<cv::Vec3b>(j, rows - i - 1) = original_pixel;
				else
					rotated_img.at<cv::Vec3b>(cols - j - 1, i) = original_pixel;
			}
		}
	}

	double stop = omp_get_wtime();

	cv::imwrite(new_path, rotated_img);

	return stop - start;
}

void experimentMode()
{
	string inputFileName = "test.png";    

    double tsc, tsd, tse, tsf, tsg, tsh, tsi, tsj, tsk, tsl, tsm, tsn;
    tsc = tsd = tse = tsf = tsg = tsh = tsi = tsj = tsk = tsl = tsm = tsn = 0;

    for (int i = 0; i < 10; i++)
    {

//         // inverse of RGB colors

//         float timeSpent = secventialInverseOfRgbColors(filename_input);
//         tsc += timeSpent;
//         drawImage("images/colors_inverse_S", i);

//         timeSpent = firstForInverseOfRgbColors(filename_input);
//         tsd += timeSpent;
//         drawImage("images/colors_inverse_FF.png", i);

//         timeSpent = secondForInverseOfRgbColors(filename_input);
//         tse += timeSpent;
//         drawImage("images/colors_inverse_SF.png", i);

//         timeSpent = bothForsInverseOfRgbColors(filename_input);
//         tsf += timeSpent;
//         drawImage("images/colors_inverse_AF.png", i);

//         // blur

//         timeSpent = secventialBlur(filename_input);
//         tsg += timeSpent;
//         drawImage("images/blur_S.png", i);

//         timeSpent = firstForBlur(filename_input);
//         tsh += timeSpent;
//         drawImage("images/blur_FF.png", i);

//         timeSpent = secondForBlur(filename_input);
//         tsi += timeSpent;
//         drawImage("images/blur_SF.png", i);

//         timeSpent = bothForsBlur(filename_input);
//         tsj += timeSpent;
//         drawImage("images/blur_AF.png", i);

        
        // rotate 90 degrees

        double timeSpent = secventialRotate(inputFileName, "img/rotate_S.png", 1);
        tsk += timeSpent;

        timeSpent = firstForRotate(inputFileName, "img/rotate_FF.png", 1);
        tsl += timeSpent;

        timeSpent = secondForRotate(inputFileName, "img/rotate_SF.png", 1);
        tsm += timeSpent;

        timeSpent = allForRotate(inputFileName, "img/rotate_AF.png", 1);
        tsn += timeSpent;
    }

    printf("%2f ", tsc / 10);
    printf("%2f ", tsd / 10);
    printf("%2f ", tse / 10);
    printf("%2f ", tsf / 10);
    printf("%2f ", tsg / 10);
    printf("%2f ", tsh / 10);
    printf("%2f ", tsi / 10);
    printf("%2f ", tsj / 10);
    printf("%2f ", tsk / 10);
    printf("%2f ", tsl / 10);
    printf("%2f ", tsm / 10);
    printf("%2f\n", tsn / 10);

    // printf("Image width : %d\n", width);
    // printf("Image height: %d\n", height);
}

int main(int argc, char *argv[])
{

	experimentMode();
}