/////////////////////////////////////////////////////////////////////////////
//
// COMS30121 - face.cpp
//
/////////////////////////////////////////////////////////////////////////////

// header inclusion
// header inclusion
#include <stdio.h>
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/opencv.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <stdio.h>

using namespace std;
using namespace cv;

/** Function Headers */
void detectAndDisplay( Mat frame );

/** Global variables */
String cascade_name = "dartcascade/cascade.xml";
CascadeClassifier cascade;

void GaussianBlur(
	cv::Mat &input,
	int size,
	cv::Mat &blurredOutput);

void sobelX(
	cv::Mat &output
);

void sobelY(
	cv::Mat &output
);

void sobel(
	cv::Mat &input,
	cv::Mat &OutputX,
	cv::Mat &OutputY,
	cv::Mat &OutputM,
	cv::Mat &OutputD,
	cv::Mat &OutputD2
);

/** @function main */
int main( int argc, const char** argv )
{
	// 1. Read Input Image
	Mat frame0 = imread("dart0.jpg", CV_LOAD_IMAGE_COLOR);
	// 2. Load the Strong Classifier in a structure called `Cascade'
	//if( !cascade.load( cascade_name ) ){ printf("--(!)Error loading\n"); return -1; };
	// 3. Detect Faces and Display Result
	//detectAndDisplay( frame0 );
	// 4. Save Result Image
	//imwrite( "dartboard0.jpg", frame0 );

	Mat frame_gray;

	// 1. Prepare Image by turning it into Grayscale and normalising lighting
	cvtColor( frame0, frame_gray, CV_BGR2GRAY );
	equalizeHist( frame_gray, frame_gray );

  Mat frame_grayblur;

	GaussianBlur(frame_gray,23,frame_grayblur);

	Mat edgemap(frame_grayblur.size(), CV_32FC1);

	Canny(frame_grayblur, edgemap, 1, 15);

	imwrite("edge.jpg", edgemap);

	Mat frameX(frame_grayblur.size(), CV_32FC1);
  Mat frameY(frame_grayblur.size(), CV_32FC1);
  Mat frameM(frame_grayblur.size(), CV_32FC1);
  Mat frameD(frame_grayblur.size(), CV_32FC1);
  Mat frameD2(frame_grayblur.size(), CV_32FC1);

	sobel(frame_grayblur,frameX,frameY,frameM,frameD,frameD2);

	Mat frameMT(frame_grayblur.size(), CV_32FC1);
  threshold(frameM,frameMT, 20, 255, THRESH_BINARY);
	Mat frameD2T(frame_grayblur.size(), CV_32FC1);
	threshold(frameD2,frameD2T, 20, 255, THRESH_BINARY);

  imwrite("mthresh.jpg", frameMT);
	imwrite("dthresh.jpg", frameD2T);
	imwrite("mag.jpg", frameM);
	imwrite("dir.jpg", frameD2);
}

// // /** @function houghSpace */
// void houghSpace( Mat frame )
// {
//   const float pi = 3.14159
// 	for (int i = 0; i < frame.row; i++){
// 		for (int j = 0; i < frame.cols; i++){
// 		  float line1[2] = {j,pi/2};
// 	  	float line2[2] = {((i*cos(pi/4))+(*sin(pi/4))),pi/4};
// 	  	float line3[2] = {i,0};
// 	  	float line4[2] = {((i*cos(pi/4))+(*sin(pi/4))),pi/4};
// 	  }
// 	}
// }

/** @function detectAndDisplay */
void detectAndDisplay( Mat frame )
{
	std::vector<Rect> faces;
	Mat frame_gray;

	// 1. Prepare Image by turning it into Grayscale and normalising lighting
	cvtColor( frame, frame_gray, CV_BGR2GRAY );
	equalizeHist( frame_gray, frame_gray );

	// 2. Perform Viola-Jones Object Detection
	cascade.detectMultiScale( frame_gray, faces, 1.1, 1, 0|CV_HAAR_SCALE_IMAGE, Size(50, 50), Size(500,500) );

       // 3. Print number of Faces found
	std::cout << faces.size() << std::endl;

       // 4. Draw box around faces found
	for( int i = 0; i < faces.size(); i++ )
	{
		rectangle(frame, Point(faces[i].x, faces[i].y), Point(faces[i].x + faces[i].width, faces[i].y + faces[i].height), Scalar( 0, 255, 0 ), 2);
	}

}

void sobelX( cv::Mat &output ) {
		output.create(3,3,CV_32FC1);
		output.at<float>(0,0) = (float) -1;
		output.at<float>(0,1) = (float) 0;
		output.at<float>(0,2) = (float) 1;
		output.at<float>(1,0) = (float) -2;
		output.at<float>(1,1) = (float) 0;
		output.at<float>(1,2) = (float) 2;
		output.at<float>(2,0) = (float) -1;
		output.at<float>(2,1) = (float) 0;
		output.at<float>(2,2) = (float) 1;
}

void sobelY( cv::Mat &output ) {
	 output.create(3,3,CV_32FC1);
	 output.at<float>(0,0) = (float) -1;
	 output.at<float>(0,1) = (float) -2;
	 output.at<float>(0,2) = (float) -1;
	 output.at<float>(1,0) = (float) 0;
	 output.at<float>(1,1) = (float) 0;
	 output.at<float>(1,2) = (float) 0;
	 output.at<float>(2,0) = (float) 1;
	 output.at<float>(2,1) = (float) 2;
	 output.at<float>(2,2) = (float) 1;
}

void sobel(cv::Mat &input, cv::Mat &OutputX, cv::Mat &OutputY, cv::Mat &OutputM, cv::Mat &OutputD, cv::Mat &OutputD2)
{
	// intialise the output using the input
	// OutputX.create(input.size(), input.type());
	// OutputY.create(input.size(), input.type());
  Mat kX;
	Mat kY;
  sobelX(kX);
	sobelY(kY);

	int kernelRadiusX = ( kX.size[0] - 1 ) / 2;
	int kernelRadiusY = ( kX.size[1] - 1 ) / 2;

  //create padded input to negate border problems
	cv::Mat paddedInput;
	cv::copyMakeBorder( input, paddedInput,
		kernelRadiusX, kernelRadiusX, kernelRadiusY, kernelRadiusY,
		cv::BORDER_REPLICATE );

	// now we can do the convolution
	for ( int i = 0; i < input.rows; i++ )
	{
		for( int j = 0; j < input.cols; j++ )
		{
			float sumX = 0.0;
			float sumY = 0.0;
			for( int m = -kernelRadiusX; m <= kernelRadiusX; m++ )
			{
				for( int n = -kernelRadiusY; n <= kernelRadiusY; n++ )
				{
					// find the correct indices we are using
					int imagex = i + m + kernelRadiusX;
					int imagey = j + n + kernelRadiusY;
					int kernelx = m + kernelRadiusX;
					int kernely = n + kernelRadiusY;

					// get the values from the padded image and the kernel
					float imageval = ( float ) paddedInput.at<uchar>( imagex, imagey );
					float kernelvalX = kX.at<float>( kernelx, kernely );
					float kernelvalY = kY.at<float>( kernelx, kernely );

					// do the multiplication
					sumX += imageval * kernelvalX;
					sumY += imageval * kernelvalY;
				}
			}
			// set the output value as the sum of the convolution
			OutputX.at<float>(i, j) = sumX;
			OutputY.at<float>(i, j) = sumY;
			OutputM.at<float>(i, j) = sqrt(pow(sumX,2)+pow(sumY,2));
			if (sumX == 0){
				OutputD.at<float>(i, j) = 0;
				OutputD2.at<float>(i, j) = 0;
			}
			else{
        OutputD.at<float>(i, j) = atan((sumY)/(sumX));
				OutputD2.at<float>(i, j) = 180*(atan((sumY)/(sumX)))/3.1415926535897;
			}


		}
	}
}

void GaussianBlur(cv::Mat &input, int size, cv::Mat &blurredOutput)
{
	// intialise the output using the input
	blurredOutput.create(input.size(), input.type());

	// create the Gaussian kernel in 1D
	cv::Mat kX = cv::getGaussianKernel(size, -1);
	cv::Mat kY = cv::getGaussianKernel(size, -1);

	// make it 2D multiply one by the transpose of the other
	cv::Mat kernel = kX * kY.t();

	//CREATING A DIFFERENT IMAGE kernel WILL BE NEEDED
	//TO PERFORM OPERATIONS OTHER THAN GUASSIAN BLUR!!!

	// we need to create a padded version of the input
	// or there will be border effects
	int kernelRadiusX = ( kernel.size[0] - 1 ) / 2;
	int kernelRadiusY = ( kernel.size[1] - 1 ) / 2;



	cv::Mat paddedInput;
	cv::copyMakeBorder( input, paddedInput,
		kernelRadiusX, kernelRadiusX, kernelRadiusY, kernelRadiusY,
		cv::BORDER_REPLICATE );

	// now we can do the convoltion
	for ( int i = 0; i < input.rows; i++ )
	{
		for( int j = 0; j < input.cols; j++ )
		{
			double sum = 0.0;
			for( int m = -kernelRadiusX; m <= kernelRadiusX; m++ )
			{
				for( int n = -kernelRadiusY; n <= kernelRadiusY; n++ )
				{
					// find the correct indices we are using
					int imagex = i + m + kernelRadiusX;
					int imagey = j + n + kernelRadiusY;
					int kernelx = m + kernelRadiusX;
					int kernely = n + kernelRadiusY;

					// get the values from the padded image and the kernel
					int imageval = ( int ) paddedInput.at<uchar>( imagex, imagey );
					double kernalval = kernel.at<double>( kernelx, kernely );

					// do the multiplication
					sum += imageval * kernalval;
				}
			}
			// set the output value as the sum of the convolution
			blurredOutput.at<uchar>(i, j) = (uchar) sum;
		}
	}
}
