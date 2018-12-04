/////////////////////////////////////////////////////////////////////////////
//
// COMS30121 - dart.cpp
//
/////////////////////////////////////////////////////////////////////////////

// header inclusion
// header inclusion
#include <stdio.h>
#include <math.h>
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/opencv.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>

using namespace std;
using namespace cv;

/** Function Headers */
void detectAndDisplay( Mat frame );
Mat cannify_image( Mat frame );
void hough_transform( Mat frame );

/** Global variables */
String cascade_name = "dartcascade/cascade.xml";
CascadeClassifier cascade;

/** @function main */
int main( int argc, const char** argv )
{
	// 1. Read Input Image
	Mat frame0 = imread("dart0.jpg", CV_LOAD_IMAGE_COLOR);
	// 2. Load the Strong Classifier in a structure called `Cascade'
	if( !cascade.load( cascade_name ) ){ printf("--(!)Error loading\n"); return -1; };
	// 3. Blur and Canny edge detect the image
  Mat edgemapped = cannify_image(frame0);
  // 4. Perform general Hough Transform on Image
  hough_transform(edgemapped);
}

//Prep the image for Hough Transform
Mat cannify_image( Mat frame) {
  Mat frame_gray;
	cvtColor( frame, frame_gray, CV_BGR2GRAY );
	equalizeHist( frame_gray, frame_gray );
  Mat frame_grayblur;
  blur( frame_gray, frame_grayblur, Size(3,3) );
	Mat edgemap(frame_grayblur.size(), CV_8UC1);
	Canny(frame_grayblur, edgemap, 100, 300, 3);
  imwrite("canny.jpg", edgemap);
  return edgemap;
}

void hough_transform( Mat frame ) {
  const float pi = 3.14159;
  int frame_height = frame.rows;
	int frame_width = frame.cols;

	//Generate empty Hough Space
  int theta_max = 180;
  //rho should be ~769
  int rho_max = round(sqrt(pow(frame_width, 2) + pow(frame_height, 2)));
  Mat hough_space = Mat::zeros(theta_max, rho_max, CV_8UC1);

	//Iterate over each pixel
	for (int x = 0; x < frame_height; x++){

		for (int y = 0; y < frame_width; y++){
      //If the pixel is an edge
      int intensity = (int)frame.at<uchar>(x, y);
      if( intensity > 250) {

        for(int theta = -90; theta < 90; theta++){
          int t = (theta * pi)/180;
          int rho = (x * cos(t)) + (y * sin(t));

          int current = (int)hough_space.at<uchar>(theta+90, rho);
          int new_current = current + 1;
          hough_space.at<uchar>(theta+90, rho) = (uchar)new_current;
  			}
      }
	  }
	}
  imwrite("hough.jpg", hough_space);
}

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
