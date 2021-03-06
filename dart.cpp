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


/** @function main */
int main( int argc, const char** argv )
{
	// 1. Read Input Image
	Mat frame0 = imread("dart0.jpg", CV_LOAD_IMAGE_COLOR);
	// 2. Load the Strong Classifier in a structure called `Cascade'
	if( !cascade.load( cascade_name ) ){ printf("--(!)Error loading\n"); return -1; };
	// 3. Detect Faces and Display Result
	detectAndDisplay( frame0 );
	// 4. Save Result Image
	imwrite( "dartboard0.jpg", frame0 );
}

void houghSpace(Mat frame) {
  const float pi = 3.14159
	for (int i = 0; i < frame.row; i++){
		for (int j = 0; i < frame.cols; i++){
		float line1[2] = {j,pi/2};
		float line2[2] = {((i*cos(pi/4))+(*sin(pi/4))),pi/4};
		float line3[2] = {i,0};
		float line4[2] = {((i*cos(pi/4))+(*sin(pi/4))),pi/4};
	  }
	}
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
