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
	// 1. Read Input Image
	Mat frame1 = imread("dart1.jpg", CV_LOAD_IMAGE_COLOR);
	// 2. Load the Strong Classifier in a structure called `Cascade'
	if( !cascade.load( cascade_name ) ){ printf("--(!)Error loading\n"); return -1; };
	// 3. Detect Faces and Display Result
	detectAndDisplay( frame1 );
	// 4. Save Result Image
	imwrite( "dartboard1.jpg", frame1 );
	// 1. Read Input Image
	Mat frame2 = imread("dart2.jpg", CV_LOAD_IMAGE_COLOR);
	// 2. Load the Strong Classifier in a structure called `Cascade'
	if( !cascade.load( cascade_name ) ){ printf("--(!)Error loading\n"); return -1; };
	// 3. Detect Faces and Display Result
	detectAndDisplay( frame2 );
	// 4. Save Result Image
	imwrite( "dartboard2.jpg", frame2 );
	// 1. Read Input Image
	Mat frame3 = imread("dart3.jpg", CV_LOAD_IMAGE_COLOR);
	// 2. Load the Strong Classifier in a structure called `Cascade'
	if( !cascade.load( cascade_name ) ){ printf("--(!)Error loading\n"); return -1; };
	// 3. Detect Faces and Display Result
	detectAndDisplay( frame3 );
	// 4. Save Result Image
	imwrite( "dartboard3.jpg", frame3 );
	// 1. Read Input Image
	Mat frame4 = imread("dart4.jpg", CV_LOAD_IMAGE_COLOR);
	// 2. Load the Strong Classifier in a structure called `Cascade'
	if( !cascade.load( cascade_name ) ){ printf("--(!)Error loading\n"); return -1; };
	// 3. Detect Faces and Display Result
	detectAndDisplay( frame4 );
	// 4. Save Result Image
	imwrite( "dartboard4.jpg", frame4 );
	// 1. Read Input Image
	Mat frame5 = imread("dart5.jpg", CV_LOAD_IMAGE_COLOR);
	// 2. Load the Strong Classifier in a structure called `Cascade'
	if( !cascade.load( cascade_name ) ){ printf("--(!)Error loading\n"); return -1; };
	// 3. Detect Faces and Display Result
	detectAndDisplay( frame5 );
	// 4. Save Result Image
	imwrite( "dartboard5.jpg", frame5 );
	// 1. Read Input Image
	Mat frame6 = imread("dart6.jpg", CV_LOAD_IMAGE_COLOR);
	// 2. Load the Strong Classifier in a structure called `Cascade'
	if( !cascade.load( cascade_name ) ){ printf("--(!)Error loading\n"); return -1; };
	// 3. Detect Faces and Display Result
	detectAndDisplay( frame6 );
	// 4. Save Result Image
	imwrite( "dartboard6.jpg", frame6 );
	// 1. Read Input Image
	Mat frame7 = imread("dart7.jpg", CV_LOAD_IMAGE_COLOR);
	// 2. Load the Strong Classifier in a structure called `Cascade'
	if( !cascade.load( cascade_name ) ){ printf("--(!)Error loading\n"); return -1; };
	// 3. Detect Faces and Display Result
	detectAndDisplay( frame7 );
	// 4. Save Result Image
	imwrite( "dartboard7.jpg", frame7 );
	// 1. Read Input Image
	Mat frame8 = imread("dart8.jpg", CV_LOAD_IMAGE_COLOR);
	// 2. Load the Strong Classifier in a structure called `Cascade'
	if( !cascade.load( cascade_name ) ){ printf("--(!)Error loading\n"); return -1; };
	// 3. Detect Faces and Display Result
	detectAndDisplay( frame8 );
	// 4. Save Result Image
	imwrite( "dartboard8.jpg", frame8 );
	// 1. Read Input Image
	Mat frame9 = imread("dart9.jpg", CV_LOAD_IMAGE_COLOR);
	// 2. Load the Strong Classifier in a structure called `Cascade'
	if( !cascade.load( cascade_name ) ){ printf("--(!)Error loading\n"); return -1; };
	// 3. Detect Faces and Display Result
	detectAndDisplay( frame9 );
	// 4. Save Result Image
	imwrite( "dartboard9.jpg", frame9 );
	// 1. Read Input Image
	Mat frame10 = imread("dart10.jpg", CV_LOAD_IMAGE_COLOR);
	// 2. Load the Strong Classifier in a structure called `Cascade'
	if( !cascade.load( cascade_name ) ){ printf("--(!)Error loading\n"); return -1; };
	// 3. Detect Faces and Display Result
	detectAndDisplay( frame10 );
	// 4. Save Result Image
	imwrite( "dartboard10.jpg", frame10 );
	// 1. Read Input Image
	Mat frame11 = imread("dart11.jpg", CV_LOAD_IMAGE_COLOR);
	// 2. Load the Strong Classifier in a structure called `Cascade'
	if( !cascade.load( cascade_name ) ){ printf("--(!)Error loading\n"); return -1; };
	// 3. Detect Faces and Display Result
	detectAndDisplay( frame11 );
	// 4. Save Result Image
	imwrite( "dartboard11.jpg", frame11 );
	// 1. Read Input Image
	Mat frame12 = imread("dart12.jpg", CV_LOAD_IMAGE_COLOR);
	// 2. Load the Strong Classifier in a structure called `Cascade'
	if( !cascade.load( cascade_name ) ){ printf("--(!)Error loading\n"); return -1; };
	// 3. Detect Faces and Display Result
	detectAndDisplay( frame12 );
	// 4. Save Result Image
	imwrite( "dartboard12.jpg", frame12 );
	// 1. Read Input Image
	Mat frame13 = imread("dart13.jpg", CV_LOAD_IMAGE_COLOR);
	// 2. Load the Strong Classifier in a structure called `Cascade'
	if( !cascade.load( cascade_name ) ){ printf("--(!)Error loading\n"); return -1; };
	// 3. Detect Faces and Display Result
	detectAndDisplay( frame13 );
	// 4. Save Result Image
	imwrite( "dartboard13.jpg", frame13 );
	// 1. Read Input Image
	Mat frame14 = imread("dart14.jpg", CV_LOAD_IMAGE_COLOR);
	// 2. Load the Strong Classifier in a structure called `Cascade'
	if( !cascade.load( cascade_name ) ){ printf("--(!)Error loading\n"); return -1; };
	// 3. Detect Faces and Display Result
	detectAndDisplay( frame14 );
	// 4. Save Result Image
	imwrite( "dartboard14.jpg", frame14 );
	// 1. Read Input Image
	Mat frame15 = imread("dart15.jpg", CV_LOAD_IMAGE_COLOR);
	// 2. Load the Strong Classifier in a structure called `Cascade'
	if( !cascade.load( cascade_name ) ){ printf("--(!)Error loading\n"); return -1; };
	// 3. Detect Faces and Display Result
	detectAndDisplay( frame15 );
	// 4. Save Result Image
	imwrite( "dartboard15.jpg", frame15 );
	return 0;
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
