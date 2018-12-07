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
void detectAndDisplay( Mat frame, int** coordinates );
float f_one();

/** Global variables */
String cascade_name = "frontalface.xml";
CascadeClassifier cascade;
int tp,truefaces,detected;

/** @function main */
int main( int argc, const char** argv )
{
  // 1. Read Input Image
  Mat frame = imread(argv[1], CV_LOAD_IMAGE_COLOR);

	int i = 0;
	char[11][4] coords;

 	while (getline(argv[2], coords[i][0] ',')) {
	getline(argv[2],coords[i][1]);
	getline(argv[2],coords[i][2]);
	getline(argv[2],coords[i][3]);
 	i++;
	}

	int truefaces = i;
	int intcoords [truefaces][4];

	for(int j = 0; j < truefaces; j++){
	   intcoords[j][0] = atoi(coords[j][0]);
	   intcoords[j][1] = atoi(coords[j][1]);
		 intcoords[j][0] = atoi(coords[j][2]);
		 intcoords[j][1] = atoi(coords[j][3]);
	}

	// 2. Load the Strong Classifier in a structure called `Cascade'
	if( !cascade.load( cascade_name ) ){ printf("--(!)Error loading\n"); return -1; };

	// 3. Detect Faces and Display Result
	detectAndDisplay( frame, intcoords );

	// 4. Save Result Image
	imwrite( "vj.jpg", frame );

	std::cout.precision(5);
  std::cout.setf(std::ios::fixed);
	std::cout << f_one() << std::endl;

	return 0;
}

float f_one(){
  int fp = truefaces - tp;
	int fn = detected - tp;
	float precision = tp/(tp+fp);
  float recall = tp/(tp+fn);
  float f1 = 2 * (precision * recall) / (precision + recall);
  return f1;
}

/** @function detectAndDisplay */
void detectAndDisplay( Mat frame, int** coordinates )
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
  detected = faces.size();


  // 4. Draw box around faces found
	for( int i = 0; i < faces.size(); i++ )
	{
		rectangle(frame, Point(faces[i].x, faces[i].y), Point(faces[i].x + faces[i].width, faces[i].y + faces[i].height), Scalar( 0, 255, 0 ), 2);
  }

	//5. calculate true positives
  for(int j = 0; i < truefaces; j++)
	{
		int truematch = 0;
		//iterate through true rectangles
		//for each, iterate through detected rectangles
		//if the detected rectangle is within the true rectangle, increment tp

		for( int i = 0; i < faces.size(); i++ )
		{
		   tl[2] = {faces[i].x, faces[i].y};
			 br[2] = {faces[i].x + faces[i].width, faces[i].y + faces[i].height};
			 //if: detected rectangle's top left point is within true rectangle's top left Point
			 //and detected rectangle's bottom right point is within true rectangle's bottom left point
		 	 if( (coordinates[i][0] >= tl[0]) && (coordinates[i][1] <= tl[1]) && (coordinates[i][2] >= br[0]) && (coordinates[i][3] <= br[1]) )
			 {
				 truematch = 1;
       }
		}
	if ( truematch == 1 ){
		tp++;
	}
	}
}
