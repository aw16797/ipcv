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
#include <string>
#include <stdlib.h>
#include <fstream>
#include <sstream>

using namespace std;
using namespace cv;

/** Function Headers */
void detectAndDisplay( Mat frame );
int toInt( char x );
float f_one();

/** Global variables */
String cascade_name = "frontalface.xml";
CascadeClassifier cascade;
int tp,truefaces,detected;
int coordinates[11][4];

/** @function main */
int main( int argc, char** argv )
{
  // 1. Read Input Image
  Mat frame = imread("images/dart15.jpg", CV_LOAD_IMAGE_COLOR);
	char coords[11][4];
  std::ifstream file("csvs/dart15face.csv");

  for(int a = 0; a < 11; a++)
  {
    std::string line;
    std::getline (file,line);
    // if(!file.good())
    // {
    //   break;
    // }
    std::stringstream iss(line);

    for(int b = 0; b < 4; b++)
    {
      std::string value;
      std::getline(iss,value,',');
      // if(!iss.good())
      // {
      //   break;
      // }
      std::stringstream convertor(value);
      convertor >> coordinates[a][b];
      std::cout << coordinates[a][b] << std::endl;
      //std::cout << coordinates[a][b] << std::endl;
    }
  }

  int foo = 1;
  int i = 0;
  while(foo != 0){
    foo = coordinates[i][0];
    
  }


	truefaces = i;
	int intcoords [truefaces][4];

	// 2. Load the Strong Classifier in a structure called `Cascade'
	if( !cascade.load( cascade_name ) ){ printf("--(!)Error loading\n"); return -1; };

	// 3. Detect Faces and Display Result
	detectAndDisplay( frame );

	// 4. Save Result Image
	imwrite( "vj.jpg", frame );

	std::cout.precision(5);
  std::cout.setf(std::ios::fixed);
	//std::cout << f_one() << std::endl;

	return 0;
}

int toInt( char x ){ return x;}

float f_one(){
  int fp = truefaces - tp;
	int fn = detected - tp;
	//float precision = tp/(tp+fp);
  // float recall = tp/(tp+fn);
  // float f1 = 2 * (precision * recall) / (precision + recall);
  // return f1;
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
	//std::cout << faces.size() << std::endl;
  detected = faces.size();


  // 4. Draw box around faces found
	for( int i = 0; i < faces.size(); i++ )
	{
		rectangle(frame, Point(faces[i].x, faces[i].y), Point(faces[i].x + faces[i].width, faces[i].y + faces[i].height), Scalar( 0, 255, 0 ), 2);
  }

  //5. calculate true positives
  //std::cout << truefaces << std::endl;
  for(int i = 0; i < truefaces; i++)
	{
		int truematch = 0;
    //std::cout << coordinates[i][i] << std::endl;
		//iterate through true rectangles
		//for each, iterate through detected rectangles
		//if the detected rectangle is within the true rectangle, increment tp

		for( int j = 0; j < faces.size(); j++ )
		{
		   int tl[2] = {faces[j].x, faces[j].y};
			 int br[2] = {faces[j].x + faces[j].width, faces[j].y + faces[j].height};
			 //if: detected rectangle's top left point is within true rectangle's top left Point
			 //and detected rectangle's bottom right point is within true rectangle's bottom left point

       if(coordinates[i][0] >= tl[0]){
          std::cout << "match1" << std::endl;
          if(coordinates[i][1] <= tl[1]){
             std::cout << "match1" << std::endl;
             if(coordinates[i][2] >= br[0]){
                std::cout << "match3" << std::endl;
                if(coordinates[i][3] <= br[1]){
                   std::cout << "it worked" << std::endl;
                   truematch = 1;

      }}}}
    }

	if ( truematch == 1 ){
		tp++;
	}
	}
  	//std::cout << tp << std::endl;
}
