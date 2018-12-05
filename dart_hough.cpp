/////////////////////////////////////////////////////////////////////////////
//
// COMS30121 - dart.cpp
//
/////////////////////////////////////////////////////////////////////////////

// header inclusion
// header inclusion
#include <stdio.h>
#include <math.h>
#include <3darray.c>
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
Mat line_transform( Mat frame );
int*** circle_transform( Mat frame );
Mat plot_lines( Mat img , Mat line_space);


/** Global variables */
String cascade_name = "dartcascade/cascade.xml";
CascadeClassifier cascade;

/** @function main */
int main( int argc, const char** argv ) {
	// 1. Read Input Image
	Mat frame0 = imread("images/dart0.jpg", CV_LOAD_IMAGE_COLOR);
	// 2. Load the Strong Classifier in a structure called `Cascade'
	if( !cascade.load( cascade_name ) ){ printf("--(!)Error loading\n"); return -1; };
	// 3. Blur and Canny edge detect the image
	Mat edgemapped = cannify_image(frame0);
	// 4. Perform line Hough Transform on image
	Mat transformed = line_transform(edgemapped);
	// 5. Perform circle Hough Transform on the image
	int*** circle_transformed = circle_transform(edgemapped);
	// 6. Pick out most luminous points
	Mat thresh;
	threshold(transformed, thresh, 100, 255, THRESH_BINARY);
	imwrite("line_thresh.jpg", thresh);
	// 7. Plot lines on image
	//Mat plotted_lines = plot_lines(frame0, thresh1);
	// 8. Plot circles on image
}

/** @function detectAndDisplay */
void detectAndDisplay( Mat frame ) {
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
	for( int i = 0; i < faces.size(); i++ )	{
		rectangle(frame, Point(faces[i].x, faces[i].y), Point(faces[i].x + faces[i].width, faces[i].y + faces[i].height), Scalar( 0, 255, 0 ), 2);
	}
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

//Perform lline hough transform on the edge image
Mat line_transform( Mat frame ) {
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

        for(int theta = 0; theta < theta_max; theta++){
          float t = (theta * pi)/180;
          int rho = (x * cos(t)) + (y * sin(t));

          int current = (int)hough_space.at<uchar>(theta, rho);
          int new_current = current + 1;
          hough_space.at<uchar>(theta, rho) = (uchar)new_current;
  			}
      }
	  }
	}
  imwrite("hough.jpg", hough_space);
	return hough_space;
}

//Perform circle hough transform on the edge image
int*** circle_transform( Mat frame ) {
  const float pi = 3.14159;
  int frame_height = frame.rows;
	int frame_width = frame.cols;

	//Generate empty circle space
  int x_dim = frame_height;
	int y_dim = frame_width;
	int r_dim = 50;
  int*** circle_space = malloc3dArray(x_dim, y_dim, r_dim);

	//Create a new image with a black border buffer
	Mat padded_frame;
	copyMakeBorder( frame, padded_frame, r_dim, r_dim, r_dim, r_dim, BORDER_CONSTANT, 0 );

	//Iterate over each pixel
	for (int x = 0; x < frame_height; x++){
		for (int y = 0; y < frame_width; y++){

			//If the pixel is an edge
      int intensity = (int)frame.at<uchar>(x, y);
      if( intensity > 250) {

				//Iterate through each pixel in a 50 pixel radius
				for (int x0 = x; x0 < (x+2*r_dim); x0++){
					for (int y0 = y; y0 < (y+2*r_dim); y0++){

						//If pixel is an edge
						int intensity2 = (int)padded_frame.at<uchar>(x0, y0);
			      if( intensity2 > 250) {
							int r = sqrt(pow((x - x0), 2) + pow((y - y0), 2));

							//Add 1 to circle space
							int current = circle_space[x][y][r];
							int new_current = current + 1;
							circle_space[x][y][r] = new_current;
						}
					}
				}
      }
	  }
	}
	return circle_space;
}

// //Plot the lines on the image
// Mat plot_lines( Mat img , Mat line_space) {
// 	//Iterate over the thresholded line space
// 	for (int theta = 0; theta < line_space.rows; theta++){
// 		for (int rho = 0; rho < line_space.cols; rho++){
//       int intensity = (int)line_space.at<uchar>(theta, rho);
//       if( intensity > 99) {
// 				//Where it finds a bright pixel, convert polar to coordinates
// 				float t = (theta * pi)/180;
// 				//(X0, Y0) = (rho * cos(theta), rho * sin(theta))
// 				int x0 = rho * cos(t);
// 				int y0 = rho * sin(t);
// 				//(dx, dy) = ( -sin(theta), cos(theta))
// 				dx = -sin(t);
// 				dy = cos(t);
//
// 			}
// 		}
// 	}
//
// 	return img;
// }
