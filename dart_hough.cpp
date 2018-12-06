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
vector<Rect> viola_jones_detection( Mat frame );
Mat cannify_image( Mat frame );
Mat line_transform( Mat frame );
int*** circle_transform( Mat frame );
vector<Rect> hough_combine( Mat lines, int*** circles, vector<Rect> boards);
void draw_lines( Mat img ,  Mat lines );
void plot_shapes( Mat img , vector<Rect> final_recs);


/** Global variables */
String cascade_name = "dartcascade/cascade.xml";
CascadeClassifier cascade;

/** @function main */
int main( int argc, const char** argv ) {
	// 1. Read Input Image
	Mat frame0 = imread("images/dart1.jpg", CV_LOAD_IMAGE_COLOR);
	// 2. Load the Strong Classifier in a structure called `Cascade'
	if( !cascade.load( cascade_name ) ){ printf("--(!)Error loading\n"); return -1; };
	// 3. Blur and Canny edge detect the image
	Mat edgemapped = cannify_image( frame0 );
	// 4. Perform line Hough Transform on image
	Mat transformed = line_transform( edgemapped );
	// 5. Perform circle Hough Transform on the image & threshold
	// int*** thresh_circ = circle_transform( edgemapped );
	int*** thresh_circ;
	// 6. Threshold the line space
	Mat thresh;
	threshold(transformed, thresh, 100, 255, THRESH_BINARY);
	draw_lines(frame0, thresh);
	// 7. Perform Viola Jones detection on images
	vector<Rect> dartboards = viola_jones_detection( frame0 );
	// 8. Combine result with Hough detected shapes
	vector<Rect> final_boards = hough_combine( thresh, thresh_circ, dartboards);
	// 9. Plot final results on image
	plot_shapes( frame0 , final_boards);
}

/** @function detectAndDisplay */
vector<Rect> viola_jones_detection( Mat frame ) {
	vector<Rect> dartboards;
	Mat frame_gray;

	// 1. Prepare Image by turning it into Grayscale and normalising lighting
	cvtColor( frame, frame_gray, CV_BGR2GRAY );
	equalizeHist( frame_gray, frame_gray );

	// 2. Perform Viola-Jones Object Detection
	cascade.detectMultiScale( frame_gray, dartboards, 1.1, 1, 0|CV_HAAR_SCALE_IMAGE, Size(50, 50), Size(500,500) );

	return dartboards;
}

// Compare result of Viola Jones detection with Hough accumulators
vector<Rect> hough_combine( Mat lines, int*** circles, vector<Rect> boards) {
	const float pi = 3.14159;
	//Iterate through the boards
	for( int i = 0; i < boards.size(); i++ )	{
		//Find the centre of each board
		int centre_x = (boards[i].x + boards[i].width)/2;
		int centre_y = (boards[i].y + boards[i].height)/2;

		// Check to see if the pixel in the thresholded linespace has 5-10 spokes
		int spokes = 0;
		for (int j = 0; j < 10; j++){
			int theta = (j * 18) + 9;
			float t = (theta * pi)/180;
			int rho = (centre_x * cos(t)) + (centre_y * sin(t));
      int intensity = (int)lines.at<uchar>(theta, rho);
      if( intensity > 250) {
				spokes++;
			}
		}

		std::cout << spokes << std::endl;
		// Check to see if the pixel in the thresholded circlespace has 3-7 concentric circles
		// If yes to both add current rectangle to new_boards
		if(!(spokes >= 3 && spokes <=10)) {
			boards.erase(boards.begin() + i);
		}
	}
	return boards;
}

void draw_lines( Mat img ,  Mat lines ) {
	const float pi = 3.14159;
	for (int theta = 0; theta < lines.rows; theta++){
		for (int rho = 0; rho < lines.cols; rho++){
      int intensity = (int)lines.at<uchar>(theta, rho);
      if( intensity > 250) {
				float t = (theta * pi)/180;
				//Draw lines
				if((theta > 90 )){
	        Point pt1(rho / cos(t), 0);
	        Point pt2( (rho - img.rows * sin(t))/cos(t), img.rows);
	        line(img, pt1, pt2, Scalar(255), 1);
		    } else {
	        Point pt1(0, rho / sin(t));
	        Point pt2(img.cols, (rho - img.cols * cos(t))/sin(t));
	        line(img, pt1, pt2, Scalar(255), 1);
		    }

			}
		}
	}
	imwrite( "lines.jpg", img );
}

//Draw bounding boxes
void plot_shapes( Mat frame , vector<Rect> final_recs) {

	// Print number of Faces found
	std::cout << final_recs.size() << std::endl;

	//Draw rectangles
	for( int i = 0; i < final_recs.size(); i++ )	{
		rectangle(frame, Point(final_recs[i].x, final_recs[i].y), Point(final_recs[i].x + final_recs[i].width, final_recs[i].y + final_recs[i].height), Scalar( 0, 255, 0 ), 2);
	}

	//Save image
	imwrite( "VJ_HT.jpg", frame );
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
  int frame_height = frame.rows;
	int frame_width = frame.cols;

	//Generate empty circle space
  int x_dim = frame_height;
	int y_dim = frame_width;
	int r_dim = 50;
  int*** circle_space = malloc3dArray(x_dim, y_dim, r_dim);
	for (int i = 0; i < x_dim; i++) {
		for (int j = 0; j < y_dim; j++) {
			for (int k = 0; k < r_dim; k++) {
				circle_space[i][j][k] = 0;
			}
		}
	}

	//Create a new image with a black border buffer
	Mat padded_frame;
	copyMakeBorder( frame, padded_frame, r_dim, r_dim, r_dim, r_dim, BORDER_CONSTANT, 0 );
	imwrite("padded.jpg", padded_frame);

	//Iterate over each pixel
	for (int x = 0; x < frame_height; x++){
		for (int y = 0; y < frame_width; y++){

			//If the pixel is an edge
      int intensity = (int)frame.at<uchar>(x, y);
      if( intensity > 250) {

				//Iterate through each pixel in a 50 pixel radius
				for (int x0 = x; x0 < (x+(2*r_dim)); x0++){
					for (int y0 = y; y0 < (y+(2*r_dim)); y0++){

						// //If pixel is an edge
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

	//Threshold the circles accumulator
	//int max_val = 0;
	for (int i = 0; i < x_dim; i++) {
		for (int j = 0; j < y_dim; j++) {
			for (int k = 5; k < r_dim; k++) {
				// if (circle_space[i][j][k] > max_val) {
				// 	max_val = circle_space[i][j][k];
				// }
				if (circle_space[i][j][k] < 100) {
					circle_space[i][j][k] = 0;
				} else {
					circle_space[i][j][k] = 255;
				}
			}
		}
	}
	//std::cout << max_val << std::endl;

	return circle_space;
}
