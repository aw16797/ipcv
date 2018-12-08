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
#include <string>
#include <stdlib.h>
#include <fstream>
#include <sstream>

using namespace std;
using namespace cv;

/** Function Headers */

//Functions to calculate images needed
void GaussianBlur(cv::Mat &input,int size,cv::Mat &blurredOutput);
void sobelX(cv::Mat &output);
void sobelY(cv::Mat &output);
void sobel(cv::Mat &input,cv::Mat &OutputX,cv::Mat &OutputY,cv::Mat &OutputM,cv::Mat &OutputD,cv::Mat &OutputD2);
//Hough transforms
// int** line_transform( Mat edges, Mat direction );
Mat circle_transform( Mat edges, Mat direction, Mat img, int max_radius);
//Detection
vector<Rect> viola_jones_detection( Mat frame );
vector<Rect> hough_combine( Mat circles, vector<Rect> boards, int radius);
//Draw output
void draw_lines( Mat img ,  Mat lines );
void plot_shapes( Mat img , vector<Rect> final_recs);
float f_one();
//char* findCSV(char* fname);

/** Global variables */
String cascade_name = "dartcascade/cascade.xml";
CascadeClassifier cascade;
int tp,trueboards,detected;
int coordinates[3][4];
int threshcircle = 145;
int maxcircles = 6;

/** @function main */
int main( int argc, char** argv ) {
	// 1. Read Input Image
  // char* imageName = argv[1];
  // Mat frame0;
  // frame0 = imread( imageName, CV_LOAD_IMAGE_COLOR);
	//
  // if( argc != 2 || !frame0.data )
  // {
  //   printf( " No image data \n " );
  //   return -1;
  // }
	Mat frame0 = imread("images/dart15.jpg", CV_LOAD_IMAGE_COLOR);

	char coords[3][4];
  std::ifstream file("csvs/dart15board.csv");

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
      //std::cout << coordinates[a][b] << std::endl;
    }
  }

  int foo = 1;
  int i = 0;
  while(foo != 0){
    foo = coordinates[i][0];
    if(foo != 0){i++;}
    //std::cout << i << std::endl;

  }
	trueboards = i;

	// 2. Load the Strong Classifier in a structure called `Cascade'
	if( !cascade.load( cascade_name ) ){ printf("--(!)Error loading\n"); return -1; };

	// 3. Blur and edge detect the image
	//greyscale conversion
	Mat gray_image;
  cvtColor( frame0, gray_image, CV_BGR2GRAY );
  // //blur gray image
  Mat blurgray_image;
  GaussianBlur(gray_image, 5, blurgray_image);
  //define new images to hold convolved image
 	Mat imageX(blurgray_image.size(), CV_32FC1);
	Mat imageY(blurgray_image.size(), CV_32FC1);
	Mat image_mag(blurgray_image.size(), CV_32FC1);
	Mat image_rad(blurgray_image.size(), CV_32FC1);
	Mat image_dir(blurgray_image.size(), CV_32FC1);
	sobel(blurgray_image,imageX,imageY,image_mag,image_rad,image_dir);
	//thresholding
	Mat image_edges;
	threshold(image_mag, image_edges, 128, 255, THRESH_BINARY);
	imwrite("mt.jpg", image_edges);

	// 4. Perform line Hough Transform on image
	// int** transformed = line_transform( image_edges, image_rad );

	// 5. Perform circle Hough Transform on the image & threshold
	int max_radius = 100;
	Mat thresh_circ = circle_transform( image_edges, image_rad, frame0, max_radius );

	// 6. Perform Viola Jones detection on images
	vector<Rect> dartboards = viola_jones_detection( frame0 );

	// 7. Combine result with Hough detected shapes
	vector<Rect> final_boards = hough_combine( thresh_circ, dartboards, max_radius);

	// 8. Plot final results on image
	plot_shapes( frame0 , final_boards);

	std::cout.precision(5);
  std::cout.setf(std::ios::fixed);
	std::cout << f_one() << std::endl;

	return 0;
}

float f_one(){
  int fn = trueboards - tp;
	int fp = detected - tp;

  std::cout << "TP" << tp << std::endl;
  std::cout << "true" << trueboards << std::endl;
  std::cout << "detected" << detected << std::endl;
  std::cout << "FP" << fp << std::endl;
  std::cout << "FN" << fn << std::endl;

	float precision = (float)tp/((float)tp+(float)fp);
  std::cout << "precision " << precision << std::endl;

  float recall = (float)tp/((float)tp+(float)fn);
  std::cout << "recall " << recall << std::endl;

  float f1 = 2 * (precision * recall) / (precision + recall);
  return f1;
}

char* findCSV( char* fname ){
	char* s = fname;
	char* csvname;
	if(strcmp(s,"dart0.jpg")==0){
		csvname = (char*)"csvs/dart0board.csv";
	}
	else if(strcmp(s,"dart1.jpg")==0){
		csvname = (char*)"csvs/dart1board.csv";
	}
	else if(strcmp(s,"dart2.jpg")==0){
		csvname = (char*)"csvs/dart2board.csv";
	}
	else if(strcmp(s,"dart3.jpg")==0){
		csvname = (char*)"csvs/dart3board.csv";
	}
	else if(strcmp(s,"dart4.jpg")==0){
		csvname = (char*)"csvs/dart4board.csv";
	}
	else if(strcmp(s,"dart5.jpg")==0){
		csvname = (char*)"csvs/dart5board.csv";
	}
	else if(strcmp(s,"dart6.jpg")==0){
		csvname = (char*)"csvs/dart6board.csv";
	}
	else if(strcmp(s,"dart7.jpg")==0){
		csvname = (char*)"csvs/dart7board.csv";
	}
	else if(strcmp(s,"dart8.jpg")==0){
		csvname = (char*)"csvs/dart8board.csv";
	}
	else if(strcmp(s,"dart9.jpg")==0){
		csvname = (char*)"csvs/dart9board.csv";
	}
	else if(strcmp(s,"dart10.jpg")==0){
		csvname = (char*)"csvs/dart10board.csv";
	}
	else if(strcmp(s,"dart11.jpg")==0){
		csvname = (char*)"csvs/dart11board.csv";
	}
	else if(strcmp(s,"dart12.jpg")==0){
		csvname = (char*)"csvs/dart12board.csv";
	}
	else if(strcmp(s,"dart13.jpg")==0){
		csvname = (char*)"csvs/dart13board.csv";
	}
	else if(strcmp(s,"dart14.jpg")==0){
		csvname = (char*)"csvs/dart14board.csv";
	}
	else if(strcmp(s,"dart15.jpg")==0){
		csvname = (char*)"csvs/dart15board.csv";
	}
	else{
		std::cout << "no csv" << std::endl;
	}
	return csvname;

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
	imwrite( "xd.jpg", OutputX );
	imwrite( "yd.jpg", OutputY );
	imwrite( "mag.jpg", OutputM );
	imwrite( "dir.jpg", OutputD2 );
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

//Perform circle hough transform on the edge image
Mat circle_transform( Mat edges, Mat direction, Mat img, int max_radius) {
  int frame_height = edges.rows;
	int frame_width = edges.cols;
	Mat hough_visual = Mat::zeros(frame_height,frame_width, CV_8UC1);
	Mat hough_thresh = Mat::zeros(frame_height,frame_width, CV_8UC1);

	//Generate empty circle space
	int r_dim = max_radius;
	int x_dim = frame_width + (2*r_dim);
  int y_dim = frame_height + (2*r_dim);
  int*** circle_space = malloc3dArray(x_dim, y_dim, r_dim);
	for (int i = 0; i < x_dim; i++) {
		for (int j = 0; j < y_dim; j++) {
			for (int k = 0; k < r_dim; k++) {
				circle_space[i][j][k] = 0;
			}
		}
	}

	//Iterate over each pixel
	for (int y = 0; y < frame_height; y++){
		for (int x = 0; x < frame_width; x++){

			//If the pixel is an edge
      float intensity = edges.at<float>(y, x);
      if( intensity > 250) {

				//Iterate over potential radius
				for (int r = 10; r < r_dim; r++){
					float dir = direction.at<float>(y, x);
					int x0p = x + r*cos(dir) + r_dim;
					int x0n = x - r*cos(dir) + r_dim;
					int y0p = y + r*sin(dir) + r_dim;
					int y0n = y - r*sin(dir) + r_dim;

					//Add 1 to circle space
					circle_space[x0p][y0p][r] = circle_space[x0p][y0p][r] + 1;
					circle_space[x0p][y0n][r] = circle_space[x0p][y0n][r] + 1;
					circle_space[x0n][y0p][r] = circle_space[x0n][y0p][r] + 1;
					circle_space[x0n][y0n][r] = circle_space[x0n][y0n][r] + 1;

				}
      }
	  }
	}
	//Create visual Hough space
	for (int y = 0; y < frame_height; y++){
		for (int x = 0; x < frame_width; x++){
			int val = 0;
			for (int r = 10; r < r_dim; r++) {
				val = val + circle_space[x+r_dim][y+r_dim][r];
			}
			hough_visual.at<uchar>(y, x) = (uchar) val;
		}
	}

	//Threshold the cumulative space
	for (int y = 0; y < frame_height; y++){
		for (int x = 0; x < frame_width; x++){
			threshold(hough_visual, hough_thresh, threshcircle, 255, THRESH_BINARY);
		}
	}

	//ALTERNATIVE METHOD
	// //Threshold the circles accumulator
	// for (int i = 0; i < x_dim; i++) {
	// 	for (int j = 0; j < y_dim; j++) {
	// 		for (int r = 10; r < r_dim; r++) {
	// 			if (circle_space[i][j][r] < 40) {
	// 				circle_space[i][j][r] = 0;
	// 			} else {
	// 				circle_space[i][j][r] = 255;
	// 				circle(img, Point(i-r_dim,j-r_dim), r, Scalar( 0, 255, 0 ), 2);
	// 			}
	// 		}
	// 	}
	// }

	// //Create visual Hough space after thresholding
	// for (int y = 0; y < frame_height; y++){
	// 	for (int x = 0; x < frame_width; x++){
	// 		int val = 0;
	// 		for (int r = 10; r < r_dim; r++) {
	// 			val = val + circle_space[x+r_dim][y+r_dim][r];
	// 		}
	// 		hough_thresh.at<uchar>(y, x) = (uchar) val;
	// 	}
	// }


	// imwrite( "circles.jpg", img );
	imwrite( "hough_visual.jpg", hough_visual );
	imwrite( "hough_thresh.jpg", hough_thresh );

	return hough_thresh;
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
vector<Rect> hough_combine( Mat circles, vector<Rect> boards, int radius) {
	imwrite( "inside.jpg", circles );
	vector<Rect> final_boards;
	// std::cout << "No of boards:"  << std::endl;
	// std::cout << boards.size() << std::endl;
	//Iterate through the boards
	for( int i = 0; i < boards.size(); i++ )	{
		//Iterate over the pixels in around the centre of the box
		int circles_detected = 0;
		// int centre_x = (boards[i].x + boards[i].width)/2;
		// int centre_y = (boards[i].y + boards[i].height)/2;
		// for (int x0 = centre_x-20; x0 < centre_x+20; x0++) {
		// 	for (int y0 = centre_y-20; y0 < centre_y-20; y0++) {
		for (int x0 = boards[i].x; x0 < boards[i].x + boards[i].width; x0++) {
			for (int y0 = boards[i].y; y0 < boards[i].y + boards[i].height; y0++) {
				//If number of circles better than current max change max
				int intensity = (int) circles.at<uchar>(y0, x0);
				if (intensity > 250) {
					circles_detected++;
				}
			}
		}
		// std::cout << circles_detected << std::endl;
		if(circles_detected > maxcircles) {
			final_boards.push_back(boards[i]);
		}
	}
	// std::cout << loopsize << std::endl;
  detected = final_boards.size();

	for(int i = 0; i < trueboards; i++)
	{
		int truematch = 0;
		for( int j = 0; j < final_boards.size(); j++ )
		{
		   int tl[2] = {final_boards[j].x, final_boards[j].y};
			 int br[2] = {final_boards[j].x + final_boards[j].width, final_boards[j].y + final_boards[j].height};
			 //if: detected rectangle's top left point is within true rectangle's top left Point
			 //and detected rectangle's bottom right point is within true rectangle's bottom left point
       if(coordinates[i][0] <= tl[0]){
          //std::cout << "match1" << std::endl;
          if(coordinates[i][1] <= tl[1]){
             //std::cout << "match2" << std::endl;
             if(coordinates[i][2] >= br[0]){
                //std::cout << "match3" << std::endl;
                if(coordinates[i][3] >= br[1]){
                   //std::cout << "it worked" << std::endl;
                   truematch = 1;
								}
						 }
					}
			 }
    }
	  if ( truematch == 1 ){
	  	tp++;
  	}
	}

	return final_boards;
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

//Perform line hough transform on the edge image
// Mat line_transform( Mat edges, Mat direction ) {
//   const float pi = 3.14159;
//   int frame_height = edges.rows;
// 	int frame_width = edges.cols;
//
// 	//Generate empty Hough Space
//   int theta_max = 360;
// 	float d_theta = pi;
//   int rho_max = round(sqrt(pow(frame_width, 2) + pow(frame_height, 2)));
//   int** line_space = malloc2dArray(theta_max, d_theta);
// 	for (int i = 0; i < x_dim; i++) {
// 		for (int j = 0; j < y_dim; j++) {
// 			circle_space[i][j] = 0;
// 		}
// 	}
//
// 	//Iterate over each pixel
// 	for (int x = 0; x < frame_height; x++){
// 		for (int y = 0; y < frame_width; y++){
//       //If the pixel is an edge
//       float intensity = edges.at<float>(x, y);
//       if( intensity > 250) {
// 				float dir = direction.at<float>(x, y);
//
//         for(float t = (dir-d_theta); t < (dir+d_theta); t++){
//           int rho = (x * cos(t)) + (y * sin(t));
//
//           int current = (int)hough_space.at<uchar>(theta, rho);
//           int new_current = current + 1;
//           hough_space.at<uchar>(theta, rho) = (uchar)new_current;
//   			}
//       }
// 	  }
// 	}
//   imwrite("hough.jpg", hough_space);
// 	return hough_space;
// }

void draw_lines( Mat img ,  Mat lines ) {
	const float pi = 3.14159;
	for (int theta = 0; theta < lines.rows; theta++){
		for (int rho = 0; rho < lines.cols; rho++){
      int intensity = (int)lines.at<uchar>(theta, rho);
      if( intensity > 250) {
				float t = (theta * pi)/180;
				//Draw lines
     		Point pt1, pt2;
		    double a = cos(theta), b = sin(theta);
		    double x0 = a*rho, y0 = b*rho;
		    pt1.x = cvRound(x0 + 1000*(-b));
		    pt1.y = cvRound(y0 + 1000*(a));
	     	pt2.x = cvRound(x0 - 1000*(-b));
	     	pt2.y = cvRound(y0 - 1000*(a));
				line( img, pt1, pt2, Scalar(0,0,255), 1, CV_AA);

			}
		}
	}
	imwrite( "lines.jpg", img );
}
