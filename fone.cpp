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
#include <cstdlib>

using namespace std;
using namespace cv;

int main( int argc, const char** argv )
{
  //fone = 2*(tp/(tp+fp))*(tp/(tp+fn))  /  (tp/(tp+fp))+(tp/(tp+fn))

  int tp = atoi(argv[0]);
  int fp = atoi(argv[1]);
  int fn = atoi(argv[2]);

  double precision = tp/(tp+fp);
  double recall = tp/(tp+fn);
  double f1 = 2 * (precision * recall) / (precision + recall);

  std::cout << f1 << std::endl;
  return 0;
}
