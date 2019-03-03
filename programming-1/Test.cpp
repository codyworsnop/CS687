#include <fstream>
#include <iostream>
#include <stdlib.h>
#include <math.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#define CV_64F 6
using namespace cv;
using namespace std;

extern "C" void solve_system(int, int, float **, float *, float *);

    const int NumberOfImagesInDataSet = 10; 
    int i;
    const int m = 4;
    const int n = 3;
    const int normalized_x = 48; 
    const int normalized_y = 40; 

    float **a, *bx, *by, *x;
    float **a1; 

                                        //image 1       //image 2      //image 3       //image 4    //image 5
    float DataSetS1_XValues [10][m] = {{26,63,44,45}, {32,75,65,60}, {25,62,46,46}, {16,58,30,32}, {33,75,69,65},
                                        //image 6       //image 7       //image 8   //image 9   
                                       {14,55,27,31}, {24,62,44,44}, {27,64,49,50}, {30,68,54,52}, {33,75,63,57}};

    float DataSetS1_YValues [10][m] = {{50,50,88,70}, {44,44,67,86}, {48,48,72,88}, {45,42,63,83}, {44,42,62,84},                                      
                                       {46,44,65,85}, {44,43,58,80}, {46,43,64,84}, {37,35,46,73}, {51,50,76,91}};

    float DataSetS1_scaledXValues [m] = {10,24,18,18};
    float DataSetS1_scaledYValues [m] = {17,17,24,30};

void print2dMatrix(int m, int n, float ** matrix)
{
    for (int row = 0; row < m; row++)
    {
        for (int column = 0; column < n; column++)
        {
            cout << matrix[row][column] << " ";
        }

        cout << endl;
    }
}

void print1dMatrix(int m, float * matrix)
{
    for (int row = 0; row < m; row++)
    {
        cout << matrix[row] << " ";
    }

    cout << endl;
}

float* flatten(int m, int n, float **matrixToFlatten)
{
    float *flattened = new float[n*m]; 
    for (int row = 0; row < m; row++)
    {
        for (int column = 0; column < n; column++)
        {
            flattened[row * n + column] = matrixToFlatten[row][column];
        }
    }

    return flattened;
}

void LoadArrays(int imageIndex)
{
    //load A matrix
    for (int row = 0; row < m; row++)
    {
        a[row][0] = DataSetS1_XValues[imageIndex][row];
        a[row][1] = DataSetS1_YValues[imageIndex][row];
        a[row][2] = 1;

        a1[row][0] = DataSetS1_XValues[imageIndex][row];
        a1[row][1] = DataSetS1_YValues[imageIndex][row];
        a1[row][2] = 1;
        a1[row][3] = DataSetS1_scaledXValues[row];

        //p-hat x
        bx[row] = DataSetS1_scaledXValues[row];
        by[row] = DataSetS1_scaledYValues[row];
    }
}

double dot(Mat matrix1, Mat matrix2)
{
    double result; 
    result = matrix1.at<float>(0) * matrix2.at<float>(0) + matrix1.at<float>(1) * matrix2.at<float>(1) + matrix1.at<float>(2) * matrix2.at<float>(2);
    return result;
}

Mat ReadOriginalImage(int imageIndex)
{
    Mat image;
    char buffer[50];
    sprintf(buffer, "%s%d%s", "./Images/S1/", imageIndex + 1, ".pgm");
    image = imread(buffer, IMREAD_GRAYSCALE); // Read the file

    if (!image.data) // Check for invalid input
    {
        cout << "Could not open or find the image" << std::endl; 
    }

    return image;
}

Mat SolveSystemsWithSVD(Mat a, Mat b)
{
    Mat w, u, vt, w_inverse;
    SVD::compute(a, w, u, vt, SVD::FULL_UV);
    Mat w_full = (Mat_<float>(4, 3) << w.at<float>(0), 0, 0, 0, w.at<float>(1), 0, 0, 0, w.at<float>(2), 0, 0, 0);

    invert(w_full, w_inverse, DECOMP_SVD);
    Mat aplus = vt.t() * w_inverse * u.t();
    Mat solution = aplus * b;

    return solution;
}

void SaveImage(Mat image, int name)
{
    char buffer[50];
    sprintf(buffer, "%s%d%s", "./Images/S1/", name, "_normalized.pgm\0");
    imwrite(buffer, image);
}

Mat *RemapImage(Mat image, Mat x_solution, Mat y_solution)
{
    Mat *mappedImage = new Mat(normalized_x, normalized_y, 0, cvScalar(0.)); 
    int x = 0; 
    int y = 0;
    float new_x = 0; 
    float new_y = 0;
    Mat *coordinates; 
   
    for (int row = 0; row < image.rows; row++)
    {
        for (int column = 0; column < image.cols; column++)
        {
            float *imageCoordinates = new float[3] {(float)row, (float)column, 1}; 
            coordinates = new Mat(1, 3, 0, imageCoordinates); 
            new_x = dot(*coordinates, x_solution); 
            new_y = dot(*coordinates, y_solution); 

            //normalize new x and y values 
            if (new_x < 0)
            {
                new_x = new_x - new_x; 
            }
            else if (new_x > normalized_x)
            {
                new_x = new_x - (normalized_x - new_x); 
            }

            if (new_y < 0)
            {
                new_y = new_y - new_y; 
            }
            else if (new_y > normalized_y)
            {
                new_y = new_y - (normalized_y - new_y); 
            }

            new_x = (int)new_x;
            new_y = (int)new_y;

            mappedImage->at<uchar>(new_x, new_y) = image.at<uchar>(row, column); 

            delete(imageCoordinates);
            delete(coordinates);
        }
    }


   // namedWindow( "Display window", WINDOW_AUTOSIZE);// Create a window for display.
   // imshow( "Display window", image); 
   // namedWindow( "new", WINDOW_AUTOSIZE );// Create a window for display.
  //  imshow( "new", *mappedImage); 
  //  waitKey(0); // Wait for a keystroke in the windowd

    return mappedImage;

}

int main(int argc, char *argv[])
{

    //initialize a, x, b
    a = new float *[m + 1];
    for (i = 0; i < m + 1; i++)
        a[i] = new float[n + 1];
    a1 = new float *[m + 1];
    for (i = 0; i < m + 1; i++)
        a1[i] = new float[n + 2];

    x = new float[n + 1];
    bx = new float[m + 1];
    by = new float[m + 1];

    for (int imageIndex = 0; imageIndex < NumberOfImagesInDataSet; imageIndex++)
    {
        LoadArrays(imageIndex);

        //flatten
        float* a_flat = flatten(m, n, a); 
        Mat a_mat = Mat(m, n, 5, a_flat);
        Mat bx_mat = Mat(m, 1, 5, bx);
        Mat by_mat = Mat(m, 1, 5, by);

        //solve for p-hat x using SVD
        Mat x_solution = SolveSystemsWithSVD(a_mat, bx_mat); 
        Mat y_solution = SolveSystemsWithSVD(a_mat, by_mat);


        Mat image = ReadOriginalImage(imageIndex); 
        Mat *remappedImage = RemapImage(image, x_solution, y_solution);

        SaveImage(*remappedImage, imageIndex + 1);
    }

    waitKey(0); // Wait for a keystroke in the window
    return 0;
}
