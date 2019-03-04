#include <fstream>
#include <iostream>
#include <stdlib.h>
#include <math.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>
#define CV_64F 6
using namespace cv;
using namespace std;

extern "C" void solve_system(int, int, float **, float *, float *);

    const int NumberOfImagesInDataSet = 10; 

    const int m = 4;
    const int n = 3;
    const int normalized_x = 48; 
    const int normalized_y = 40; 

    //problem 1 variables
    float **a, *bx, *by, *x;

    //problem 2 variables
    float **a2, *x2; 
    Mat *b2; 

                                        //image 1       //image 2      //image 3       //image 4    //image 5
    float DataSetS1_XValues [10][m] = {{26,63,44,45}, {32,75,65,60}, {25,62,46,46}, {16,58,30,32}, {33,75,69,65},
                                        //image 6       //image 7       //image 8   //image 9        //image 10
                                       {14,55,27,31}, {24,62,44,44}, {27,64,49,50}, {30,68,54,52}, {33,75,63,57}};

                                        //image 1       //image 2     //image 3       //image 4      //image 5
    float DataSetS1_YValues [10][m] = {{50,50,88,70}, {44,44,67,86}, {48,48,72,88}, {45,42,63,83}, {44,42,62,84},  
                                        //image 6       //image 7      //image 8      //image 9      //image 10                                    
                                       {46,44,65,85}, {44,43,58,80}, {46,43,64,84}, {37,35,46,73}, {51,50,76,91}};

    float DataSetS1_scaledXValues [m] = {10,38,24,24};
    float DataSetS1_scaledYValues [m] = {10,10,20,35};

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

void DisplayImage(Mat image)
{
    namedWindow( "Display window", WINDOW_AUTOSIZE );// Create a window for display.
    imshow( "Display window", image ); 
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

Mat ReadOriginalImage(int imageIndex, bool readNormalizedImages)
{
    Mat image;
    char buffer[50];
    if (readNormalizedImages)
    {  
        sprintf(buffer, "%s%d%s", "./Images/S1/", imageIndex + 1, "_normalized.pgm");
    }
    else
    {        
        sprintf(buffer, "%s%d%s", "./Images/S1/", imageIndex + 1, ".pgm");          
    }

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
    cout << w << endl;
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

Mat RemapImage(Mat image, Mat x_solution, Mat y_solution)
{
    Mat mappedImage = Mat(normalized_x, normalized_y, 0, cvScalar(0)); 
    int x = 0; 
    int y = 0;
    int new_x = 0; 
    int new_y = 0;
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
            if (new_x >= 0 && new_x < normalized_x && new_y >= 0 && new_y < normalized_y)
            {
                new_x = (int)new_x;
                new_y = (int)new_y;

                mappedImage.at<uchar>(new_x, new_y) = image.at<uchar>(row, column);
            }

            delete(imageCoordinates);
            delete(coordinates);
        }
    } 

    mappedImage.reshape(normalized_y, normalized_x); 
    return mappedImage;
}

void Problem1()
{
    //initialize a, x, b
    a = new float *[m];
    for (int i = 0; i < m; i++)
        a[i] = new float[n];

    x = new float[n];
    bx = new float[m];
    by = new float[m];

    for (int imageIndex = 0; imageIndex < NumberOfImagesInDataSet; imageIndex++)
    {
        LoadArrays(imageIndex);

        //flatten
        float* a_flat = flatten(m, n, a); 
        Mat a_mat = Mat(m, n, 5, a_flat);
        Mat bx_mat = Mat(m, 1, 5, bx);
        Mat by_mat = Mat(m, 1, 5, by);

        //solve for p-hat x, p-hat y using SVD
        Mat x_solution = SolveSystemsWithSVD(a_mat, bx_mat); 
        Mat y_solution = SolveSystemsWithSVD(a_mat, by_mat);
        const Mat tm = (Mat_<float>(2, 3) << x_solution.at<float>(0), x_solution.at<float>(1), x_solution.at<float>(2), y_solution.at<float>(0), y_solution.at<float>(1), y_solution.at<float>(2));

        Mat image = ReadOriginalImage(imageIndex, false); 
        Mat remappedImage = RemapImage(image, x_solution, y_solution);

        SaveImage(remappedImage, imageIndex + 1);
    }
}

void LoadAMatrix(Mat image)
{
    for (int row = 0; row < image.rows; row++)
    {
        for (int col = 0; col < image.cols; col++)
        {
            a2[row][0] = row;
            a2[row][1] = col;                          
            a2[row][2] = row * col; 
            a2[row][3] = 1; 
        }
    }
}

void Problem2()
{
    int mnNormSIze = normalized_x * normalized_y; 

    a2 = new float *[mnNormSIze];
    for (int i = 0; i < mnNormSIze; i++)
        a2[i] = new float[4];

    for (int imageIndex = 0; imageIndex < NumberOfImagesInDataSet; imageIndex++)
    {
        Mat image = ReadOriginalImage(imageIndex, true);

        //b matrix - a matrix of all the pixel intensities
        b2 = new Mat(image.rows, image.cols, 0, &image);
        b2->reshape(normalized_x * normalized_y, 1);

        //x matrix - a matrix to be solved for (4x1?), a, b, c, d
        x2 = new float[m];

        //A matrix - x, y, xy, 1
        LoadAMatrix(image); 
        
        float *a2_flat = flatten(mnNormSIze, 4, a2);

        Mat a2_mat = Mat(mnNormSIze, 4, 5, a2_flat); 
        Mat b2_mat = Mat(mnNormSIze, 1, 5, b2);

        Mat wut = SolveSystemsWithSVD(a2_mat, b2_mat); 
    }
}

int main(int argc, char *argv[])
{
    //problem 1 using SVD to solve the affine transformations
    //Problem1();     

    //problem 2 using SVD to solve the over constrained systems of pixel intensities 
    Problem2(); 

    //wait for keystroke so user knows program is done
    waitKey(0);

    return 0;
}