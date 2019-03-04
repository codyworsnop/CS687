/**
 * @file Test.cpp
 * @author cody worsnop
 * @brief Programming assignment 1 for CS 685 
 * @version 0.1
 * @date 2019-03-04
 * 
 * @copyright Copyright (c) 2019
 * 
 */

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
const int NumberOfDataSets = 2;

const int m = 4;
const int n = 3;
const int normalized_x = 48;
const int normalized_y = 40;

float (*CurrentDataSet_Y)[m];
float (*CurrentDataSet_X)[m];
float *CurrentScaled_X;
float *CurrentScaled_Y;

//problem 1 variables
float **a, *bx, *by, *x;

//problem 2 variables
float **a2, *x2;
Mat *b2;

//these are the mapped data sets. They are broken up by x and y values. 
float DataSetS1_YValues[10][m] = {{26, 63, 44, 45},
                                  {32, 75, 65, 60},
                                  {25, 62, 46, 46},
                                  {16, 58, 30, 32},
                                  {33, 75, 69, 65},
                                  {14, 55, 27, 31},
                                  {24, 62, 44, 44},
                                  {27, 64, 49, 50},
                                  {30, 68, 54, 52},
                                  {33, 75, 63, 57}};

float DataSetS1_XValues[10][m] = {{50, 50, 88, 70},
                                  {44, 44, 67, 86},
                                  {48, 48, 72, 88},
                                  {45, 42, 63, 83},
                                  {44, 42, 62, 84},
                                  {46, 44, 65, 85},
                                  {44, 43, 58, 80},
                                  {46, 43, 64, 84},
                                  {37, 35, 46, 73},
                                  {51, 50, 76, 91}};

float DataSetS2_YValues[10][m] = {{25, 60, 41, 40},
                                  {35, 68, 54, 52},
                                  {21, 56, 33, 33},
                                  {32, 65, 51, 49},
                                  {38, 73, 61, 58},
                                  {16, 51, 26, 29},
                                  {29, 62, 45, 44},
                                  {21, 58, 35, 35},
                                  {34, 37, 51, 50},
                                  {45, 77, 71, 66}};

float DataSetS2_XValues[10][m] = {{51, 54, 67, 88},
                                  {51, 52, 66, 67},
                                  {51, 53, 67, 87},
                                  {53, 52, 68, 88},
                                  {52, 52, 68, 88},
                                  {51, 53, 67, 88},
                                  {52, 53, 69, 89},
                                  {50, 52, 66, 88},
                                  {52, 53, 68, 90},
                                  {54, 53, 67, 88}};

float DataSetS1_scaledYValues[m] = {10, 38, 24, 24};
float DataSetS1_scaledXValues[m] = {10, 10, 20, 35};
float DataSetS2_scaledYValues[m] = {10, 38, 24, 24};
float DataSetS2_scaledXValues[m] = {10, 10, 20, 35};

/**
 * @brief Prints a 2d matrix 
 * 
 * @param m the row count of the matrix to print
 * @param n the column count of the matrix to print
 * @param matrix the matrix to print 
 */
void print2dMatrix(int m, int n, float **matrix)
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

/**
 * @brief Prints a 1d matrix
 * 
 * @param m the row count in the matrix to print
 * @param matrix the matrix to print
 */
void print1dMatrix(int m, float *matrix)
{
    for (int row = 0; row < m; row++)
    {
        cout << matrix[row] << " ";
    }

    cout << endl;
}

/**
 * @brief Displays a Mat image to the screen
 * 
 * @param image the image to display
 */
void DisplayImage(Mat image)
{
    namedWindow("Display window", WINDOW_AUTOSIZE); // Create a window for display.
    imshow("Display window", image);
}

/**
 * @brief flattens a 2d float vector into a single vector. Used for initializing Mat with a float data source
 * 
 * @param m the row dimension
 * @param n the column dimension
 * @param matrixToFlatten the matrix to flatten
 * @return float* a flattened matrix
 */
float *flatten(int m, int n, float **matrixToFlatten)
{
    float *flattened = new float[n * m];
    for (int row = 0; row < m; row++)
    {
        for (int column = 0; column < n; column++)
        {
            flattened[row * n + column] = matrixToFlatten[row][column];
        }
    }

    return flattened;
}

/**
 * @brief Loads the a, p-hat x and p-hat y matrix for problem 1
 * 
 * @param imageIndex the index of the image we are working on 
 */
void LoadArrays(int imageIndex)
{
    //load A matrix
    for (int row = 0; row < m; row++)
    {
        a[row][0] = CurrentDataSet_X[imageIndex][row];
        a[row][1] = CurrentDataSet_Y[imageIndex][row];
        a[row][2] = 1;

        //p-hat x
        bx[row] = CurrentScaled_X[row];
        by[row] = CurrentScaled_Y[row];
    }
}

/**
 * @brief finds the dot product between to 1x3 and 3x1 matrices. Built as I was having issues with Mat.dot
 * 
 * @param matrix1 the first matrix to dot product
 * @param matrix2 the matrix the first is dotted with
 * @return double a scalar result of the dot product operation
 */
double dot(Mat matrix1, Mat matrix2)
{
    double result;
    result = matrix1.at<float>(0) * matrix2.at<float>(0) + matrix1.at<float>(1) * matrix2.at<float>(1) + matrix1.at<float>(2) * matrix2.at<float>(2);
    return result;
}

/**
 * @brief the original image for computation in the program
 * 
 * @param path the path to the image to read
 * @return Mat the image in mat form
 */
Mat ReadOriginalImage(char *path)
{
    Mat image;
    image = imread(path, IMREAD_GRAYSCALE); // Read the file

    if (!image.data) // Check for invalid input
    {
        cout << "Could not open or find the image" << std::endl;
    }

    return image;
}

/**
 * @brief Fills the diagonal matrix from an SVD result. Built as SVD::Compute would return a mx1 matrix 
 * and the math computation needed the full diagonal 
 * @param w the matrix to build the diagonal with
 * @param row the count of rows
 * @param col the count of columns
 * @return Mat the built diagonal
 */
Mat FillDiagonal(Mat w, int row, int col)
{
    Mat w_full = Mat(row, col, 5, cvScalar(0));

    for (int i = 0; i < col; i++)
    {
        w_full.at<float>(i, i) = w.at<float>(i);
    }

    return w_full;
}

/**
 * @brief Solves a over-determined system of linear equation using singular value decomposition.
 * 
 * @param a The A matrix with the x y 1 values
 * @param b The b matrix with the predetermined normalized locations
 * @param row the rows in the diagonal. Used to compute the eigenvector diagonal.
 * @param col The cols in the diagonal. Used to compute the eigenvector diagonal.
 * @return Mat A solution to the system 
 */
Mat SolveSystemsWithSVD(Mat a, Mat b, int row, int col)
{
    Mat w, u, vt, w_inverse;
    SVD::compute(a, w, u, vt, SVD::FULL_UV);

    Mat w_full = FillDiagonal(w, row, col);

    invert(w_full, w_inverse, DECOMP_SVD);
    Mat aplus = vt.t() * w_inverse * u.t();
    Mat solution = aplus * b;

    return solution;
}

/**
 * @brief Saves an image to a predetermined location
 * 
 * @param image the image to save
 * @param path the path to save it to 
 */
void SaveImage(Mat image, char *path)
{
    imwrite(path, image);
}

/**
 * @brief Remaps an image given a transformation
 * 
 * @param image the original image to remap
 * @param x_solution the solution to the x transformation
 * @param y_solution the solution to the y transformation
 * @return Mat a remapped image
 */
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
            float *imageCoordinates = new float[3]{(float)row, (float)column, 1};
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

            delete (imageCoordinates);
            delete (coordinates);
        }
    }

    mappedImage.reshape(normalized_y, normalized_x);
    return mappedImage;
}

/**
 * @brief Problem 1 - using svd to solve a system and then remap to a new iamge
 * 
 */
void Problem1()
{
    //initialize a, x, b
    a = new float *[m];
    for (int i = 0; i < m; i++)
        a[i] = new float[n];

    x = new float[n];
    bx = new float[m];
    by = new float[m];

    for (int dataSetIndex = 0; dataSetIndex < NumberOfDataSets; dataSetIndex++)
    {
        if (dataSetIndex == 0)
        {
            CurrentDataSet_X = DataSetS1_XValues;
            CurrentDataSet_Y = DataSetS1_YValues;
            CurrentScaled_X = DataSetS1_scaledXValues;
            CurrentScaled_Y = DataSetS1_scaledYValues;
        }
        else if (dataSetIndex == 1)
        {
            CurrentDataSet_X = DataSetS2_XValues;
            CurrentDataSet_Y = DataSetS2_YValues;
            CurrentScaled_X = DataSetS2_scaledXValues;
            CurrentScaled_Y = DataSetS2_scaledYValues;
        }

        for (int imageIndex = 0; imageIndex < NumberOfImagesInDataSet; imageIndex++)
        {
            LoadArrays(imageIndex);

            //flatten
            float *a_flat = flatten(m, n, a);
            Mat a_mat = Mat(m, n, 5, a_flat);
            Mat bx_mat = Mat(m, 1, 5, bx);
            Mat by_mat = Mat(m, 1, 5, by);

            //solve for p-hat x, p-hat y using SVD
            Mat x_solution = SolveSystemsWithSVD(a_mat, bx_mat, m, n);
            Mat y_solution = SolveSystemsWithSVD(a_mat, by_mat, m, n);

            cout << x_solution << endl; 
            cout << y_solution << endl; 
            const Mat tm = (Mat_<float>(2, 3) << x_solution.at<float>(0), x_solution.at<float>(1), x_solution.at<float>(2), y_solution.at<float>(0), y_solution.at<float>(1), y_solution.at<float>(2));

            char pathToRead[50]; 
            sprintf(pathToRead, "%s%d%s%d%s", "./Images/S",  dataSetIndex + 1, "/", imageIndex + 1, ".pgm\0");
            Mat image = ReadOriginalImage(pathToRead);
            Mat remappedImage = RemapImage(image, x_solution, y_solution);

            char pathToWrite[50];
            sprintf(pathToWrite, "%s%d%s%d%s", "./Images/S",  dataSetIndex + 1, "/", imageIndex + 1, "_normalized.pgm");
            SaveImage(remappedImage, pathToWrite);
        }
    }
}

/**
 * @brief Loads the matrix for problem 2
 * 
 * @param image the image to load from 
 */
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

/**
 * @brief Problem 2 - using svd to solve an over-determined system of equations to find and smooth lighting
 * 
 */
void Problem2()
{
    int mnNormSIze = normalized_x * normalized_y;

    a2 = new float *[mnNormSIze];
    for (int i = 0; i < mnNormSIze; i++)
        a2[i] = new float[4];

    for (int dataSetIndex = 0; dataSetIndex < NumberOfDataSets; dataSetIndex++)
    {
        for (int imageIndex = 0; imageIndex < NumberOfImagesInDataSet; imageIndex++)
        {
            char buffer[50]; 
            sprintf(buffer, "%s%d%s%d%s", "./Images/S",  dataSetIndex + 1, "/", imageIndex, ".pgm");
            Mat image = ReadOriginalImage(buffer);

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

            Mat wut = SolveSystemsWithSVD(a2_mat, b2_mat, mnNormSIze, 4);
        }
    }
}

/**
 * @brief calls each of the sub problems
 * 
 * @param argc argument count
 * @param argv arguments
 * @return int 
 */
int main(int argc, char *argv[])
{
    //problem 1 using SVD to solve the affine transformations
    Problem1();

    //problem 2 using SVD to solve the over constrained systems of pixel intensities
    // Problem2();

    //wait for keystroke so user knows program is done
    waitKey(0);

    return 0;
}