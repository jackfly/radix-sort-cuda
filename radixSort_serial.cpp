// Sequenctial Implementation (Radix Sort)
// Final Project
#include<iostream>
#include <fstream>
#include <vector>
#include <string> 
#include <sstream>
#include <math.h>
#include <time.h>
using namespace std;
 
int getMaxNum(int arr[], int n)
{
    int maxNum = arr[0];
    for (int i = 1; i < n; i++)
        if (arr[i] > maxNum)
            maxNum = arr[i];
    return maxNum;
}
 
// This is for the counting sort
void countSort(int arr[], int n, int exp)
{
    int* output = new int[n];
    int i, count[10] = {0};
    // Calcuate count number
    for (i = 0; i < n; i++)
        count[ (arr[i]/exp)%10 ]++;
 
    // Prefix sum calculation
    for (i = 1; i < 10; i++)
        count[i] += count[i - 1];
 
    // Output array
    for (i = n - 1; i >= 0; i--)
    {
        output[count[ (arr[i]/exp)%10 ] - 1] = arr[i];
        count[ (arr[i]/exp)%10 ]--;
    }
 
    // Copy the output array to arr[]
    for (i = 0; i < n; i++)
        arr[i] = output[i];
}
 
// Radix Sort
void radixsort(int arr[], int n)
{
    // Find the maximum number
    int m = getMaxNum(arr, n);
    // perform counting sort
    for (int exp = 1; m/exp > 0; exp *= 10)
        countSort(arr, n, exp);
}
 
int main()
{
    struct timespec start, stop;
    for (int i = 0; i<= 18; ){
        i = i+2;
        int num = pow(2,i);
        //printf("%d\n",num);
        int linecount = 0;
        string filename = "RandomNumbers/" + to_string(i) + ".txt";
        int* numbers = new int[num];

        fstream file(filename);
        for(int m=0; m<num; m++)
        {
            file >> numbers[linecount];
            linecount++;
        }
        //Close the file stream
        file.close();

        clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &start);
        radixsort(numbers, num);
        clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &stop);
        double dt = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) / 1e3;    // in microseconds
        printf("@time of serial run:\t\t\t[%.3f] microseconds\n", dt);
        delete[] numbers;
    }
    return 0;
}