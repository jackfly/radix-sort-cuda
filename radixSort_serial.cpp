#include<iostream>
#include <fstream>
#include <vector>
#include <string> 
#include <sstream>
#include <math.h>
#include <time.h>
using namespace std;
 
// A utility function to get maximum value in arr[]
int getMax(int arr[], int n)
{
    int mx = arr[0];
    for (int i = 1; i < n; i++)
        if (arr[i] > mx)
            mx = arr[i];
    return mx;
}
 
// A function to do counting sort of arr[] according to
// the digit represented by exp.
void countSort(int arr[], int n, int exp)
{
    //int output[n]; // output array
    int* output = new int[n];
    int i, count[10] = {0};
 
    // Store count of occurrences in count[]
    for (i = 0; i < n; i++)
        count[ (arr[i]/exp)%10 ]++;
 
    // Change count[i] so that count[i] now contains actual
    //  position of this digit in output[]
    for (i = 1; i < 10; i++)
        count[i] += count[i - 1];
 
    // Build the output array
    for (i = n - 1; i >= 0; i--)
    {
        output[count[ (arr[i]/exp)%10 ] - 1] = arr[i];
        count[ (arr[i]/exp)%10 ]--;
    }
 
    // Copy the output array to arr[], so that arr[] now
    // contains sorted numbers according to current digit
    for (i = 0; i < n; i++)
        arr[i] = output[i];
}
 
// The main function to that sorts arr[] of size n using 
// Radix Sort
void radixsort(int arr[], int n)
{
    // Find the maximum number to know number of digits
    int m = getMax(arr, n);
    // Do counting sort for every digit. Note that instead
    // of passing digit number, exp is passed. exp is 10^i
    // where i is current digit number
    for (int exp = 1; m/exp > 0; exp *= 10)
        countSort(arr, n, exp);
}
 
// A utility function to print an array
void print(int arr[], int n)
{
    for (int i = 0; i < n; i++)
        cout << arr[i] << " ";
}
 
// Driver program to test above functions
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
        //int numbers[num];
        //Create an input file stream
        fstream file(filename);
        for(int m=0; m<num; m++)
        {
            file >> numbers[linecount];
            linecount++;
        }
        //Close the file stream
        file.close();
        //int n = sizeof((int *)&numbers[0])/sizeof(numbers[0]);
        //print(&numbers[1048565], 10);
        clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &start);
        radixsort(numbers, num);
        clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &stop);
        double dt = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) / 1e3;    // in microseconds
        printf("@time of serial run:\t\t\t[%.3f] microseconds\n", dt);
        delete[] numbers;
    }
    return 0;
}