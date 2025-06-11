#pragma once

#include <stdint.h>
#include <limits.h>

double max(double a, double b) { return (a > b) ? a : b; }
double min(double a, double b) { return (a < b) ? a : b; }

int _total_CPUS;
int cache_block_size = 64; // in bytes, which configured in Gem5

typedef struct _index3d
{
    int height_;
    int width_;
    int depth_;
}index3d;

int get_index(index3d *index, int x, int y, int channel)
{
    return (index->height_ * channel + y) * index->width_ + x;
}

index3d new_index3d(int x, int y, int c)
{
    index3d ret = {y, x, c};
    return ret;
}

typedef struct _cnn_controller
{
    void *nwk_cur_ptr;
    void *lyr_cur_ptr;
    void *wgt_cur_ptr;
    int padding_size;
    int total_CPUs;
}cnn_controller;

typedef struct _output_index_name
{
    int index;
    float value;
    char name[40];
}output_index_name;


// int atomic_or(volatile int *addr, int val) {
//     int old;
//     asm volatile (
//         "amoor.d %0, %2, %1\n"
//         : "=r"(old), "+A"(*addr)
//         : "r"(val)
//         : "memory"
//     );
//     return old;
// }   

// int atomic_or(volatile int *addr, int val) {
//     return __sync_fetch_and_or(addr, val);
// }

void init_util(int total_CPUS)
{
    _total_CPUS = total_CPUS;
}

int compute_block_size(int in)
{
    int block_size = in / _total_CPUS;
    if (in % _total_CPUS)
    {
        block_size++;
    }
    return block_size;
}

int my_sprintf(char *str, const char *format, int value) {
    int i = 0;
    int j = 0;
    char buffer[20]; // Sufficient for most integers

    while (format[i] != '\0') {
        if (format[i] == '%') {
            i++;
            if (format[i] == 'd') {
                // Integer formatting
                int temp = value;
                int k = 0;
                if (temp == 0) {
                    buffer[k++] = '0';
                } else {
                    if (temp < 0) {
                        str[j++] = '-';
                        temp = -temp;
                    }

                    while (temp != 0) {
                        buffer[k++] = (temp % 10) + '0';
                        temp /= 10;
                    }

                    // Reverse the buffer
                    for (int l = k - 1; l >= 0; l--) {
                        str[j++] = buffer[l];
                    }
                }
                i++; // Move past the format specifier
            } else if (format[i] == '%') {
                str[j++] = '%';
                i++;
            } else {
                // Handle unsupported format specifier (optional)
                str[j++] = '%'; // Copy the original '%'
                str[j++] = format[i++]; // Copy the character after '%'
            }
        } else {
            str[j++] = format[i++];
        }
    }

    str[j] = '\0'; // Null-terminate the string
    return j; // Return the number of characters written
}

double no_math_ceil(double x)
{
    if (x >= LLONG_MAX && x < LLONG_MIN)
    {
        intmax_t i = (intmax_t) x;      // this rounds towards 0
        if (i < 0 || x == i) return i;  // negative x is already rounded up.
        return i + 1.0;
    }
    return x;
}