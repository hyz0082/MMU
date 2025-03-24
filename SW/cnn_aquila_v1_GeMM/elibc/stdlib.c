// =============================================================================
//  Program : stdlib.c
//  Author  : Chun-Jen Tsai
//  Date    : Dec/09/2019
// -----------------------------------------------------------------------------
//  Description:
//  This is the minimal stdlib library for Aquila.
// -----------------------------------------------------------------------------
//  Revision information:
//
//  Apr/01/2021, by Po-Wei Ho:
//     Fixed two bugs in malloc(). The first bug is that the for-loop index 'ptr'
//     was sometimes updated without clearing the used/unused flag.
//     The second bug is that 'curr_top' can somtimes point to itself.
//
//  None.
// -----------------------------------------------------------------------------
//  License information:
//
//  This software is released under the BSD-3-Clause Licence,
//  see https://opensource.org/licenses/BSD-3-Clause for details.
//  In the following license statements, "software" refers to the
//  "source code" of the complete hardware/software system.
//
//  Copyright 2019,
//                    Embedded Intelligent Systems Lab (EISL)
//                    Deparment of Computer Science
//                    National Chiao Tung Uniersity
//                    Hsinchu, Taiwan.
//
//  All rights reserved.
//
//  Redistribution and use in source and binary forms, with or without
//  modification, are permitted provided that the following conditions are met:
//
//  1. Redistributions of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//  2. Redistributions in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//  3. Neither the name of the copyright holder nor the names of its contributors
//     may be used to endorse or promote products derived from this software
//     without specific prior written permission.
//
//  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
//  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
//  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
//  ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
//  LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
//  CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
//  SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
//  INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
//  CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
//  ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
//  POSSIBILITY OF SUCH DAMAGE.
// =============================================================================
#include <stdio.h>
#include <stdlib.h>
#include <string.h>


// typedef unsigned long ulong;

// extern ulong __heap_start; /* declared in the linker script */
// extern ulong __heap_size;  /* declared in the linker script */
// static ulong heap_top = (ulong) &__heap_start;
// static ulong heap_size = (ulong) &__heap_size;
// static ulong *curr_top = (ulong *) 0xFFFFFFF0, *heap_end = (ulong *) 0xFFFFFFF0;

// void *malloc(size_t n)
// {
//     void *return_ptr;
//     ulong *ptr, temp;
//     int r;

//     // printf("curr heap size: %d\n", heap_size);

//     if (curr_top == heap_end) // first time to call malloc()?
//     {
//         curr_top = (ulong *) heap_top;
//         heap_end = (ulong *) ((heap_top + heap_size) & 0xFFFFFFF0);
//         *curr_top = (ulong) heap_end;
//     }

//     // Search for a large-enough free memory block (FMB).
//     return_ptr = NULL;
//     for (ptr = curr_top; ptr < heap_end; ptr = (ulong *) (*ptr & 0xFFFFFFFE))
//     {
//         if ((*ptr & 1) == 0 && (*ptr - (ulong) ptr > n))
//         {
//             temp = ((ulong) ptr) & 0xFFFFFFFE;
//             return_ptr = (void *) (temp + sizeof(ulong));

//             // Update the FMB link list structure.
//             r = n % sizeof(ulong);
//             temp = n + sizeof(ulong) + ((r)? 4-r : 0);
//             curr_top = ptr + temp/sizeof(ulong);
//             if (curr_top != (ulong *) *ptr)
//                 *curr_top = *ptr;
//             *ptr = (ulong) curr_top | 1;
//             break;
//         }
//     }

//     if (return_ptr != NULL) return return_ptr;

//     // Search again for a FMB from heap_top to curr_top
//     for (ptr = (ulong *) heap_top; ptr < curr_top; ptr = (ulong *) (*ptr & 0xFFFFFFFE))
//     {
//         if ((*ptr & 1) == 0 && (*ptr - (ulong) ptr > n))
//         {
//             temp = ((ulong) ptr) & 0xFFFFFFFE;
//             return_ptr = (void *) (temp + sizeof(ulong));

//             // Update the FMB link list structure.
//             r = n % sizeof(ulong);
//             temp = n + sizeof(ulong) + ((r)? 4-r : 0);
//             curr_top = ptr + temp/sizeof(ulong);
//             if (curr_top != (ulong *) *ptr)
//                 *curr_top = *ptr;
//             *ptr = (ulong) curr_top | 1;
//             break;
//         }
//     }

//     return return_ptr;
// }

// void free(void *mptr)
// {
//     ulong *ptr, *next;

//     ptr = ((ulong *) mptr) - 1;
//     *ptr = *ptr & 0xFFFFFFFE; // Free the FMB.
//     next = (ulong *) *ptr;
//     if ((*next & 1) == 0)
//     {
//         *ptr = *next; // Merge with the next FMB.
//         curr_top = ptr;
//     }
// }


#include <stddef.h>
#include <stdint.h>

typedef uintptr_t uptr;

typedef struct header_t {
    uptr size;
    struct header_t* next;
    int is_free;
} header_t;

extern uptr __heap_start;
extern uptr __heap_size;

static header_t* heap_start = NULL;
static header_t* heap_end = NULL;

void* malloc(size_t size) {
    if (size == 0) {
        return NULL;
    }

    if (heap_start == NULL) {
        heap_start = (header_t*)&__heap_start;
        heap_end = (header_t*)((uptr)heap_start + __heap_size);
        heap_start->size = __heap_size - sizeof(header_t);
        heap_start->next = NULL;
        heap_start->is_free = 1;
    }

    header_t* current = heap_start;
    header_t* best_fit = NULL;

    while (current != NULL) {
        if (current->is_free && current->size >= size) {
            if (best_fit == NULL || current->size < best_fit->size) {
                best_fit = current;
            }
        }
        current = current->next;
    }

    if (best_fit == NULL) {
        return NULL;
    }

    if (best_fit->size > size + sizeof(header_t)) {
        header_t* new_header = (header_t*)((uptr)best_fit + sizeof(header_t) + size);
        new_header->size = best_fit->size - size - sizeof(header_t);
        new_header->next = best_fit->next;
        new_header->is_free = 1;

        best_fit->size = size;
        best_fit->next = new_header;
    }

    best_fit->is_free = 0;
    return (void*)((uptr)best_fit + sizeof(header_t));
}

void free(void* ptr) {
    if (ptr == NULL) {
        return;
    }

    header_t* header = (header_t*)((uptr)ptr - sizeof(header_t));
    header->is_free = 1;

    header_t* current = heap_start;
    header_t* previous = NULL;

    while (current != NULL) {
        if (current == header) {
            break;
        }
        previous = current;
        current = current->next;
    }

    if (current == NULL) {
        return; // Invalid pointer
    }

    // Coalesce with next block
    if (header->next != NULL && header->next->is_free) {
        header->size += sizeof(header_t) + header->next->size;
        header->next = header->next->next;
    }

    // Coalesce with previous block
    if (previous != NULL && previous->is_free) {
        previous->size += sizeof(header_t) + header->size;
        previous->next = header->next;
    }

}

void *calloc(size_t n, size_t size)
{
    void *mptr;
    mptr = malloc(n*size);
    memset(mptr, 0, n*size);
    return mptr;
}

int atoi(char *s)
{
    int value, sign;

    /* skip leading while characters */
    while (*s == ' ' || *s == '\t') s++;
    if (*s == '-') sign = -1, s++;
    else sign = 1;
    if (*s >= '0' && *s <= '9') value = (*s - '0');
    else return 0;
    s++;
    while (*s != 0)
    {
       if (*s >= '0' && *s <= '9')
       {
           value = value * 10 + (*s - '0');
           s++;
       }
       else return 0;
    }

    return value * sign;
}

int abs(int n)
{
    int j;

    if (n >= 0) j = n; else j = -n;

	return j;
}

#pragma GCC push_options
#pragma GCC optimize ("O0")
void exit(int status)
{
    printf("\n-----------------------------------------------------------------------\n");
    printf("Program exit with a status code %d\n", status);
    printf("Press <reset> on the FPGA board to reboot the cpu ...\n\n");

    // If Aquila is running in a waveform simulator, we can use putchar(03)
    // to inform the simulator to end simulation if exit() has been called.
    // However, you need a UART module that invokes $finish() when a 0x03 code
    // has been sent to the UART device in simulation mode.
    putchar(03);

    while (1);
}
#pragma GCC pop_options

static int rand_seed = 27182;

void srand(unsigned int seed)
{
    rand_seed = (long) seed;
}

int rand(void)
{
    return(((rand_seed = rand_seed * 214013L + 2531011L) >> 16) & 0x7fff);
}
