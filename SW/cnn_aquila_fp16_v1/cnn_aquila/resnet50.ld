/*
// =============================================================================
//  Program : ocr.ld
//  Author  : Chun-Jen Tsai
//  Date    : Dec/06/2023
// -----------------------------------------------------------------------------
//  Revision information:
//
//  None.
// -----------------------------------------------------------------------------
//  Description:
//  This is the linker script of a program compiled to run from DDRx memory.
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
 // 061B1EA0 0x200000
     7000000
*/

/*
__stack_size = 0x800;
__heap_size  = 0x200000;

MEMORY
{
    code_ram   (rx!rw) : ORIGIN = 0x80001000, LENGTH = 0x10000
    data_ram   (rw!x)  : ORIGIN = 0x80011000, LENGTH = 0x310000
}

ENTRY(crt0)

SECTIONS
{
    .text :
    {
        *(.text*)
    } > code_ram

    .data :
    {
        *(.data)
        *(.bss)
        *(.rodata*)
    } > data_ram

    .heap : ALIGN(0x10)
    {
        __heap_start = .;
        . += __heap_size;
    } > data_ram

    .stack : ALIGN(0x10)
    {
        . += __stack_size;
        __stack_top = .;
    } > data_ram
}
*/

/*
gpt generated
*/
/*
__stack_size = 0x800;
__heap_size = 0x700000;

MEMORY
{
    code_ram   (rx!rw) : ORIGIN = 0x80001000, LENGTH = 0x10000;
    data_ram   (rw!x)  : ORIGIN = 0x80011000, LENGTH = 0xA10000;
}

ENTRY(crt0)

SECTIONS
{
    .text :
    {
        *(.text*)
    } > code_ram

    .data :
    {
        *(.data)
        *(.bss)
        *(.rodata*)
    } > data_ram

    .heap : ALIGN(0x10)
    {
        __heap_start = .;
        . += __heap_size;
    } > data_ram

    .stack : ALIGN(0x10)
    {
        . += __stack_size;
        __stack_top = .;
    } > data_ram
}
*/



/*
__stack_size = 0x30000;
__heap_size  = 0xFD00000;

MEMORY
{
    code_ram   (rx!rw) : ORIGIN = 0x80001000, LENGTH = 0x100000
    data_ram   (rw!x)  : ORIGIN = 0x80101000, LENGTH = 0xFF00000
}
*/
/*
__stack_size = 0x30000;
__heap_size  = 0x2D000000;
*/

__stack_size = 0x60000;
__heap_size  = 0xD000000;
MEMORY
{
    code_ram   (rx!rw) : ORIGIN = 0x80001000, LENGTH = 0x100000
    data_ram   (rw!x)  : ORIGIN = 0x80201000, LENGTH = 0xF000000
}

/*
__stack_size = 0x62000;
__heap_size  = 0xD000000;
MEMORY
{
    code_ram   (rx!rw) : ORIGIN = 0x80001000, LENGTH = 0x120000
    data_ram   (rw!x)  : ORIGIN = 0x80201000, LENGTH = 0xF000000
}
*/


ENTRY(crt0)

SECTIONS
{
    .text :
    {
        *(.text*)
    } > code_ram

    .data :
    {
        *(.data)
        *(.bss)
        *(.rodata*)
    } > data_ram

    .heap : ALIGN(0x10)
    {
        __heap_start = .;
        . += __heap_size;
    } > data_ram

    .stack : ALIGN(0x10)
    {
        . += __stack_size;
        __stack_top = .;
    } > data_ram

    
}

/*
ENTRY(crt0)

SECTIONS
{
    .text :
    {
        *(.text*)
    } > code_ram

    .data :
    {
        *(.data)
        *(.bss)
        *(.rodata*)
    } > data_ram

    .stack : ALIGN(0x10)
    {
        . += __stack_size;
        __stack_top = .;
    } > data_ram

    .heap : ALIGN(0x10)
    {
        __heap_start = .;
        . += __heap_size;
    } > data_ram

    
}
*/