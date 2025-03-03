# all: alexnet resnet18 resnet50 ultranet

# # CC = riscv64-unknown-linux-gnu-gcc
# CC = gcc
# CFLAGS = -static -lm
# SRCDIR = ./src

# alexnet: $(SRCDIR)/alexnet.c
# 	$(CC) $^ -DUSING_GEM5 -DPRINT_LAYER $(CFLAGS) -o $@
# resnet18: $(SRCDIR)/resnet18.c
# 	$(CC) $^ -DUSING_GEM5 -DPRINT_LAYER $(CFLAGS) -o $@
# resnet50: $(SRCDIR)/resnet50.c
# 	$(CC) $^  -DPRINT_LAYER $(CFLAGS) -o $@
# ultranet: $(SRCDIR)/ultranet.c
# 	$(CC) $^ -DUSING_GEM5 -DPRINT_LAYER $(CFLAGS) -o $@

# clean:
# 	rm -f alexnet resnet18 resnet50 ultranet

# ======================================================================
#  This Makefile builds an application *.elf for Aquila.
#
#  Chun-Jen Tsai, Dec/05/2024
# ======================================================================

CROSS = riscv32-unknown-elf
CCPATH = $(RISCV)/bin

CC = $(CCPATH)/$(CROSS)-gcc
LD = $(CCPATH)/$(CROSS)-ld
AR = $(CCPATH)/$(CROSS)-ar
OD = $(CCPATH)/$(CROSS)-objdump
SP = $(CCPATH)/$(CROSS)-strip

GCCVERSION = $(shell $(CC) --version | grep gcc | sed 's/^.* //g')

LIBC = ../elibc
LIBC_OBJS = $(LIBC)/crt0.o $(LIBC)/stdio.o $(LIBC)/stdlib.o $(LIBC)/string.o $(LIBC)/time.o $(LIBC)/uart.o

LIB_FILE = ./sdcard
LIB_FILE_OBJS = $(LIB_FILE)/sd.o $(LIB_FILE)/spi.o $(LIB_FILE)/fat32.o

CCFLAGS = -Wall -O3 -I$(LIBC) -I$(LIB_FILE) -fno-builtin -march=rv32ima_zicsr_zifencei -mstrict-align -mabi=ilp32 
LDFLAGS = -L. -Map=$*.map 
LD_SOFT_FP = -L$(RISCV)/lib/gcc/riscv32-unknown-elf/$(GCCVERSION) -lgcc
CNN = ./inc_cnn

OCFLAGS = -O binary
ODFLAGS = -d

PROJ = resnet50
OBJS = $(PROJ).o file_read.o
HDRS = $(CNN)/activation_function.h \
       $(CNN)/average_pooling_layer.h \
	   $(CNN)/batchnorm_layer.h \
	   $(CNN)/class_name.h \
	   $(CNN)/config.h \
       $(CNN)/convolutional_layer.h \
	   $(CNN)/dummy_head_layer.h \
	   $(CNN)/fully_connected_layer.h \
	   $(CNN)/layer.h \
	   $(CNN)/list.h \
	   $(CNN)/loader.h \
	   $(CNN)/max_pooling_layer.h \
	   $(CNN)/network.h \
	   $(CNN)/residual_block_interface.h \
	   $(CNN)/util.h \
	   file_read.h 


all:
	make $(PROJ).elf 
	gcc -static -lm -g -DPRINT_LAYER -DSTORE_MID resnet50_sw.c

clean:
	rm -f *.elf *.objdump *.map *.o *.a *.out

%.o: %.c  $(HDRS)
	$(CC) -I$(CNN) -DUSING_GEM5 -DPRINT_LAYER $(CCFLAGS) -c $< -o $@

libelibc.a : $(LIBC_OBJS) $(LIB_FILE_OBJS)
	$(AR) r libelibc.a $(LIBC_OBJS) $(LIB_FILE_OBJS)
	rm $^

%.elf: $(OBJS) libelibc.a
	$(LD) $(LDFLAGS) -T$(PROJ).ld $(OBJS) -lelibc $(LD_SOFT_FP) -o $@
	$(OD) $(ODFLAGS) $@ > $*.objdump
	$(SP) -g $@ 
	rm $^

