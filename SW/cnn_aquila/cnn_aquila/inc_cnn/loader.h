#pragma once

#include <sys/stat.h>
#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/mman.h>

void *weight_base;
void *input_base;

void init_loader(char *weight_name, char *input_name)
{
    int weight_fd = open(weight_name, O_RDONLY);
    int input_fd = open(input_name, O_RDWR);

    if (weight_fd == -1 || input_fd == -1)
    {
        perror("open");
        exit(-1);
    }

    struct stat weightstat, inputstat;

    if (fstat(weight_fd, &weightstat) == -1)
    {
        perror("fstat");
        close(weight_fd);
        exit(-1);
    }

    if (fstat(input_fd, &inputstat) == -1)
    {
        perror("fstat");
        close(input_fd);
        exit(-1);
    }

    size_t weight_size = weightstat.st_size;
    size_t input_size = inputstat.st_size;

    weight_base = mmap(NULL, weight_size, PROT_READ, MAP_SHARED, weight_fd, 0);
    if (weight_base == MAP_FAILED) {
        perror("mmap");
        close(weight_fd);
        exit(-1);
    }

    input_base = mmap(NULL, input_size, PROT_READ | PROT_WRITE, MAP_SHARED, input_fd, 0);
    if (input_base == MAP_FAILED) {
        perror("mmap");
        close(input_fd);
        exit(-1);
    }

    close(weight_fd);
    close(input_fd);
}