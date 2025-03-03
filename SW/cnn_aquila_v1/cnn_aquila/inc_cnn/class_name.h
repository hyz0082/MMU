#pragma once

#include <sys/stat.h>
#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <fcntl.h>
// #include <sys/mman.h>
#include <string.h>

void *class_name_base;

void init_class_name(char *class_name)
{
    printf("Loading class name: <%s>\n", class_name);
    int class_name_fd = open(class_name, O_RDONLY);

    if (class_name_fd == -1)
    {
        perror("open");
        exit(-1);
    }

    struct stat class_name_tstat;

    if (fstat(class_name_fd, &class_name_tstat) == -1)
    {
        perror("fstat");
        close(class_name_fd);
        exit(-1);
    }

    size_t class_name_size = class_name_tstat.st_size;

    class_name_base = mmap(NULL, class_name_size, PROT_READ, MAP_SHARED, class_name_fd, 0);
    if (class_name_base == MAP_FAILED) {
        perror("mmap");
        close(class_name_fd);
        exit(-1);
    }

    close(class_name_fd);
}

char *mmap_fgets(char *buf, size_t size, char **mmap_ptr) {

    char *current_ptr = *mmap_ptr;
    char *newline_ptr = memchr(current_ptr, '\n', size);

    if (newline_ptr != NULL) {
        size_t line_length = newline_ptr - current_ptr + 1;
        if (line_length >= size) {
            line_length = size - 1;
        }
        memcpy(buf, current_ptr, line_length);
        buf[line_length] = '\0';
        *mmap_ptr = current_ptr + line_length;
    }

    return buf;
}