#include <errno.h>
#include <stdint.h>

int _close(int file) {
  errno = EBADF;
  return -1;
}

int _lseek(int file, int ptr, int dir) {
  errno = EBADF;
  return -1;
}

int _read(int file, char *ptr, int len) {
  errno = EBADF;
  return -1;
}

int _write(int file, char *ptr, int len) {
  errno = EBADF;
  return -1;
}

void *_sbrk(int incr) {
  // extern char _heap_start; // Defined in linker script
  // extern char _heap_end;
  // static char *heap_ptr = &_heap_start;

  // if (heap_ptr + incr > &_heap_end) {
  //   errno = ENOMEM;
  //   return (void *) -1;
  // }
  // char *prev_heap_ptr = heap_ptr;
  // heap_ptr += incr;
  // return (void *) prev_heap_ptr;
  char *prev_heap_ptr;
  return (void *) prev_heap_ptr;
}