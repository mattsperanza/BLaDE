#ifndef MAIN_DEFINES_H
#define MAIN_DEFINES_H

#define MAXLENGTHSTRING 1024
#define ANGSTROM 0.1
#define KCAL_MOL 4.186
#define DEGREES 0.017453292519943
#define kB 0.0083144598

// CUDA block size for system update kernels
#define BLUP 256

#ifdef DOUBLE
typedef double real;
#else
typedef float real;
#endif

struct Int2 {
  int i[2];
};

struct Int3 {
  int i[3];
};

struct Int4 {
  int i[4];
};

struct Int8 {
  int i[8];
};

struct Real3 {
  real i[3];
};

#endif
