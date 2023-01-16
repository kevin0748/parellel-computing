# Sorting

## Pthread
```sh
icc -o sort_list_pthread.exe sort_list_pthread.c
./sort_list_pthread.exe 20 4
```

## OpenMP
```sh
icc -qopenmp -o sort_list_openmp.exe sort_list_openmp.c
export OMP_NUM_THREADS=8
sort_list_openmp.exe 20 4
```

## MPI
```sh
mpiicpc -o qsort_hypercube.exe qsort_hypercube.cpp
mpirun -np 2 ./qsort_hypercube.exe 4 -1
```