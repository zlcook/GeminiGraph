/*
Copyright (c) 2015-2016 Xiaowei Zhu, Tsinghua University

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.

   MPI refer to https://computing.llnl.gov/tutorials/mpi/#BuildScripts
*/

#ifndef MPI_HPP
#define MPI_HPP

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <mpi.h>

template <typename T>
MPI_Datatype get_mpi_data_type() { // MPI predefines its data types for the reasons of portability
  if (std::is_same<T, char>::value) {
    return MPI_CHAR;       // 
  } else if (std::is_same<T, unsigned char>::value) {
    return MPI_UNSIGNED_CHAR;
  } else if (std::is_same<T, int>::value) {
    return MPI_INT;
  } else if (std::is_same<T, unsigned>::value) {
    return MPI_UNSIGNED;
  } else if (std::is_same<T, long>::value) {
    return MPI_LONG;
  } else if (std::is_same<T, unsigned long>::value) {
    return MPI_UNSIGNED_LONG;
  } else if (std::is_same<T, float>::value) {
    return MPI_FLOAT;
  } else if (std::is_same<T, double>::value) {
    return MPI_DOUBLE;
  } else {
    printf("type not supported\n");
    exit(-1);
  }
}

class MPI_Instance {
  int partition_id;
  int partitions;
public:
  MPI_Instance(int * argc, char *** argv) {
    int provided;

    MPI_Init_thread(argc, argv, MPI_THREAD_MULTIPLE, &provided); //Initialize the execution environment. used to pass command line arguments to all processes. This function must be called before any other MPI functions.
    MPI_Comm_rank(MPI_COMM_WORLD, &partition_id); //return the rank of the calling MPI process within the specified communicator. Each process will be assigned a unique rank between 0 and number of tasks -1 within the communicator MPI_COMM_WORLD
    MPI_Comm_size(MPI_COMM_WORLD, &partitions); //return the total number of MPI processes in the communicator MPI_COMM_WORLD¡£ 
    #ifdef PRINT_DEBUG_MESSAGES
    if (partition_id==0) {
      printf("thread support level provided by MPI: ");
      switch (provided) {
        case MPI_THREAD_MULTIPLE:  // 3: The process may be multi-thread, and multiple threads may call MPI with no restrictions.
          printf("MPI_THREAD_MULTIPLE\n"); break;
        case MPI_THREAD_SERIALIZED:  // 2: The process may be mutlti-thread, and multiple threads call make MPI calls , but only one at a time.
          printf("MPI_THREAD_SERIALIZED\n"); break;
        case MPI_THREAD_FUNNELED:   //1: The process may be multi-thread, but only the main thread can make MPI calls - all MPI calls are funneled to the main thread.
          printf("MPI_THREAD_FUNNELED\n"); break;
        case MPI_THREAD_SINGLE:     //0 : Only one thread will execute in the process.
          printf("MPI_THREAD_SINGLE\n"); break;
        default:
          assert(false);
      }
    }
    #endif
  }
  ~MPI_Instance() {
    MPI_Finalize();  // Terminates the MPL execution environment.This function should be the last MPI routine called in every MPI program.
  }
  void pause() {
    if (partition_id==0) {
      getchar(); // read next char from stdin
    }
    MPI_Barrier(MPI_COMM_WORLD);  //Each task ,when reaching the MPI_Barrier call, blocks until all tasks in the group reach the same MPI_Barrier
  }
};

#endif
