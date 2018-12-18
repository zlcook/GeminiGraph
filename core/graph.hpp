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
*/

#ifndef GRAPH_HPP
#define GRAPH_HPP

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <fcntl.h>
#include <malloc.h>
#include <sys/mman.h>
#include <numa.h>
#include <omp.h>

#include <string>
#include <vector>
#include <thread>
#include <mutex>

#include "core/atomic.hpp"
#include "core/bitmap.hpp"
#include "core/constants.hpp"
#include "core/filesystem.hpp"
#include "core/mpi.hpp"
#include "core/time.hpp"
#include "core/type.hpp"

enum ThreadStatus {
  WORKING,
  STEALING
};

enum MessageTag {
  ShuffleGraph,
  PassMessage,
  GatherVertexArray
};

struct ThreadState {
  VertexId curr;
  VertexId end;
  ThreadStatus status;
};

struct MessageBuffer {
  size_t capacity;
  int count; // the actual size (i.e. bytes) should be sizeof(element) * count
  char * data;
  MessageBuffer () {
    capacity = 0;
    count = 0;
    data = NULL;
  }
  void init (int socket_id) {
    capacity = 4096;
    count = 0;
    data = (char*)numa_alloc_onnode(capacity, socket_id);
  }
  void resize(size_t new_capacity) {
    if (new_capacity > capacity) {
      char * new_data = (char*)numa_realloc(data, capacity, new_capacity);
      assert(new_data!=NULL);
      data = new_data;
      capacity = new_capacity;
    }
  }
};

template <typename MsgData>
struct MsgUnit {
  VertexId vertex;   //����id��
  MsgData msg_data;  //��Ϣ
} __attribute__((packed));

template <typename EdgeData = Empty>
class Graph {
public:
  int partition_id;  //��ǰ�����ķ���id, between 0 and the number of processors -1
  int partitions;  //��Ⱥ������ = the number of processors

  size_t alpha;   //�������ᵽ�ģ����ڸ��ؾ���ĳ��Ρ�

  int threads;          //��ǰ�����������߳�����
  int sockets;         //��ǰ����������numa�ڵ���
  int threads_per_socket;  //ÿ��numa�ڵ���߳���

  size_t edge_data_size;   // ��ֵ��С�� weight
  size_t unit_size;        // �ߴ�С: һ������+weight
  size_t edge_unit_size;   //һ���ߴ�С�� src +dst+ weight

  bool symmetric;
  VertexId vertices;
  EdgeId edges;
  VertexId * out_degree; // VertexId [vertices]; numa-aware ��������ʵ����numa��֪
  VertexId * in_degree; // VertexId [vertices]; numa-aware  ����ĳ���

  VertexId * partition_offset; // VertexId [partitions+1]   ÿ����������Ķ�������λ�ƣ�i��������Ķ�������Ϊ[ partition_offset[i],partition_offset[i+1] )
  VertexId * local_partition_offset; // VertexId [sockets+1] ��ǰ������ÿ��numa�ڵ㴦��Ķ�������λ��

  VertexId owned_vertices;
  EdgeId * outgoing_edges; // EdgeId [sockets]  ÿ��numa�ڵ��ϵĳ�����
  EdgeId * incoming_edges; // EdgeId [sockets]

  Bitmap ** incoming_adj_bitmap;    //numa-aware��ά��������ߵ�dst���㣬����߷��䵽��ͬ�Ļ����ϵĲ�ͬnuma�ϣ���numa��ӵ�еı�{��src,dst)| src<- vertices, dst<-numa���䶥��}��incoming_adj_bitmap���ڱ��ÿ�������ڵ�ǰ������numa�ڵ����Ƿ�����������
  //ͨ��ѹ��ϡ���У�CSR)��������������
  EdgeId ** incoming_adj_index; // EdgeId [sockets] [vertices+1]; numa-aware ����¼���ж������ڵ�ǰ������numa�ڵ�������ߵĶ��㣬�������incoming_adj_list�е�����λ�á�
  AdjUnit<EdgeData> ** incoming_adj_list; // AdjUnit<EdgeData> [sockets] [vertices+1]; numa-aware�����ж������ڵ�ǰ������numa�ڵ���ӵ����ߵĶ��㣬����ߴ���incoming_adj_list�С�

  Bitmap ** outgoing_adj_bitmap;    //��ά��
  //ͨ��ѹ��ϡ���ࣨCSC)����������ĳ���
  EdgeId ** outgoing_adj_index; // EdgeId [sockets] [vertices+1]; numa-aware
  AdjUnit<EdgeData> ** outgoing_adj_list; // AdjUnit<EdgeData> [sockets] [vertices+1]; numa-aware

  //ͨ��Doubly CSR ��Doubly CSC������������ߡ�������Ϣ
  VertexId * compressed_incoming_adj_vertices;     //numa-aware ��ǰ������ÿ��numa�ڵ�ӵ����ߵĶ�������
  CompressedAdjIndexUnit ** compressed_incoming_adj_index; // CompressedAdjIndexUnit [sockets] [...+1]; numa-aware ����ߵĶ��㼰����incoming_adj_list�е�����λ�á�
  VertexId * compressed_outgoing_adj_vertices;
  CompressedAdjIndexUnit ** compressed_outgoing_adj_index; // CompressedAdjIndexUnit [sockets] [...+1]; numa-aware

  ThreadState ** thread_state; // ThreadState* [threads]; numa-aware
  ThreadState ** tuned_chunks_dense; // ThreadState [partitions][threads];
  ThreadState ** tuned_chunks_sparse; // ThreadState [partitions][threads];

  size_t local_send_buffer_limit;
  MessageBuffer ** local_send_buffer; // MessageBuffer* [threads]; numa-aware

  int current_send_part_id;
  MessageBuffer *** send_buffer; // MessageBuffer* [partitions] [sockets]; numa-aware
  MessageBuffer *** recv_buffer; // MessageBuffer* [partitions] [sockets]; numa-aware

  Graph() {
    threads = numa_num_configured_cpus();
    sockets = numa_num_configured_nodes();  
    threads_per_socket = threads / sockets;

    init();
  }

  //�����߳�������numa�ڵ�id�ţ�socket)
  inline int get_socket_id(int thread_id) {
    return thread_id / threads_per_socket;
  }

  //�߳�����numa�ڵ��е���һ���̺߳�
  inline int get_socket_offset(int thread_id) {
    return thread_id % threads_per_socket;
  }

  void init() {
    edge_data_size = std::is_same<EdgeData, Empty>::value ? 0 : sizeof(EdgeData);
    unit_size = sizeof(VertexId) + edge_data_size;
    edge_unit_size = sizeof(VertexId) + unit_size;

    assert( numa_available() != -1 );
    assert( sizeof(unsigned long) == 8 ); // assume unsigned long is 64-bit

    char nodestring[sockets*2+1]; // 0,1,2,3,...,sockets-1
    nodestring[0] = '0';
    for (int s_i=1;s_i<sockets;s_i++) {
      nodestring[s_i*2-1] = ',';
      nodestring[s_i*2] = '0'+s_i;// char m = '0' + 2 = '2'
    }
    struct bitmask * nodemask = numa_parse_nodestring(nodestring);
    numa_set_interleave_mask(nodemask);

    omp_set_dynamic(0);  //Explicitly turn off dynamic threads. then the number of threads available for parallel regions can't be adjusted automatically by the run-time environment
    omp_set_num_threads(threads);  //set the number of threads that will be used in next parallel region
    thread_state = new ThreadState * [threads];
    local_send_buffer_limit = 16;
    local_send_buffer = new MessageBuffer * [threads];
    for (int t_i=0;t_i<threads;t_i++) {                    //bound thread_state and local_send_buffer per thread to socket
      thread_state[t_i] = (ThreadState*)numa_alloc_onnode( sizeof(ThreadState), get_socket_id(t_i)); //��t_i numa�ڵ��Ϸ����ڴ�
      local_send_buffer[t_i] = (MessageBuffer*)numa_alloc_onnode( sizeof(MessageBuffer), get_socket_id(t_i));
      local_send_buffer[t_i]->init(get_socket_id(t_i));
    }

	//thread bound to socket
    #pragma omp parallel for
    for (int t_i=0;t_i<threads;t_i++) {
      int s_i = get_socket_id(t_i);
      assert(numa_run_on_node(s_i)==0);  
      #ifdef PRINT_DEBUG_MESSAGES
      // printf("thread-%d bound to socket-%d\n", t_i, s_i);
      #endif
    }
    #ifdef PRINT_DEBUG_MESSAGES
    // printf("threads=%d*%d\n", sockets, threads_per_socket);
    // printf("interleave on %s\n", nodestring);
    #endif

	//bound send_buffer and recv_buffer to each socket
    MPI_Comm_rank(MPI_COMM_WORLD, &partition_id);  //get the rank of current processor
    MPI_Comm_size(MPI_COMM_WORLD, &partitions);   //get the total number of processors
    send_buffer = new MessageBuffer ** [partitions];
    recv_buffer = new MessageBuffer ** [partitions];
    for (int i=0;i<partitions;i++) {
      send_buffer[i] = new MessageBuffer * [sockets];
      recv_buffer[i] = new MessageBuffer * [sockets];
      for (int s_i=0;s_i<sockets;s_i++) {
        send_buffer[i][s_i] = (MessageBuffer*)numa_alloc_onnode( sizeof(MessageBuffer), s_i);
        send_buffer[i][s_i]->init(s_i);
        recv_buffer[i][s_i] = (MessageBuffer*)numa_alloc_onnode( sizeof(MessageBuffer), s_i);
        recv_buffer[i][s_i]->init(s_i);
      }
    }

    alpha = 8 * (partitions - 1);  //�������Ż���������ķ���

    MPI_Barrier(MPI_COMM_WORLD);  //Each tasks ,when reaching the MPI_Barrier ,blocks until all tasks in the group reach the same MPI_Barrier 
  }

  // fill a vertex array with a specific value
  template<typename T>
  void fill_vertex_array(T * array, T value) {
    #pragma omp parallel for
    for (VertexId v_i=partition_offset[partition_id];v_i<partition_offset[partition_id+1];v_i++) {
      array[v_i] = value;
    }
  }

  // allocate a numa-aware vertex array
  template<typename T>
  T * alloc_vertex_array() {
    char * array = (char *)mmap(NULL, sizeof(T) * vertices, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0); //�������ݴ��ڲ�ͬ��numa�ڵ��ϣ��˹���ʽʵ��interleave
    assert(array!=NULL);
    for (int s_i=0;s_i<sockets;s_i++) {
      numa_tonode_memory(array + sizeof(T) * local_partition_offset[s_i], sizeof(T) * (local_partition_offset[s_i+1] - local_partition_offset[s_i]), s_i);
    }
    return (T*)array;
  }

  // deallocate a vertex array
  template<typename T>
  T * dealloc_vertex_array(T * array) {
    numa_free(array, sizeof(T) * vertices);
  }

  // allocate a numa-oblivious vertex array
  template<typename T>
  T * alloc_interleaved_vertex_array() {
    T * array = (T *)numa_alloc_interleaved( sizeof(T) * vertices ); //allocate size bytes of memory page interleaved on all nodes .On error ,NULL is returned. refer to https://linux.die.net/man/3/numa
    assert(array!=NULL);
    return array;
  }

  // dump a vertex array to path
  template<typename T>
  void dump_vertex_array(T * array, std::string path) {
    long file_length = sizeof(T) * vertices;
    if (!file_exists(path) || file_size(path) != file_length) {
      if (partition_id==0) {
        FILE * fout = fopen(path.c_str(), "wb");
        char * buffer = new char [PAGESIZE];
        for (long offset=0;offset<file_length;) {
          if (file_length - offset >= PAGESIZE) {
            fwrite(buffer, 1, PAGESIZE, fout);
            offset += PAGESIZE;
          } else {
            fwrite(buffer, 1, file_length - offset, fout);
            offset += file_length - offset;
          }
        }
        fclose(fout);
      }
      MPI_Barrier(MPI_COMM_WORLD);
    }
    int fd = open(path.c_str(), O_RDWR);
    assert(fd!=-1);
    long offset = sizeof(T) * partition_offset[partition_id];
    long end_offset = sizeof(T) * partition_offset[partition_id+1];
    void * data = (void *)array;
    assert(lseek(fd, offset, SEEK_SET)!=-1);
    while (offset < end_offset) {
      long bytes = write(fd, data + offset, end_offset - offset);
      assert(bytes!=-1);
      offset += bytes;
    }
    assert(close(fd)==0);
  }

  // restore a vertex array from path
  template<typename T>
  void restore_vertex_array(T * array, std::string path) {
    long file_length = sizeof(T) * vertices;
    if (!file_exists(path) || file_size(path) != file_length) {
      assert(false);
    }
    int fd = open(path.c_str(), O_RDWR);
    assert(fd!=-1);
    long offset = sizeof(T) * partition_offset[partition_id];
    long end_offset = sizeof(T) * partition_offset[partition_id+1];
    void * data = (void *)array;
    assert(lseek(fd, offset, SEEK_SET)!=-1);
    while (offset < end_offset) {
      long bytes = read(fd, data + offset, end_offset - offset);
      assert(bytes!=-1);
      offset += bytes;
    }
    assert(close(fd)==0);
  }

  // gather a vertex array
  template<typename T>
  void gather_vertex_array(T * array, int root) {
    if (partition_id!=root) {
      MPI_Send(array + partition_offset[partition_id], sizeof(T) * owned_vertices, MPI_CHAR, root, GatherVertexArray, MPI_COMM_WORLD);
    } else {
      for (int i=0;i<partitions;i++) {
        if (i==partition_id) continue;
        MPI_Status recv_status;
        MPI_Recv(array + partition_offset[i], sizeof(T) * (partition_offset[i + 1] - partition_offset[i]), MPI_CHAR, i, GatherVertexArray, MPI_COMM_WORLD, &recv_status);
        int length;
        MPI_Get_count(&recv_status, MPI_CHAR, &length);
        assert(length == sizeof(T) * (partition_offset[i + 1] - partition_offset[i]));
      }
    }
  }

  // allocate a vertex subset
  VertexSubset * alloc_vertex_subset() {
    return new VertexSubset(vertices);
  }

  int get_partition_id(VertexId v_i){
    for (int i=0;i<partitions;i++) {
      if (v_i >= partition_offset[i] && v_i < partition_offset[i+1]) {
        return i;
      }
    }
    assert(false);
  }

  //�õ���������numa��֪������sub-partition
  int get_local_partition_id(VertexId v_i){
    for (int s_i=0;s_i<sockets;s_i++) {
      if (v_i >= local_partition_offset[s_i] && v_i < local_partition_offset[s_i+1]) {
        return s_i;
      }
    }
    assert(false);
  }

  // load a directed graph and make it undirected
  void load_undirected_from_directed(std::string path, VertexId vertices) {
    double prep_time = 0;
    prep_time -= MPI_Wtime();

    symmetric = true;

    MPI_Datatype vid_t = get_mpi_data_type<VertexId>();

    this->vertices = vertices;
    long total_bytes = file_size(path.c_str());
    this->edges = total_bytes / edge_unit_size;
    #ifdef PRINT_DEBUG_MESSAGES
    if (partition_id==0) {
      printf("|V| = %u, |E| = %lu\n", vertices, edges);
    }
    #endif

    EdgeId read_edges = edges / partitions;
    if (partition_id==partitions-1) {
      read_edges += edges % partitions;
    }
    long bytes_to_read = edge_unit_size * read_edges;
    long read_offset = edge_unit_size * (edges / partitions * partition_id);
    long read_bytes;
    int fin = open(path.c_str(), O_RDONLY);
    EdgeUnit<EdgeData> * read_edge_buffer = new EdgeUnit<EdgeData> [CHUNKSIZE];

    out_degree = alloc_interleaved_vertex_array<VertexId>();
    for (VertexId v_i=0;v_i<vertices;v_i++) {
      out_degree[v_i] = 0;
    }
    assert(lseek(fin, read_offset, SEEK_SET)==read_offset);
    read_bytes = 0;
    while (read_bytes < bytes_to_read) {
      long curr_read_bytes;
      if (bytes_to_read - read_bytes > edge_unit_size * CHUNKSIZE) {
        curr_read_bytes = read(fin, read_edge_buffer, edge_unit_size * CHUNKSIZE);
      } else {
        curr_read_bytes = read(fin, read_edge_buffer, bytes_to_read - read_bytes);
      }
      assert(curr_read_bytes>=0);
      read_bytes += curr_read_bytes;
      EdgeId curr_read_edges = curr_read_bytes / edge_unit_size;
      // #pragma omp parallel for
      for (EdgeId e_i=0;e_i<curr_read_edges;e_i++) {
        VertexId src = read_edge_buffer[e_i].src;
        VertexId dst = read_edge_buffer[e_i].dst;
        __sync_fetch_and_add(&out_degree[src], 1);
        __sync_fetch_and_add(&out_degree[dst], 1);
      }
    }
    MPI_Allreduce(MPI_IN_PLACE, out_degree, vertices, vid_t, MPI_SUM, MPI_COMM_WORLD);

    // locality-aware chunking
    partition_offset = new VertexId [partitions + 1];
    partition_offset[0] = 0;
    EdgeId remained_amount = edges * 2 + EdgeId(vertices) * alpha;  // alpha = 8 * (partitions - 1);
    for (int i=0;i<partitions;i++) {
      VertexId remained_partitions = partitions - i;
      EdgeId expected_chunk_size = remained_amount / remained_partitions;
      if (remained_partitions==1) {
        partition_offset[i+1] = vertices;
      } else {
        EdgeId got_edges = 0;
        for (VertexId v_i=partition_offset[i];v_i<vertices;v_i++) {
          got_edges += out_degree[v_i] + alpha;
          if (got_edges > expected_chunk_size) {
            partition_offset[i+1] = v_i;
            break;
          }
        }
        partition_offset[i+1] = (partition_offset[i+1]) / PAGESIZE * PAGESIZE; // aligned with pages
      }
      for (VertexId v_i=partition_offset[i];v_i<partition_offset[i+1];v_i++) {
        remained_amount -= out_degree[v_i] + alpha;
      }
    }
    assert(partition_offset[partitions]==vertices);
    owned_vertices = partition_offset[partition_id+1] - partition_offset[partition_id];
    // check consistency of partition boundaries
    VertexId * global_partition_offset = new VertexId [partitions + 1];
    MPI_Allreduce(partition_offset, global_partition_offset, partitions + 1, vid_t, MPI_MAX, MPI_COMM_WORLD);
    for (int i=0;i<=partitions;i++) {
      assert(partition_offset[i] == global_partition_offset[i]);
    }
    MPI_Allreduce(partition_offset, global_partition_offset, partitions + 1, vid_t, MPI_MIN, MPI_COMM_WORLD);
    for (int i=0;i<=partitions;i++) {
      assert(partition_offset[i] == global_partition_offset[i]);
    }
    #ifdef PRINT_DEBUG_MESSAGES
    if (partition_id==0) {
      for (int i=0;i<partitions;i++) {
        EdgeId part_out_edges = 0;
        for (VertexId v_i=partition_offset[i];v_i<partition_offset[i+1];v_i++) {
          part_out_edges += out_degree[v_i];
        }
        printf("|V'_%d| = %u |E_%d| = %lu\n", i, partition_offset[i+1] - partition_offset[i], i, part_out_edges);
      }
    }
    MPI_Barrier(MPI_COMM_WORLD);
    #endif
    delete [] global_partition_offset;
    {
      // NUMA-aware sub-chunking
      local_partition_offset = new VertexId [sockets + 1];
      EdgeId part_out_edges = 0;
      for (VertexId v_i=partition_offset[partition_id];v_i<partition_offset[partition_id+1];v_i++) {
        part_out_edges += out_degree[v_i];
      }
      local_partition_offset[0] = partition_offset[partition_id];
      EdgeId remained_amount = part_out_edges + EdgeId(owned_vertices) * alpha;
      for (int s_i=0;s_i<sockets;s_i++) {
        VertexId remained_partitions = sockets - s_i;
        EdgeId expected_chunk_size = remained_amount / remained_partitions;
        if (remained_partitions==1) {
          local_partition_offset[s_i+1] = partition_offset[partition_id+1];
        } else {
          EdgeId got_edges = 0;
          for (VertexId v_i=local_partition_offset[s_i];v_i<partition_offset[partition_id+1];v_i++) {
            got_edges += out_degree[v_i] + alpha;
            if (got_edges > expected_chunk_size) {
              local_partition_offset[s_i+1] = v_i;
              break;
            }
          }
          local_partition_offset[s_i+1] = (local_partition_offset[s_i+1]) / PAGESIZE * PAGESIZE; // aligned with pages
        }
        EdgeId sub_part_out_edges = 0;
        for (VertexId v_i=local_partition_offset[s_i];v_i<local_partition_offset[s_i+1];v_i++) {
          remained_amount -= out_degree[v_i] + alpha;
          sub_part_out_edges += out_degree[v_i];
        }
        #ifdef PRINT_DEBUG_MESSAGES
        printf("|V'_%d_%d| = %u |E_%d| = %lu\n", partition_id, s_i, local_partition_offset[s_i+1] - local_partition_offset[s_i], partition_id, sub_part_out_edges);
        #endif
      }
    }

    VertexId * filtered_out_degree = alloc_vertex_array<VertexId>();
    for (VertexId v_i=partition_offset[partition_id];v_i<partition_offset[partition_id+1];v_i++) {
      filtered_out_degree[v_i] = out_degree[v_i];
    }
    numa_free(out_degree, sizeof(VertexId) * vertices);
    out_degree = filtered_out_degree;
    in_degree = out_degree;

    int * buffered_edges = new int [partitions];
    std::vector<char> * send_buffer = new std::vector<char> [partitions];
    for (int i=0;i<partitions;i++) {
      send_buffer[i].resize(edge_unit_size * CHUNKSIZE);
    }
    EdgeUnit<EdgeData> * recv_buffer = new EdgeUnit<EdgeData> [CHUNKSIZE];

    // constructing symmetric edges
    EdgeId recv_outgoing_edges = 0;
    outgoing_edges = new EdgeId [sockets];
    outgoing_adj_index = new EdgeId* [sockets];
    outgoing_adj_list = new AdjUnit<EdgeData>* [sockets];
    outgoing_adj_bitmap = new Bitmap * [sockets];
    for (int s_i=0;s_i<sockets;s_i++) {
      outgoing_adj_bitmap[s_i] = new Bitmap (vertices);
      outgoing_adj_bitmap[s_i]->clear();
      outgoing_adj_index[s_i] = (EdgeId*)numa_alloc_onnode(sizeof(EdgeId) * (vertices+1), s_i);
    }
    {
      std::thread recv_thread_dst([&](){
        int finished_count = 0;
        MPI_Status recv_status;
        while (finished_count < partitions) {
          MPI_Probe(MPI_ANY_SOURCE, ShuffleGraph, MPI_COMM_WORLD, &recv_status);
          int i = recv_status.MPI_SOURCE;
          assert(recv_status.MPI_TAG == ShuffleGraph && i >=0 && i < partitions);
          int recv_bytes;
          MPI_Get_count(&recv_status, MPI_CHAR, &recv_bytes);
          if (recv_bytes==1) {
            finished_count += 1;
            char c;
            MPI_Recv(&c, 1, MPI_CHAR, i, ShuffleGraph, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            continue;
          }
          assert(recv_bytes % edge_unit_size == 0);
          int recv_edges = recv_bytes / edge_unit_size;
          MPI_Recv(recv_buffer, edge_unit_size * recv_edges, MPI_CHAR, i, ShuffleGraph, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
          // #pragma omp parallel for
          for (EdgeId e_i=0;e_i<recv_edges;e_i++) {
            VertexId src = recv_buffer[e_i].src;
            VertexId dst = recv_buffer[e_i].dst;
            assert(dst >= partition_offset[partition_id] && dst < partition_offset[partition_id+1]);
            int dst_part = get_local_partition_id(dst);
            if (!outgoing_adj_bitmap[dst_part]->get_bit(src)) {
              outgoing_adj_bitmap[dst_part]->set_bit(src);
              outgoing_adj_index[dst_part][src] = 0;
            }
            __sync_fetch_and_add(&outgoing_adj_index[dst_part][src], 1);
          }
          recv_outgoing_edges += recv_edges;
        }
      });
      for (int i=0;i<partitions;i++) {
        buffered_edges[i] = 0;
      }
      assert(lseek(fin, read_offset, SEEK_SET)==read_offset);
      read_bytes = 0;
      while (read_bytes < bytes_to_read) {
        long curr_read_bytes;
        if (bytes_to_read - read_bytes > edge_unit_size * CHUNKSIZE) {
          curr_read_bytes = read(fin, read_edge_buffer, edge_unit_size * CHUNKSIZE);
        } else {
          curr_read_bytes = read(fin, read_edge_buffer, bytes_to_read - read_bytes);
        }
        assert(curr_read_bytes>=0);
        read_bytes += curr_read_bytes;
        EdgeId curr_read_edges = curr_read_bytes / edge_unit_size;
        for (EdgeId e_i=0;e_i<curr_read_edges;e_i++) {
          VertexId dst = read_edge_buffer[e_i].dst;
          int i = get_partition_id(dst);
          memcpy(send_buffer[i].data() + edge_unit_size * buffered_edges[i], &read_edge_buffer[e_i], edge_unit_size);
          buffered_edges[i] += 1;
          if (buffered_edges[i] == CHUNKSIZE) {
            MPI_Send(send_buffer[i].data(), edge_unit_size * buffered_edges[i], MPI_CHAR, i, ShuffleGraph, MPI_COMM_WORLD);
            buffered_edges[i] = 0;
          }
        }
        for (EdgeId e_i=0;e_i<curr_read_edges;e_i++) {
          // std::swap(read_edge_buffer[e_i].src, read_edge_buffer[e_i].dst);
          VertexId tmp = read_edge_buffer[e_i].src;
          read_edge_buffer[e_i].src = read_edge_buffer[e_i].dst;
          read_edge_buffer[e_i].dst = tmp;
        }
        for (EdgeId e_i=0;e_i<curr_read_edges;e_i++) {
          VertexId dst = read_edge_buffer[e_i].dst;
          int i = get_partition_id(dst);
          memcpy(send_buffer[i].data() + edge_unit_size * buffered_edges[i], &read_edge_buffer[e_i], edge_unit_size);
          buffered_edges[i] += 1;
          if (buffered_edges[i] == CHUNKSIZE) {
            MPI_Send(send_buffer[i].data(), edge_unit_size * buffered_edges[i], MPI_CHAR, i, ShuffleGraph, MPI_COMM_WORLD);
            buffered_edges[i] = 0;
          }
        }
      }
      for (int i=0;i<partitions;i++) {
        if (buffered_edges[i]==0) continue;
        MPI_Send(send_buffer[i].data(), edge_unit_size * buffered_edges[i], MPI_CHAR, i, ShuffleGraph, MPI_COMM_WORLD);
        buffered_edges[i] = 0;
      }
      for (int i=0;i<partitions;i++) {
        char c = 0;
        MPI_Send(&c, 1, MPI_CHAR, i, ShuffleGraph, MPI_COMM_WORLD);
      }
      recv_thread_dst.join();
      #ifdef PRINT_DEBUG_MESSAGES
      printf("machine(%d) got %lu symmetric edges\n", partition_id, recv_outgoing_edges);
      #endif
    }
    compressed_outgoing_adj_vertices = new VertexId [sockets];
    compressed_outgoing_adj_index = new CompressedAdjIndexUnit * [sockets];
    for (int s_i=0;s_i<sockets;s_i++) {
      outgoing_edges[s_i] = 0;
      compressed_outgoing_adj_vertices[s_i] = 0;
      for (VertexId v_i=0;v_i<vertices;v_i++) {
        if (outgoing_adj_bitmap[s_i]->get_bit(v_i)) {
          outgoing_edges[s_i] += outgoing_adj_index[s_i][v_i];
          compressed_outgoing_adj_vertices[s_i] += 1;
        }
      }
      compressed_outgoing_adj_index[s_i] = (CompressedAdjIndexUnit*)numa_alloc_onnode( sizeof(CompressedAdjIndexUnit) * (compressed_outgoing_adj_vertices[s_i] + 1) , s_i );
      compressed_outgoing_adj_index[s_i][0].index = 0;
      EdgeId last_e_i = 0;
      compressed_outgoing_adj_vertices[s_i] = 0;
      for (VertexId v_i=0;v_i<vertices;v_i++) {
        if (outgoing_adj_bitmap[s_i]->get_bit(v_i)) {
          outgoing_adj_index[s_i][v_i] = last_e_i + outgoing_adj_index[s_i][v_i];
          last_e_i = outgoing_adj_index[s_i][v_i];
          compressed_outgoing_adj_index[s_i][compressed_outgoing_adj_vertices[s_i]].vertex = v_i;
          compressed_outgoing_adj_vertices[s_i] += 1;
          compressed_outgoing_adj_index[s_i][compressed_outgoing_adj_vertices[s_i]].index = last_e_i;
        }
      }
      for (VertexId p_v_i=0;p_v_i<compressed_outgoing_adj_vertices[s_i];p_v_i++) {
        VertexId v_i = compressed_outgoing_adj_index[s_i][p_v_i].vertex;
        outgoing_adj_index[s_i][v_i] = compressed_outgoing_adj_index[s_i][p_v_i].index;
        outgoing_adj_index[s_i][v_i+1] = compressed_outgoing_adj_index[s_i][p_v_i+1].index;
      }
      #ifdef PRINT_DEBUG_MESSAGES
      printf("part(%d) E_%d has %lu symmetric edges\n", partition_id, s_i, outgoing_edges[s_i]);
      #endif
      outgoing_adj_list[s_i] = (AdjUnit<EdgeData>*)numa_alloc_onnode(unit_size * outgoing_edges[s_i], s_i);
    }
    {
      std::thread recv_thread_dst([&](){
        int finished_count = 0;
        MPI_Status recv_status;
        while (finished_count < partitions) {
          MPI_Probe(MPI_ANY_SOURCE, ShuffleGraph, MPI_COMM_WORLD, &recv_status);
          int i = recv_status.MPI_SOURCE;
          assert(recv_status.MPI_TAG == ShuffleGraph && i >=0 && i < partitions);
          int recv_bytes;
          MPI_Get_count(&recv_status, MPI_CHAR, &recv_bytes);
          if (recv_bytes==1) {
            finished_count += 1;
            char c;
            MPI_Recv(&c, 1, MPI_CHAR, i, ShuffleGraph, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            continue;
          }
          assert(recv_bytes % edge_unit_size == 0);
          int recv_edges = recv_bytes / edge_unit_size;
          MPI_Recv(recv_buffer, edge_unit_size * recv_edges, MPI_CHAR, i, ShuffleGraph, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
          #pragma omp parallel for
          for (EdgeId e_i=0;e_i<recv_edges;e_i++) {
            VertexId src = recv_buffer[e_i].src;
            VertexId dst = recv_buffer[e_i].dst;
            assert(dst >= partition_offset[partition_id] && dst < partition_offset[partition_id+1]);
            int dst_part = get_local_partition_id(dst);
            EdgeId pos = __sync_fetch_and_add(&outgoing_adj_index[dst_part][src], 1);
            outgoing_adj_list[dst_part][pos].neighbour = dst;
            if (!std::is_same<EdgeData, Empty>::value) {
              outgoing_adj_list[dst_part][pos].edge_data = recv_buffer[e_i].edge_data;
            }
          }
        }
      });
      for (int i=0;i<partitions;i++) {
        buffered_edges[i] = 0;
      }
      assert(lseek(fin, read_offset, SEEK_SET)==read_offset);
      read_bytes = 0;
      while (read_bytes < bytes_to_read) {
        long curr_read_bytes;
        if (bytes_to_read - read_bytes > edge_unit_size * CHUNKSIZE) {
          curr_read_bytes = read(fin, read_edge_buffer, edge_unit_size * CHUNKSIZE);
        } else {
          curr_read_bytes = read(fin, read_edge_buffer, bytes_to_read - read_bytes);
        }
        assert(curr_read_bytes>=0);
        read_bytes += curr_read_bytes;
        EdgeId curr_read_edges = curr_read_bytes / edge_unit_size;
        for (EdgeId e_i=0;e_i<curr_read_edges;e_i++) {
          VertexId dst = read_edge_buffer[e_i].dst;
          int i = get_partition_id(dst);
          memcpy(send_buffer[i].data() + edge_unit_size * buffered_edges[i], &read_edge_buffer[e_i], edge_unit_size);
          buffered_edges[i] += 1;
          if (buffered_edges[i] == CHUNKSIZE) {
            MPI_Send(send_buffer[i].data(), edge_unit_size * buffered_edges[i], MPI_CHAR, i, ShuffleGraph, MPI_COMM_WORLD);
            buffered_edges[i] = 0;
          }
        }
        for (EdgeId e_i=0;e_i<curr_read_edges;e_i++) {
          // std::swap(read_edge_buffer[e_i].src, read_edge_buffer[e_i].dst);
          VertexId tmp = read_edge_buffer[e_i].src;
          read_edge_buffer[e_i].src = read_edge_buffer[e_i].dst;
          read_edge_buffer[e_i].dst = tmp;
        }
        for (EdgeId e_i=0;e_i<curr_read_edges;e_i++) {
          VertexId dst = read_edge_buffer[e_i].dst;
          int i = get_partition_id(dst);
          memcpy(send_buffer[i].data() + edge_unit_size * buffered_edges[i], &read_edge_buffer[e_i], edge_unit_size);
          buffered_edges[i] += 1;
          if (buffered_edges[i] == CHUNKSIZE) {
            MPI_Send(send_buffer[i].data(), edge_unit_size * buffered_edges[i], MPI_CHAR, i, ShuffleGraph, MPI_COMM_WORLD);
            buffered_edges[i] = 0;
          }
        }
      }
      for (int i=0;i<partitions;i++) {
        if (buffered_edges[i]==0) continue;
        MPI_Send(send_buffer[i].data(), edge_unit_size * buffered_edges[i], MPI_CHAR, i, ShuffleGraph, MPI_COMM_WORLD);
        buffered_edges[i] = 0;
      }
      for (int i=0;i<partitions;i++) {
        char c = 0;
        MPI_Send(&c, 1, MPI_CHAR, i, ShuffleGraph, MPI_COMM_WORLD);
      }
      recv_thread_dst.join();
    }
    for (int s_i=0;s_i<sockets;s_i++) {
      for (VertexId p_v_i=0;p_v_i<compressed_outgoing_adj_vertices[s_i];p_v_i++) {
        VertexId v_i = compressed_outgoing_adj_index[s_i][p_v_i].vertex;
        outgoing_adj_index[s_i][v_i] = compressed_outgoing_adj_index[s_i][p_v_i].index;
        outgoing_adj_index[s_i][v_i+1] = compressed_outgoing_adj_index[s_i][p_v_i+1].index;
      }
    }
    MPI_Barrier(MPI_COMM_WORLD);

    incoming_edges = outgoing_edges;
    incoming_adj_index = outgoing_adj_index;
    incoming_adj_list = outgoing_adj_list;
    incoming_adj_bitmap = outgoing_adj_bitmap;
    compressed_incoming_adj_vertices = compressed_outgoing_adj_vertices;
    compressed_incoming_adj_index = compressed_outgoing_adj_index;
    MPI_Barrier(MPI_COMM_WORLD);

    delete [] buffered_edges;
    delete [] send_buffer;
    delete [] read_edge_buffer;
    delete [] recv_buffer;
    close(fin);

    tune_chunks();
    tuned_chunks_sparse = tuned_chunks_dense;

    prep_time += MPI_Wtime();

    #ifdef PRINT_DEBUG_MESSAGES
    if (partition_id==0) {
      printf("preprocessing cost: %.2lf (s)\n", prep_time);
    }
    #endif
  }

  // transpose the graph
  void transpose() {
    std::swap(out_degree, in_degree);
    std::swap(outgoing_edges, incoming_edges);
    std::swap(outgoing_adj_index, incoming_adj_index);
    std::swap(outgoing_adj_bitmap, incoming_adj_bitmap);
    std::swap(outgoing_adj_list, incoming_adj_list);
    std::swap(tuned_chunks_dense, tuned_chunks_sparse); // dense �� parse
    std::swap(compressed_outgoing_adj_vertices, compressed_incoming_adj_vertices);
    std::swap(compressed_outgoing_adj_index, compressed_incoming_adj_index);
  }

  // load a directed graph from path
  void load_directed(std::string path, VertexId vertices) {
   /*
   1. �õ�path������ͼ��ȫ����Ϣ���������������ֽ�����������
   */
    double prep_time = 0;
    prep_time -= MPI_Wtime(); // Returns an elapsed time on the calling processor

    symmetric = false;

    MPI_Datatype vid_t = get_mpi_data_type<VertexId>(); //vertex type

    this->vertices = vertices;
    long total_bytes = file_size(path.c_str());
    this->edges = total_bytes / edge_unit_size;
    #ifdef PRINT_DEBUG_MESSAGES
    if (partition_id==0) {
      printf("|V| = %u, |E| = %lu\n", vertices, edges);
    }
    #endif

	/*
	2.�õ���ǰ������Ҫ�����������(bytes_to_read)��λ����Ϣ(read_offset),Ȼ���path��read_offsetλ�ÿ�ʼ��ȡbytes_to_read�ֽ����ݵ�read_edge_buffer�����С�
	�ڶ��Ĺ�����ͬʱͳ�ƶ���ĳ�����Ϣ������out_degree�У�����ͨ��reduce������ʹout_degree�б��������ж���ĳ�����Ϣ��
	����IO:1��Դ�ļ���ȡ��ÿ̨����ֻ��ȡ�������ݣ�
	*/
    EdgeId read_edges = edges / partitions;
    if (partition_id==partitions-1) {
      read_edges += edges % partitions;
    }
    long bytes_to_read = edge_unit_size * read_edges;
    long read_offset = edge_unit_size * (edges / partitions * partition_id);
    long read_bytes;
    int fin = open(path.c_str(), O_RDONLY);
    EdgeUnit<EdgeData> * read_edge_buffer = new EdgeUnit<EdgeData> [CHUNKSIZE];

    out_degree = alloc_interleaved_vertex_array<VertexId>();
    for (VertexId v_i=0;v_i<vertices;v_i++) {
      out_degree[v_i] = 0;
    }
    assert(lseek(fin, read_offset, SEEK_SET)==read_offset);
    read_bytes = 0;
    while (read_bytes < bytes_to_read) {
      long curr_read_bytes;
      if (bytes_to_read - read_bytes > edge_unit_size * CHUNKSIZE) {
        curr_read_bytes = read(fin, read_edge_buffer, edge_unit_size * CHUNKSIZE);
      } else {
        curr_read_bytes = read(fin, read_edge_buffer, bytes_to_read - read_bytes);
      }
      assert(curr_read_bytes>=0);
      read_bytes += curr_read_bytes;
      EdgeId curr_read_edges = curr_read_bytes / edge_unit_size;
      #pragma omp parallel for
      for (EdgeId e_i=0;e_i<curr_read_edges;e_i++) {
        VertexId src = read_edge_buffer[e_i].src;
        VertexId dst = read_edge_buffer[e_i].dst;
        __sync_fetch_and_add(&out_degree[src], 1);
      }
    }
    MPI_Allreduce(MPI_IN_PLACE, out_degree, vertices, vid_t, MPI_SUM, MPI_COMM_WORLD); //������processor�ϵ�out_degree����reduce������sum)��Ȼ���ڷ��������processor��ִ�����out_degree�оͰ��������ж���ĳ�����Ϣ��

    /*
	3. locality-aware chunking ����locality-awawre��ǿ�����õ�ÿ������(processor)��Ҫ����Ķ������䣺
 ����洢��partition_offset�У�����i��Ҫ����Ķ�������Ϊ[ partition_offset[i], partition_offset[i+1] )
	*/
    partition_offset = new VertexId [partitions + 1];
    partition_offset[0] = 0;
    // ÿ��������ƽ�����ӣ�����alpha Ĭ�ϵ��� 8 * (partitions - 1)�����ݲ�ͬ��Ӳ�������������Ҫ������
    EdgeId remained_amount = edges + EdgeId(vertices) * alpha;  
    for (int i=0;i<partitions;i++) {
      VertexId remained_partitions = partitions - i;
      EdgeId expected_chunk_size = remained_amount / remained_partitions;
      if (remained_partitions==1) {
        partition_offset[i+1] = vertices;
      } else {
        EdgeId got_edges = 0;
        for (VertexId v_i=partition_offset[i];v_i<vertices;v_i++) {
          got_edges += out_degree[v_i] + alpha;
          if (got_edges > expected_chunk_size) {
            partition_offset[i+1] = v_i;
            break;
          }
        }
        partition_offset[i+1] = (partition_offset[i+1]) / PAGESIZE * PAGESIZE; // aligned with pages ҳ����
      }
      for (VertexId v_i=partition_offset[i];v_i<partition_offset[i+1];v_i++) {
        remained_amount -= out_degree[v_i] + alpha;
      }
    }
    assert(partition_offset[partitions]==vertices);
    owned_vertices = partition_offset[partition_id+1] - partition_offset[partition_id];
    // check consistency of partition boundaries
    VertexId * global_partition_offset = new VertexId [partitions + 1];
    MPI_Allreduce(partition_offset, global_partition_offset, partitions + 1, vid_t, MPI_MAX, MPI_COMM_WORLD); // reduce����Ϊ��max),���ս���洢��global_partition_offset
    for (int i=0;i<=partitions;i++) {
      assert(partition_offset[i] == global_partition_offset[i]);
    }
    MPI_Allreduce(partition_offset, global_partition_offset, partitions + 1, vid_t, MPI_MIN, MPI_COMM_WORLD);
    for (int i=0;i<=partitions;i++) {
      assert(partition_offset[i] == global_partition_offset[i]);
    }
    #ifdef PRINT_DEBUG_MESSAGES
    if (partition_id==0) {
      for (int i=0;i<partitions;i++) {
        EdgeId part_out_edges = 0;
        for (VertexId v_i=partition_offset[i];v_i<partition_offset[i+1];v_i++) {
          part_out_edges += out_degree[v_i];
        }
        printf("|V'_%d| = %u |E^dense_%d| = %lu\n", i, partition_offset[i+1] - partition_offset[i], i, part_out_edges);
      }
    }
    #endif
    delete [] global_partition_offset;
    {
      /*
	  4. NUMA-aware sub-chunking  �õ�ÿ��numa�ڵ���Ҫ����Ķ������䣬����洢��local_partition_offset�У�
   numa�ڵ�i��Ҫ����Ķ�������Ϊ[ local_partition_offset[i], local_partition_offset[i+1] )
	   * ʵ��out_degree���� numa-aware
	  */
      local_partition_offset = new VertexId [sockets + 1];  //���ÿ��num�ڵ���Ҫ����Ķ�������
      EdgeId part_out_edges = 0;   //��ǰ���������ĳ�����
      for (VertexId v_i=partition_offset[partition_id];v_i<partition_offset[partition_id+1];v_i++) {
        part_out_edges += out_degree[v_i];
      }
      local_partition_offset[0] = partition_offset[partition_id];
      EdgeId remained_amount = part_out_edges + EdgeId(owned_vertices) * alpha;
      for (int s_i=0;s_i<sockets;s_i++) {
        VertexId remained_partitions = sockets - s_i;
        EdgeId expected_chunk_size = remained_amount / remained_partitions;
        if (remained_partitions==1) {
          local_partition_offset[s_i+1] = partition_offset[partition_id+1];
        } else {
          EdgeId got_edges = 0;
          for (VertexId v_i=local_partition_offset[s_i];v_i<partition_offset[partition_id+1];v_i++) {
            got_edges += out_degree[v_i] + alpha;
            if (got_edges > expected_chunk_size) {
              local_partition_offset[s_i+1] = v_i;
              break;
            }
          }
          local_partition_offset[s_i+1] = (local_partition_offset[s_i+1]) / PAGESIZE * PAGESIZE; // aligned with pages
        }
        EdgeId sub_part_out_edges = 0;
        for (VertexId v_i=local_partition_offset[s_i];v_i<local_partition_offset[s_i+1];v_i++) {
          remained_amount -= out_degree[v_i] + alpha;
          sub_part_out_edges += out_degree[v_i];
        }
        #ifdef PRINT_DEBUG_MESSAGES
        printf("|V'_%d_%d| = %u |E^dense_%d_%d| = %lu\n", partition_id, s_i, local_partition_offset[s_i+1] - local_partition_offset[s_i], partition_id, s_i, sub_part_out_edges);
        #endif
      }
    }

	//ʵ��out_degree��numa��֪����
    VertexId * filtered_out_degree = alloc_vertex_array<VertexId>();
    for (VertexId v_i=partition_offset[partition_id];v_i<partition_offset[partition_id+1];v_i++) {
      filtered_out_degree[v_i] = out_degree[v_i];
    }
    numa_free(out_degree, sizeof(VertexId) * vertices);
    out_degree = filtered_out_degree;
    in_degree = alloc_vertex_array<VertexId>();
    for (VertexId v_i=partition_offset[partition_id];v_i<partition_offset[partition_id+1];v_i++) {
      in_degree[v_i] = 0;
    }
	
	/*
	5.����ͼ���ڴ��еĴ洢�ṹDCSR��ʵ��numa-aware�����ɣ�5.1��5.2������ɣ�
	5.1 �ȹ���outgoing_adj_index��compressed_outgoing_adj_index��outgoing_adj_bitmap��
	�漰��shuffle������ÿ���������ļ��ж�ȡ�����ݣ����ձ�dst������������������ͬ�����ı߷��͵���Ӧ�Ļ�����producer�� 
	ÿ���������յ�������<src,dst>��,�ڸ���dst����numa�ڵ㣬���ߵĴ��������벻ͬ��numa�ڵ��ϣ�
	����outgoing_adj_bitmap��outgoing_adj_bitmap[partition_of_dst]��Ƕ���src�Ƿ��г��ߣ�
	outgoing_adj_index��outgoing_adj_index[partition_of_dst]��Ƕ���src�����ĳ��߸�����
	in_degree[dst]��dst������ȡ�Ȼ���ڽ���ѹ��������compressed_outgoing_adj_vertices��compressed_outgoing_adj_index

 ��ס��sparseģʽ�¸��ݳ��߽��и��£�mster�������signal�����������꽫���ݷ��͸�mirror���㣬
 mirror�����յ����ݺ����slot����������**����**�����ھӣ���
 master���㣺���ڵ�ǰ���������Ķ��㡣mirror���㣺master�������������������ϵľ���
 ÿ����������vertices��mirror���㣬��ֻ�в���master���㡣
	����IO:һ���ļ���ȡ��ÿ̨����ֻ��ȡ�������ݣ�
	*/
    int * buffered_edges = new int [partitions]; //��¼��Ӧsend_buffer�бߵ�����
    std::vector<char> * send_buffer = new std::vector<char> [partitions]; // ��ǰ����ӵ��send_buffer[p],��Ӧ�洢����Ҫ���͸�p�������ı����ݣ���buffered_edges���ʹ��ʵ�ֱ����ݵ��������͡����緢�͸�2�����ı��������ﵽCHUNKSIZE��С�����������͹�ȥ��
    for (int i=0;i<partitions;i++) {
      send_buffer[i].resize(edge_unit_size * CHUNKSIZE);
    }
    EdgeUnit<EdgeData> * recv_buffer = new EdgeUnit<EdgeData> [CHUNKSIZE]; //��ʱ�洢��������Ϣ����Ϣ����Ϊ�����ݣ��ߵ�dstΪ��ǰ�����������㡣

    EdgeId recv_outgoing_edges = 0;
    outgoing_edges = new EdgeId [sockets];               //ÿ��numa�ڵ��а����ĳ���������������������һ���ߣ�����ߵ�dst���ڵ�s_i��numa�ڵ�������Ķ������䣬��ôoutgoing_edges[s_i]���1.
    outgoing_adj_index = new EdgeId* [sockets];          //numa�ڵ�(s_i)�ж���src�����ж���[0-vertices]���ĳ�����outgoing_adj_list[s_i]�е�����λ�á���5.1�лṹ��
    outgoing_adj_list = new AdjUnit<EdgeData>* [sockets]; //outgoing_adj_list[s_i]�洢���ж���srcָ��numa�ڵ㣨s_i)�ĳ��ߡ����ߵ�dstΪnuma�ڵ㣨s_i)������Ķ������䡣��5.2�лṹ�죬�乹�������outgoing_adj_index��ָ��ÿ���ߴ�ŵ�λ�á�
    outgoing_adj_bitmap = new Bitmap * [sockets];        //outgoing_adj_bitmap[s_i]��Ƕ���src��numa�ڵ�[s_i]���Ƿ��г��ߡ���5.1�й��졣
    for (int s_i=0;s_i<sockets;s_i++) {
      outgoing_adj_bitmap[s_i] = new Bitmap (vertices);
      outgoing_adj_bitmap[s_i]->clear();
      outgoing_adj_index[s_i] = (EdgeId*)numa_alloc_onnode(sizeof(EdgeId) * (vertices+1), s_i);
    }
    {
	  /*
	  consumer.�����̴߳���ǰ�������յ�����Ϣ�������ݣ�dst���ڵ�ǰ���������������䣩������outgoing_adj_bitmap��
   ��Ƕ���src�Ƿ��г��ߣ�outgoing_adj_index����Ƕ���src�����ĳ��߸�����in_degree��dst�������
      */
	  std::thread recv_thread_dst([&](){
        int finished_count = 0;
        MPI_Status recv_status; //��¼��Ϣ��Source��Tag��Ϣ
        while (finished_count < partitions) {
          MPI_Probe(MPI_ANY_SOURCE, ShuffleGraph, MPI_COMM_WORLD, &recv_status);  //����Ƿ�����Ϣ�������ʽ��������ϢSourceΪ��MPI_ANY_SOURCE����ϢTagΪ��ShuffleGraph�������⵽����Ϣ�����ʹ��recv_status��������Ϣ��������Ϣ��
          int i = recv_status.MPI_SOURCE;
          assert(recv_status.MPI_TAG == ShuffleGraph && i >=0 && i < partitions);
          int recv_bytes;  //�洢��Ϣ�������ֽ���
          MPI_Get_count(&recv_status, MPI_CHAR, &recv_bytes); //��ȡ��Ϣ��������Ϣ��recv_status)���������ͣ�MPI_CHAR)���ֽ�����recv_bytes)��
          if (recv_bytes==1) {  //ÿ��������������Ϣ���ٷ���һ���ֽ���Ϣ��ʾ������ϡ�
            finished_count += 1; 
            char c;
			//MPI_Recv (&buf,count,datatype,source,tag,comm,&status)  ��ȡһ���ֽ���Ϣ����c�У�������Ϣ״̬����ΪMPI_STATUS_IGNORE�����ԣ�������ʽ
            MPI_Recv(&c, 1, MPI_CHAR, i, ShuffleGraph, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            continue;
          }
          assert(recv_bytes % edge_unit_size == 0);
          int recv_edges = recv_bytes / edge_unit_size; //��Ϣ�����ı�����
          MPI_Recv(recv_buffer, edge_unit_size * recv_edges, MPI_CHAR, i, ShuffleGraph, MPI_COMM_WORLD, MPI_STATUS_IGNORE); //��ȡ��Ϣ��recv_buffer��
          // #pragma omp parallel for  ������յ��ı����ݣ�����outgoing_adj_bitmap����Ƕ���src�Ƿ��г��ߣ�
          // outgoing_adj_index����Ƕ���src�����ĳ��߸�����in_degree��dst�������
          for (EdgeId e_i=0;e_i<recv_edges;e_i++) {
            VertexId src = recv_buffer[e_i].src;
            VertexId dst = recv_buffer[e_i].dst;
            assert(dst >= partition_offset[partition_id] && dst < partition_offset[partition_id+1]);  // �ߵ�dst ���ڵ�ǰ��������Ķ�������
            int dst_part = get_local_partition_id(dst);  //dst������ǰ������numa����
            if (!outgoing_adj_bitmap[dst_part]->get_bit(src)) {
              outgoing_adj_bitmap[dst_part]->set_bit(src);
              outgoing_adj_index[dst_part][src] = 0;
            }
            __sync_fetch_and_add(&outgoing_adj_index[dst_part][src], 1);
            __sync_fetch_and_add(&in_degree[dst], 1);
          }
          recv_outgoing_edges += recv_edges;
        }
      });// end thread
      for (int i=0;i<partitions;i++) {
        buffered_edges[i] = 0;
      }
	  /*
	  producer:��ʼ��path�ж�ȡ��ǰ�����������������Ӧ��������ݣ�һ�ζ�ȡCHUNKSIZE�������ݣ�Ȼ�����dst�������䣨��������
   ��������Ϣ����ʽ���͵���Ӧ�Ļ����ϣ�Ϊ����߷���Ч��ͨ��send_buffer��buffered_edgesʵ��������CHUNKSIZE������
      
	  */
	  assert(lseek(fin, read_offset, SEEK_SET)==read_offset);
      read_bytes = 0;
      while (read_bytes < bytes_to_read) {
        long curr_read_bytes;
        if (bytes_to_read - read_bytes > edge_unit_size * CHUNKSIZE) {
          curr_read_bytes = read(fin, read_edge_buffer, edge_unit_size * CHUNKSIZE);
        } else {
          curr_read_bytes = read(fin, read_edge_buffer, bytes_to_read - read_bytes);
        }
        assert(curr_read_bytes>=0);
        read_bytes += curr_read_bytes;
        EdgeId curr_read_edges = curr_read_bytes / edge_unit_size;
        for (EdgeId e_i=0;e_i<curr_read_edges;e_i++) {
          VertexId dst = read_edge_buffer[e_i].dst; //�ߵ�Ŀ�Ķ�������������������ķ���
          int i = get_partition_id(dst);   //dst���������ķ���������������
          memcpy(send_buffer[i].data() + edge_unit_size * buffered_edges[i], &read_edge_buffer[e_i], edge_unit_size);  
          buffered_edges[i] += 1;
          if (buffered_edges[i] == CHUNKSIZE) { // ������͸�i��������ݴﵽCHUNKSIZE��С�����������͹�ȥ
			  //MPI_Ssend (&buf,count,datatype,dest,tag,comm)  ��buf�е�count���ֽ����ݷ��͵� i�����ϣ���������Ϊdatatype
            MPI_Send(send_buffer[i].data(), edge_unit_size * buffered_edges[i], MPI_CHAR, i, ShuffleGraph, MPI_COMM_WORLD); //�������ݷ��͵�
            buffered_edges[i] = 0;
          }
        }
      }
      for (int i=0;i<partitions;i++) {  //��������CHUNKSIZE��С�ı߷��ͳ�ȥ
        if (buffered_edges[i]==0) continue;
        MPI_Send(send_buffer[i].data(), edge_unit_size * buffered_edges[i], MPI_CHAR, i, ShuffleGraph, MPI_COMM_WORLD);
        buffered_edges[i] = 0;
      }
      for (int i=0;i<partitions;i++) {  //���ͽ�����־
        char c = 0;
        MPI_Send(&c, 1, MPI_CHAR, i, ShuffleGraph, MPI_COMM_WORLD);
      }
      recv_thread_dst.join();  //������߳�join ���߳�
      #ifdef PRINT_DEBUG_MESSAGES
      printf("machine(%d) got %lu sparse mode edges\n", partition_id, recv_outgoing_edges);
      #endif
    }
    compressed_outgoing_adj_vertices = new VertexId [sockets]; // ÿ��numa�ڵ�ӵ�г��ߵĶ��������
    compressed_outgoing_adj_index = new CompressedAdjIndexUnit * [sockets];//
    for (int s_i=0;s_i<sockets;s_i++) { 
      outgoing_edges[s_i] = 0;
      compressed_outgoing_adj_vertices[s_i] = 0;
	  /*
	  ���ж����У��ڵ�s_i ��numa������ӵ�г��ߵĶ��������������compressed_outgoing_adj_vertices[s_i]�У�
   Ϊ�����compressed_outgoing_adj_index[s_i]�����ڴ�ռ�ʹ�ã�ÿ��numa�ڵ�����ĳ�������������outgoing_edges[s_i]��
	  outgoing_adj_index[s_i]�������ж����ڵ�s_i ��numa������ӵ�еĵĳ���������Ϣ��
	  compressed_outgoing_adj_vertices[s_i]��
	  outgoing_adj_list[s_i]�����ж����ڵ�s_i��numa�����еĳ��ߡ�
	  */
      for (VertexId v_i=0;v_i<vertices;v_i++) { 
        if (outgoing_adj_bitmap[s_i]->get_bit(v_i)) {
          outgoing_edges[s_i] += outgoing_adj_index[s_i][v_i];
          compressed_outgoing_adj_vertices[s_i] += 1;
        }
      }
	  //����ÿ��numa�ڵ��������С ���߶�����+1  ��  CompressedAdjIndexUnit��{ EdgeId index; VertexId vertex; }
      compressed_outgoing_adj_index[s_i] = (CompressedAdjIndexUnit*)numa_alloc_onnode( sizeof(CompressedAdjIndexUnit) * (compressed_outgoing_adj_vertices[s_i] + 1) , s_i );
      compressed_outgoing_adj_index[s_i][0].index = 0;
      EdgeId last_e_i = 0;
      compressed_outgoing_adj_vertices[s_i] = 0;   
	  /*
	  ִ�����outgoing_adj_index[s_i]�����prefix-sums������iλ�ô洢��ǰi+1��Ԫ��֮�ͣ�i��0��ʼ����
	  compressed_outgoing_adj_index[s_i]�洢���г��ߵĶ����Լ����������outgoing_adj_list[s_i]�еĿ�ʼλ�á�
	  ������һ��CompressedAdjIndexUnit����index,vertex)������vertex��outgoing_adj_list�Ŀ�ʼλ��Ϊ��outgoing_adj_list[index],����vertex���г��ߵĶ���
	  */
      for (VertexId v_i=0;v_i<vertices;v_i++) {  
        if (outgoing_adj_bitmap[s_i]->get_bit(v_i)) {
          outgoing_adj_index[s_i][v_i] = last_e_i + outgoing_adj_index[s_i][v_i];
          last_e_i = outgoing_adj_index[s_i][v_i];
          compressed_outgoing_adj_index[s_i][compressed_outgoing_adj_vertices[s_i]].vertex = v_i; //�г��ߵĶ���v_i
          compressed_outgoing_adj_vertices[s_i] += 1;
          compressed_outgoing_adj_index[s_i][compressed_outgoing_adj_vertices[s_i]].index = last_e_i;//��һ������ĳ�����outgoing_adj_list�еĿ�ʼλ�ã�Ҳ�Ƕ���v_i���һ��������outgoing_adj_list�еı߽�λ�á�
        }
      }
	  /*
	  ִ����ɺ�outgoing_adj_index[s_i]����˶��������outgoing_adj_list������λ����Ϣ��
	  ��������i�ĳ�����outgoing_adj_list[s_i]����λ��Ϊ��outgoing_adj_list[s_i][i]��
   ���߸���Ϊcount =outgoing_adj_list[s_i][i+1]-outgoing_adj_list[s_i][i]�����count=0��iû�г��ߡ�
	 
	 */
      for (VertexId p_v_i=0;p_v_i<compressed_outgoing_adj_vertices[s_i];p_v_i++) {
        VertexId v_i = compressed_outgoing_adj_index[s_i][p_v_i].vertex;
        outgoing_adj_index[s_i][v_i] = compressed_outgoing_adj_index[s_i][p_v_i].index;
        outgoing_adj_index[s_i][v_i+1] = compressed_outgoing_adj_index[s_i][p_v_i+1].index;
      }
      #ifdef PRINT_DEBUG_MESSAGES
      printf("part(%d) E_%d has %lu sparse mode edges\n", partition_id, s_i, outgoing_edges[s_i]);//��partition_id���������������ϵĵ�s_i��numa������ӵ��outgoing_edges[s_i]�����ߡ�
      #endif
      outgoing_adj_list[s_i] = (AdjUnit<EdgeData>*)numa_alloc_onnode(unit_size * outgoing_edges[s_i], s_i);
    }
	/*
	5.2.�ٹ���outgoing_adj_list
	����IO:һ���ļ���ȡ��ÿ̨����ֻ��ȡ�������ݣ�
	*/
    {
	   /*
	    consumer.�����̴߳���ǰ�������յ�����Ϣ�������ݣ�dst���ڵ�ǰ���������������䣩��
     ����outgoing_adj_list[dst_part]:�洢dst_part numa�ڵ������г��ߣ��乹����Ҫoutgoing_adj_index[dst_part]��֧�֣�
		��src�����ÿ����Ӧ�÷���outgoing_adj_list[dst_part]���ĸ�λ����outgoing_adj_index[dst_part][src]������
		
		�Խ��յ���ÿ����(src,dst)ִ�����²�����
		1.���ݱ�dst�õ�������numa�ڵ�ţ�dst_part��
		2.��ȡsrc�ıߴ����outgoing_adj_list[dst_part]�е�λ����Ϣ��pos = outgoing_adj_index[dst_part][src]��
		3.�� outgoing_adj_list[dst_part][pos]�д������Ϣ��dst,edgeData)��
		4.outgoing_adj_index[dst_part][src]ֵ��1��Ϊsrc����һ���ߴ���list��λ����׼����2��4����������һ��ԭ�Ӳ�������ɣ�
	   */
      std::thread recv_thread_dst([&](){
        int finished_count = 0;
        MPI_Status recv_status;
        while (finished_count < partitions) {
          MPI_Probe(MPI_ANY_SOURCE, ShuffleGraph, MPI_COMM_WORLD, &recv_status);
          int i = recv_status.MPI_SOURCE;
          assert(recv_status.MPI_TAG == ShuffleGraph && i >=0 && i < partitions);
          int recv_bytes;
          MPI_Get_count(&recv_status, MPI_CHAR, &recv_bytes);
          if (recv_bytes==1) {    // ���Ϊһ���ֽڴ�����һ�����������б߶�ȫ�����ͳ�ȥ�ˣ����finished_count=partitions�������л�����Ϣ��������ϣ���Ϣ�Ĵﵽ��˳��ġ�
            finished_count += 1;
            char c;
            MPI_Recv(&c, 1, MPI_CHAR, i, ShuffleGraph, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            continue;
          }
          assert(recv_bytes % edge_unit_size == 0);
          int recv_edges = recv_bytes / edge_unit_size;
          MPI_Recv(recv_buffer, edge_unit_size * recv_edges, MPI_CHAR, i, ShuffleGraph, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
          #pragma omp parallel for
          for (EdgeId e_i=0;e_i<recv_edges;e_i++) {
            VertexId src = recv_buffer[e_i].src;
            VertexId dst = recv_buffer[e_i].dst;
            assert(dst >= partition_offset[partition_id] && dst < partition_offset[partition_id+1]);
            int dst_part = get_local_partition_id(dst);
            EdgeId pos = __sync_fetch_and_add(&outgoing_adj_index[dst_part][src], 1);// ԭ�Ӳ������ȷ����ڼ�1
            outgoing_adj_list[dst_part][pos].neighbour = dst;
            if (!std::is_same<EdgeData, Empty>::value) {
              outgoing_adj_list[dst_part][pos].edge_data = recv_buffer[e_i].edge_data;
            }
          }
        }
      });

	  /*
	  producer:��ʼ��path�ж�ȡ��ǰ�����������������Ӧ��������ݣ�һ�ζ�ȡCHUNKSIZE�������ݣ�Ȼ�����dst�������䣨��������
	  ��������Ϣ����ʽ���͵���Ӧ�Ļ����ϣ�Ϊ����߷���Ч��ͨ��send_buffer��buffered_edgesʵ����Ϣ������CHUNKSIZE������.
      
	  */
      for (int i=0;i<partitions;i++) {
        buffered_edges[i] = 0;
      }
      assert(lseek(fin, read_offset, SEEK_SET)==read_offset);
      read_bytes = 0;
      while (read_bytes < bytes_to_read) {
        long curr_read_bytes;
        if (bytes_to_read - read_bytes > edge_unit_size * CHUNKSIZE) {
          curr_read_bytes = read(fin, read_edge_buffer, edge_unit_size * CHUNKSIZE);
        } else {
          curr_read_bytes = read(fin, read_edge_buffer, bytes_to_read - read_bytes);
        }
        assert(curr_read_bytes>=0);
        read_bytes += curr_read_bytes;
        EdgeId curr_read_edges = curr_read_bytes / edge_unit_size;
        for (EdgeId e_i=0;e_i<curr_read_edges;e_i++) {
          VertexId dst = read_edge_buffer[e_i].dst;
          int i = get_partition_id(dst);
          memcpy(send_buffer[i].data() + edge_unit_size * buffered_edges[i], &read_edge_buffer[e_i], edge_unit_size);
          buffered_edges[i] += 1;
          if (buffered_edges[i] == CHUNKSIZE) {
            MPI_Send(send_buffer[i].data(), edge_unit_size * buffered_edges[i], MPI_CHAR, i, ShuffleGraph, MPI_COMM_WORLD);
            buffered_edges[i] = 0;
          }
        }
      }
      for (int i=0;i<partitions;i++) {
        if (buffered_edges[i]==0) continue;
        MPI_Send(send_buffer[i].data(), edge_unit_size * buffered_edges[i], MPI_CHAR, i, ShuffleGraph, MPI_COMM_WORLD);
        buffered_edges[i] = 0;
      }
      for (int i=0;i<partitions;i++) {  //���ͽ�����־
        char c = 0;
        MPI_Send(&c, 1, MPI_CHAR, i, ShuffleGraph, MPI_COMM_WORLD);
      }
      recv_thread_dst.join();
    }
	/*
	ͨ��compressed_outgoing_adj_index��outgoing_adj_index���л�ԭ������ɺ󶥵�src��outgoing_adj_list�Ŀ�ʼ����λ��Ϊ��outgoing_adj_index[s_i][src]

	*/
    for (int s_i=0;s_i<sockets;s_i++) {
      for (VertexId p_v_i=0;p_v_i<compressed_outgoing_adj_vertices[s_i];p_v_i++) {
        VertexId v_i = compressed_outgoing_adj_index[s_i][p_v_i].vertex;
        outgoing_adj_index[s_i][v_i] = compressed_outgoing_adj_index[s_i][p_v_i].index;
        outgoing_adj_index[s_i][v_i+1] = compressed_outgoing_adj_index[s_i][p_v_i+1].index;
      }
    }
    MPI_Barrier(MPI_COMM_WORLD);  //ͨ��դ����ʹÿ�������϶����DCSR������Ž�����һ��������DCSC


	/*
	6.����ͼ���ڴ��еĴ洢�ṹDCSC��ʵ��numa-aware����5���ơ�
	6.1�ȹ���incoming_adj_index��compressed_incoming_adj_index��ingoing_adj_bitmap
	�漰��shuffle������ÿ���������ļ��ж�ȡ�����ݣ����ձ�src������������������ͬ�����ı߷��͵���Ӧ�Ļ�����producer�� 
 ÿ���������յ�������<src,dst>�󣬹���ingoing_adj_bitmap��ingoing_adj_bitmap[src]��Ƕ���dst�Ƿ��г��ߣ�
 ingoing_adj_index��ingoing_adj_index[src]��Ƕ���dst�����ĳ��߸�����in_degree[src]��src������ȡ�
 Ȼ���ڽ���ѹ������compressed_incoming_adj_vertices��compressed_incoming_adj_index
	����IO:һ���ļ���ȡ��ÿ̨����ֻ��ȡ�������ݣ�
	*/
    EdgeId recv_incoming_edges = 0;
    incoming_edges = new EdgeId [sockets];
    incoming_adj_index = new EdgeId* [sockets];
    incoming_adj_list = new AdjUnit<EdgeData>* [sockets];
    incoming_adj_bitmap = new Bitmap * [sockets];
    for (int s_i=0;s_i<sockets;s_i++) {
      incoming_adj_bitmap[s_i] = new Bitmap (vertices);
      incoming_adj_bitmap[s_i]->clear();
      incoming_adj_index[s_i] = (EdgeId*)numa_alloc_onnode(sizeof(EdgeId) * (vertices+1), s_i);
    }
    {
      std::thread recv_thread_src([&](){
        int finished_count = 0;
        MPI_Status recv_status;
        while (finished_count < partitions) {
          MPI_Probe(MPI_ANY_SOURCE, ShuffleGraph, MPI_COMM_WORLD, &recv_status);
          int i = recv_status.MPI_SOURCE;
          assert(recv_status.MPI_TAG == ShuffleGraph && i >=0 && i < partitions);
          int recv_bytes;
          MPI_Get_count(&recv_status, MPI_CHAR, &recv_bytes);
          if (recv_bytes==1) {
            finished_count += 1;
            char c;
            MPI_Recv(&c, 1, MPI_CHAR, i, ShuffleGraph, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            continue;
          }
          assert(recv_bytes % edge_unit_size == 0);
          int recv_edges = recv_bytes / edge_unit_size;
          MPI_Recv(recv_buffer, edge_unit_size * recv_edges, MPI_CHAR, i, ShuffleGraph, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
          // #pragma omp parallel for
          for (EdgeId e_i=0;e_i<recv_edges;e_i++) {
            VertexId src = recv_buffer[e_i].src;
            VertexId dst = recv_buffer[e_i].dst;
            assert(src >= partition_offset[partition_id] && src < partition_offset[partition_id+1]);
            int src_part = get_local_partition_id(src);
            if (!incoming_adj_bitmap[src_part]->get_bit(dst)) {
              incoming_adj_bitmap[src_part]->set_bit(dst);
              incoming_adj_index[src_part][dst] = 0;
            }
            __sync_fetch_and_add(&incoming_adj_index[src_part][dst], 1);
          }
          recv_incoming_edges += recv_edges;
        }
      });
      for (int i=0;i<partitions;i++) {
        buffered_edges[i] = 0;
      }
      assert(lseek(fin, read_offset, SEEK_SET)==read_offset);
      read_bytes = 0;
      while (read_bytes < bytes_to_read) {
        long curr_read_bytes;
        if (bytes_to_read - read_bytes > edge_unit_size * CHUNKSIZE) {
          curr_read_bytes = read(fin, read_edge_buffer, edge_unit_size * CHUNKSIZE);
        } else {
          curr_read_bytes = read(fin, read_edge_buffer, bytes_to_read - read_bytes);
        }
        assert(curr_read_bytes>=0);
        read_bytes += curr_read_bytes;
        EdgeId curr_read_edges = curr_read_bytes / edge_unit_size;
        for (EdgeId e_i=0;e_i<curr_read_edges;e_i++) {
          VertexId src = read_edge_buffer[e_i].src;
          int i = get_partition_id(src);
          memcpy(send_buffer[i].data() + edge_unit_size * buffered_edges[i], &read_edge_buffer[e_i], edge_unit_size);
          buffered_edges[i] += 1;
          if (buffered_edges[i] == CHUNKSIZE) {
            MPI_Send(send_buffer[i].data(), edge_unit_size * buffered_edges[i], MPI_CHAR, i, ShuffleGraph, MPI_COMM_WORLD);
            buffered_edges[i] = 0;
          }
        }
      }
      for (int i=0;i<partitions;i++) {
        if (buffered_edges[i]==0) continue;
        MPI_Send(send_buffer[i].data(), edge_unit_size * buffered_edges[i], MPI_CHAR, i, ShuffleGraph, MPI_COMM_WORLD);
        buffered_edges[i] = 0;
      }
      for (int i=0;i<partitions;i++) {
        char c = 0;
        MPI_Send(&c, 1, MPI_CHAR, i, ShuffleGraph, MPI_COMM_WORLD);
      }
      recv_thread_src.join();
      #ifdef PRINT_DEBUG_MESSAGES
      printf("machine(%d) got %lu dense mode edges\n", partition_id, recv_incoming_edges);
      #endif
    }
    compressed_incoming_adj_vertices = new VertexId [sockets];
    compressed_incoming_adj_index = new CompressedAdjIndexUnit * [sockets];
    for (int s_i=0;s_i<sockets;s_i++) {
      incoming_edges[s_i] = 0;
      compressed_incoming_adj_vertices[s_i] = 0;
      for (VertexId v_i=0;v_i<vertices;v_i++) {
        if (incoming_adj_bitmap[s_i]->get_bit(v_i)) {
          incoming_edges[s_i] += incoming_adj_index[s_i][v_i];
          compressed_incoming_adj_vertices[s_i] += 1;
        }
      }
      compressed_incoming_adj_index[s_i] = (CompressedAdjIndexUnit*)numa_alloc_onnode( sizeof(CompressedAdjIndexUnit) * (compressed_incoming_adj_vertices[s_i] + 1) , s_i );
      compressed_incoming_adj_index[s_i][0].index = 0;
      EdgeId last_e_i = 0;
      compressed_incoming_adj_vertices[s_i] = 0;
      for (VertexId v_i=0;v_i<vertices;v_i++) {
        if (incoming_adj_bitmap[s_i]->get_bit(v_i)) {
          incoming_adj_index[s_i][v_i] = last_e_i + incoming_adj_index[s_i][v_i];
          last_e_i = incoming_adj_index[s_i][v_i];
          compressed_incoming_adj_index[s_i][compressed_incoming_adj_vertices[s_i]].vertex = v_i;
          compressed_incoming_adj_vertices[s_i] += 1;
          compressed_incoming_adj_index[s_i][compressed_incoming_adj_vertices[s_i]].index = last_e_i;
        }
      }
      for (VertexId p_v_i=0;p_v_i<compressed_incoming_adj_vertices[s_i];p_v_i++) {
        VertexId v_i = compressed_incoming_adj_index[s_i][p_v_i].vertex;
        incoming_adj_index[s_i][v_i] = compressed_incoming_adj_index[s_i][p_v_i].index;
        incoming_adj_index[s_i][v_i+1] = compressed_incoming_adj_index[s_i][p_v_i+1].index;
      }
      #ifdef PRINT_DEBUG_MESSAGES
      printf("part(%d) E_%d has %lu dense mode edges\n", partition_id, s_i, incoming_edges[s_i]);
      #endif
      incoming_adj_list[s_i] = (AdjUnit<EdgeData>*)numa_alloc_onnode(unit_size * incoming_edges[s_i], s_i);
    }
	/*
	6.2�ٹ���incoming_adj_list
	*/
    {
      std::thread recv_thread_src([&](){
        int finished_count = 0;
        MPI_Status recv_status;
        while (finished_count < partitions) {
          MPI_Probe(MPI_ANY_SOURCE, ShuffleGraph, MPI_COMM_WORLD, &recv_status);
          int i = recv_status.MPI_SOURCE;
          assert(recv_status.MPI_TAG == ShuffleGraph && i >=0 && i < partitions);
          int recv_bytes;
          MPI_Get_count(&recv_status, MPI_CHAR, &recv_bytes);
          if (recv_bytes==1) {
            finished_count += 1;
            char c;
            MPI_Recv(&c, 1, MPI_CHAR, i, ShuffleGraph, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            continue;
          }
          assert(recv_bytes % edge_unit_size == 0);
          int recv_edges = recv_bytes / edge_unit_size;
          MPI_Recv(recv_buffer, edge_unit_size * recv_edges, MPI_CHAR, i, ShuffleGraph, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
          #pragma omp parallel for
          for (EdgeId e_i=0;e_i<recv_edges;e_i++) {
            VertexId src = recv_buffer[e_i].src;
            VertexId dst = recv_buffer[e_i].dst;
            assert(src >= partition_offset[partition_id] && src < partition_offset[partition_id+1]);
            int src_part = get_local_partition_id(src);
            EdgeId pos = __sync_fetch_and_add(&incoming_adj_index[src_part][dst], 1);
            incoming_adj_list[src_part][pos].neighbour = src;
            if (!std::is_same<EdgeData, Empty>::value) {
              incoming_adj_list[src_part][pos].edge_data = recv_buffer[e_i].edge_data;
            }
          }
        }
      });
      for (int i=0;i<partitions;i++) {
        buffered_edges[i] = 0;
      }
      assert(lseek(fin, read_offset, SEEK_SET)==read_offset);
      read_bytes = 0;
      while (read_bytes < bytes_to_read) {
        long curr_read_bytes;
        if (bytes_to_read - read_bytes > edge_unit_size * CHUNKSIZE) {
          curr_read_bytes = read(fin, read_edge_buffer, edge_unit_size * CHUNKSIZE);
        } else {
          curr_read_bytes = read(fin, read_edge_buffer, bytes_to_read - read_bytes);
        }
        assert(curr_read_bytes>=0);
        read_bytes += curr_read_bytes;
        EdgeId curr_read_edges = curr_read_bytes / edge_unit_size;
        for (EdgeId e_i=0;e_i<curr_read_edges;e_i++) {
          VertexId src = read_edge_buffer[e_i].src;
          int i = get_partition_id(src);
          memcpy(send_buffer[i].data() + edge_unit_size * buffered_edges[i], &read_edge_buffer[e_i], edge_unit_size);
          buffered_edges[i] += 1;
          if (buffered_edges[i] == CHUNKSIZE) {
            MPI_Send(send_buffer[i].data(), edge_unit_size * buffered_edges[i], MPI_CHAR, i, ShuffleGraph, MPI_COMM_WORLD);
            buffered_edges[i] = 0;
          }
        }
      }
      for (int i=0;i<partitions;i++) {
        if (buffered_edges[i]==0) continue;
        MPI_Send(send_buffer[i].data(), edge_unit_size * buffered_edges[i], MPI_CHAR, i, ShuffleGraph, MPI_COMM_WORLD);
        buffered_edges[i] = 0;
      }
      for (int i=0;i<partitions;i++) {
        char c = 0;
        MPI_Send(&c, 1, MPI_CHAR, i, ShuffleGraph, MPI_COMM_WORLD);
      }
      recv_thread_src.join();
    }
    for (int s_i=0;s_i<sockets;s_i++) {
      for (VertexId p_v_i=0;p_v_i<compressed_incoming_adj_vertices[s_i];p_v_i++) {
        VertexId v_i = compressed_incoming_adj_index[s_i][p_v_i].vertex;
        incoming_adj_index[s_i][v_i] = compressed_incoming_adj_index[s_i][p_v_i].index;
        incoming_adj_index[s_i][v_i+1] = compressed_incoming_adj_index[s_i][p_v_i+1].index;
      }
    }
    MPI_Barrier(MPI_COMM_WORLD);

    delete [] buffered_edges;
    delete [] send_buffer;
    delete [] read_edge_buffer;
    delete [] recv_buffer;
    close(fin);

	//��ʼ��tuned_chunks_sparse����:���Ρ�tuned_chunks_sparse[i][0,threads-1]��¼��i��������ÿ���̴߳���Ķ������䡣
    //������ָi������s_i�߳�����numa��������Ķ��㼯��ӵ�г��ߵĵ㣨�г��ߵ�master���е�һ���֡�
    transpose(); 
    tune_chunks();
	//��ʼ��tuned_chunks_dense����
    transpose(); 
    tune_chunks();

    prep_time += MPI_Wtime();

    #ifdef PRINT_DEBUG_MESSAGES
    if (partition_id==0) {
      printf("preprocessing cost: %.2lf (s)\n", prep_time);
    }
    #endif
  }

  /*
  ���ڵ�partition_id�������ȵõ�i = current_send_part_id = ��partition_id+1��%partitions��
  Ȼ���ʼ��tuned_chunks_dense[i]����Ĵ�СΪthreads��
  tuned_chunks_dense[i][t_i]��¼��ǰһ������������ǰ�������ϵ�t_i���߳�������Ķ������䣬��������ķ�ΧΪ��t_i�߳�����numa�ڵ�����е�һ���ֶ��㣬
  ʹ��curr,end��¼����������compressed_incoming_adj_index�еĿ�ʼ�ͽ���λ�ã���t_i�̴߳���compressed_incoming_adj_index���±��curr��end�Ķ��㡣
 
  */
  void tune_chunks() {
    tuned_chunks_dense = new ThreadState * [partitions];
    int current_send_part_id = partition_id;   //??? ������
    for (int step=0;step<partitions;step++) {
      current_send_part_id = (current_send_part_id + 1) % partitions; //���Σ���������current_send_part_id��ֵΪʲô������partition_id
      int i = current_send_part_id;  
      tuned_chunks_dense[i] = new ThreadState [threads];
      EdgeId remained_edges;
      int remained_partitions;
      VertexId last_p_v_i;
      VertexId end_p_v_i;
      for (int t_i=0;t_i<threads;t_i++) {
        tuned_chunks_dense[i][t_i].status = WORKING;
        int s_i = get_socket_id(t_i);  //numa�ڵ���s_i
        int s_j = get_socket_offset(t_i); //numa�ڵ��еĵ�s_j���߳�
        if (s_j==0) {
          VertexId p_v_i = 0;
          while (p_v_i<compressed_incoming_adj_vertices[s_i]) {  //��s_i�ڵ�������Ķ��������У��ҵ���һ������ߵĶ�����compressed_incoming_adj_index�е�λ��
            VertexId v_i = compressed_incoming_adj_index[s_i][p_v_i].vertex;
            if (v_i >= partition_offset[i]) { 
              break;
            }
            p_v_i++;
          }
          last_p_v_i = p_v_i;
          while (p_v_i<compressed_incoming_adj_vertices[s_i]) { //�ҵ���������
            VertexId v_i = compressed_incoming_adj_index[s_i][p_v_i].vertex;
            if (v_i >= partition_offset[i+1]) {
              break;
            }
            p_v_i++;
          }
          end_p_v_i = p_v_i;
          remained_edges = 0;
          for (VertexId p_v_i=last_p_v_i;p_v_i<end_p_v_i;p_v_i++) { //ͳ��compressed_incoming_adj_index��[last_p_v_i, end_p_v_i)����������ܱ���
            remained_edges += compressed_incoming_adj_index[s_i][p_v_i+1].index - compressed_incoming_adj_index[s_i][p_v_i].index;
            remained_edges += alpha;
          }
        }
        tuned_chunks_dense[i][t_i].curr = last_p_v_i;  //ÿ���̴߳���Ķ����Ŀ�ʼλ�ã���t_i���̴߳���ĵ�һ������Ϊ��compressed_incoming_adj_index[ tuned_chunks_dense[i][t_i].curr ],
        tuned_chunks_dense[i][t_i].end = last_p_v_i;
        remained_partitions = threads_per_socket - s_j;
        EdgeId expected_chunk_size = remained_edges / remained_partitions;  //����numa��ÿ��thread����ı����������һ���߳�������ı����п������ڸ�ֵ��
        if (remained_partitions==1) {  //���һ���̴߳�����ʣ�µ����б�
          tuned_chunks_dense[i][t_i].end = end_p_v_i;
        } else {      //��֤ÿ���̴߳���expected_chunk_size����
          EdgeId got_edges = 0;
          for (VertexId p_v_i=last_p_v_i;p_v_i<end_p_v_i;p_v_i++) {
            got_edges += compressed_incoming_adj_index[s_i][p_v_i+1].index - compressed_incoming_adj_index[s_i][p_v_i].index + alpha;
            if (got_edges >= expected_chunk_size) {
              tuned_chunks_dense[i][t_i].end = p_v_i;     //�̴߳������Ľ���λ�ã���t_i���̴߳�������һ������Ϊ��compressed_incoming_adj_index[ tuned_chunks_dense[i][t_i].end ]
              last_p_v_i = tuned_chunks_dense[i][t_i].end;
              break;
            }
          }
          got_edges = 0;
          for (VertexId p_v_i=tuned_chunks_dense[i][t_i].curr;p_v_i<tuned_chunks_dense[i][t_i].end;p_v_i++) {
            got_edges += compressed_incoming_adj_index[s_i][p_v_i+1].index - compressed_incoming_adj_index[s_i][p_v_i].index + alpha;
          }
          remained_edges -= got_edges;  //�õ�ʣ����������
        }
      }
    }
  }

  // process vertices
  template<typename R>
  R process_vertices(std::function<R(VertexId)> process, Bitmap * active) {
    double stream_time = 0;
    stream_time -= MPI_Wtime();

    R reducer = 0;
    size_t basic_chunk = 64;
    for (int t_i=0;t_i<threads;t_i++) {  //Ϊÿ���̷߳��䴦����飬������СΪbasic_chunk�������߳�i����������numa�ڵ�Ķ������䡣
      int s_i = get_socket_id(t_i);
      int s_j = get_socket_offset(t_i);
      VertexId partition_size = local_partition_offset[s_i+1] - local_partition_offset[s_i];
      thread_state[t_i]->curr = local_partition_offset[s_i] + partition_size / threads_per_socket  / basic_chunk * basic_chunk * s_j;
      thread_state[t_i]->end = local_partition_offset[s_i] + partition_size / threads_per_socket / basic_chunk * basic_chunk * (s_j+1);
      if (s_j == threads_per_socket - 1) {
        thread_state[t_i]->end = local_partition_offset[s_i+1];
      }
      thread_state[t_i]->status = WORKING;
    }
    #pragma omp parallel reduction(+:reducer) //����������
    {
      R local_reducer = 0;
      int thread_id = omp_get_thread_num();
      while (true) {
        VertexId v_i = __sync_fetch_and_add(&thread_state[thread_id]->curr, basic_chunk);
        if (v_i >= thread_state[thread_id]->end) break;
        unsigned long word = active->data[WORD_OFFSET(v_i)];
        while (word != 0) {
          if (word & 1) {
            local_reducer += process(v_i);
          }
          v_i++;
          word = word >> 1;
        }
      }
      thread_state[thread_id]->status = STEALING;
      for (int t_offset=1;t_offset<threads;t_offset++) {
        int t_i = (thread_id + t_offset) % threads;
        while (thread_state[t_i]->status!=STEALING) {
          VertexId v_i = __sync_fetch_and_add(&thread_state[t_i]->curr, basic_chunk);
          if (v_i >= thread_state[t_i]->end) continue;
          unsigned long word = active->data[WORD_OFFSET(v_i)];
          while (word != 0) {
            if (word & 1) {
              local_reducer += process(v_i);
            }
            v_i++;
            word = word >> 1;
          }
        }
      }
      reducer += local_reducer;
    }
    R global_reducer;
    MPI_Datatype dt = get_mpi_data_type<R>();
    MPI_Allreduce(&reducer, &global_reducer, 1, dt, MPI_SUM, MPI_COMM_WORLD); //�����л����ϵĽ������reduce�����浽global_reducer��
    stream_time += MPI_Wtime();
    #ifdef PRINT_DEBUG_MESSAGES
    if (partition_id==0) {
      printf("process_vertices took %lf (s)\n", stream_time);
    }
    #endif
    return global_reducer;
  }

  template<typename M>
  void flush_local_send_buffer(int t_i) {
    int s_i = get_socket_id(t_i);
    int pos = __sync_fetch_and_add(&send_buffer[current_send_part_id][s_i]->count, local_send_buffer[t_i]->count);
    memcpy(send_buffer[current_send_part_id][s_i]->data + sizeof(MsgUnit<M>) * pos, local_send_buffer[t_i]->data, sizeof(MsgUnit<M>) * local_send_buffer[t_i]->count);
    local_send_buffer[t_i]->count = 0;
  }

  // emit a message to a vertex's master (dense) / mirror (sparse)
  template<typename M>
  void emit(VertexId vtx, M msg) {
    int t_i = omp_get_thread_num();
    MsgUnit<M> * buffer = (MsgUnit<M>*)local_send_buffer[t_i]->data;
    buffer[local_send_buffer[t_i]->count].vertex = vtx;
    buffer[local_send_buffer[t_i]->count].msg_data = msg;
    local_send_buffer[t_i]->count += 1;
    if (local_send_buffer[t_i]->count==local_send_buffer_limit) {
      flush_local_send_buffer<M>(t_i);
    }
  }

  // process edges ����������
  template<typename R, typename M>
  R process_edges(std::function<void(VertexId)> sparse_signal, std::function<R(VertexId, M, VertexAdjList<EdgeData>)> sparse_slot, std::function<void(VertexId, VertexAdjList<EdgeData>)> dense_signal, std::function<R(VertexId, M)> dense_slot, Bitmap * active, Bitmap * dense_selective = nullptr) {
    double stream_time = 0;
    stream_time -= MPI_Wtime();

    for (int t_i=0;t_i<threads;t_i++) {
      local_send_buffer[t_i]->resize( sizeof(MsgUnit<M>) * local_send_buffer_limit ); // MessageBuffer* [threads]; numa-aware
      local_send_buffer[t_i]->count = 0;
    }
    R reducer = 0;
    EdgeId active_edges = process_vertices<EdgeId>(
      [&](VertexId vtx){
        return (EdgeId)out_degree[vtx];
      },
      active
    );
    bool sparse = (active_edges < edges / 20);
    if (sparse) {
      for (int i=0;i<partitions;i++) {
        for (int s_i=0;s_i<sockets;s_i++) {
          recv_buffer[i][s_i]->resize( sizeof(MsgUnit<M>) * (partition_offset[i+1] - partition_offset[i]) * sockets );
          send_buffer[i][s_i]->resize( sizeof(MsgUnit<M>) * owned_vertices * sockets );
          send_buffer[i][s_i]->count = 0;
          recv_buffer[i][s_i]->count = 0;
        }
      }
    } else {
      for (int i=0;i<partitions;i++) {
        for (int s_i=0;s_i<sockets;s_i++) {
          recv_buffer[i][s_i]->resize( sizeof(MsgUnit<M>) * owned_vertices * sockets );
          send_buffer[i][s_i]->resize( sizeof(MsgUnit<M>) * (partition_offset[i+1] - partition_offset[i]) * sockets );
          send_buffer[i][s_i]->count = 0;
          recv_buffer[i][s_i]->count = 0;
        }
      }
    }
    size_t basic_chunk = 64;
    if (sparse) {                 //ϡ��ģʽ�� masterִ��sparse_signal, mirrorִ��sparse_slot
      #ifdef PRINT_DEBUG_MESSAGES
      if (partition_id==0) {
        printf("sparse mode\n");
      }
      #endif
      int * recv_queue = new int [partitions];
      int recv_queue_size = 0;
      std::mutex recv_queue_mutex;

      current_send_part_id = partition_id;
      #pragma omp parallel for
      for (VertexId begin_v_i=partition_offset[partition_id];begin_v_i<partition_offset[partition_id+1];begin_v_i+=basic_chunk) {  //��ǰ������masterִ�� signal����
        VertexId v_i = begin_v_i;
        unsigned long word = active->data[WORD_OFFSET(v_i)];
        while (word != 0) {
          if (word & 1) {
            sparse_signal(v_i);
          }
          v_i++;
          word = word >> 1;
        }
      }
      #pragma omp parallel for
      for (int t_i=0;t_i<threads;t_i++) {
        flush_local_send_buffer<M>(t_i);
      }
      recv_queue[recv_queue_size] = partition_id;  //recv_queue ʹ�ö��д��������������master���ݵ���ǰ�����ֵ��Ⱥ��ϵ����Ϊ2��3��������0��������ʱ����Ϊ�������������С��ԭ�������ݴﵽ���Ⱥ�˳�򲻹̶�����ǰ�������͸��Լ�������Ϊ���еĵ�һ����
      recv_queue_mutex.lock();
      recv_queue_size += 1;
      recv_queue_mutex.unlock();
      std::thread send_thread([&](){  // ����ǰ����������master�����signalֵ������send_buffer[partition_id][s_i]��)���͸����������ж�Ӧ��mirror���㣬�Ի��η�ʽ���Ϳ���ʵ�������������������4̨��������ÿ����������˳��Ϊ��0->[3,2,1] 1->[0,3,2] 2->[1,0,3] 3->[2,1,0]
        for (int step=1;step<partitions;step++) {
          int i = (partition_id - step + partitions) % partitions;
          for (int s_i=0;s_i<sockets;s_i++) { //����ǰ�����и���numa�ڵ���master����ִ��signal������ֵ���͵������������˴�socket����˳��0-sockets,�Է�����ʱ�Ϳ��Ը���0-sockets�����ݴ�ŵ���Ӧnuma��
            MPI_Send(send_buffer[partition_id][s_i]->data, sizeof(MsgUnit<M>) * send_buffer[partition_id][s_i]->count, MPI_CHAR, i, PassMessage, MPI_COMM_WORLD);
          }
        }
      });
      std::thread recv_thread([&](){ // ��ǰ�������������������͸���������mirror������Ϣ����ŵ�re���Ի��η�ʽ����ʵ��������0<-[1,2,3] 1<-[2,3,0] 2<-[3,0,1] 3<-[0,1,2]
        for (int step=1;step<partitions;step++) {
          int i = (partition_id + step) % partitions;
          for (int s_i=0;s_i<sockets;s_i++) {
            MPI_Status recv_status;
            MPI_Probe(i, PassMessage, MPI_COMM_WORLD, &recv_status); //��������������Ƿ�����Ϣ����
            MPI_Get_count(&recv_status, MPI_CHAR, &recv_buffer[i][s_i]->count);
            MPI_Recv(recv_buffer[i][s_i]->data, recv_buffer[i][s_i]->count, MPI_CHAR, i, PassMessage, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            recv_buffer[i][s_i]->count /= sizeof(MsgUnit<M>);
          }
          recv_queue[recv_queue_size] = i;
          recv_queue_mutex.lock();
          recv_queue_size += 1;
          recv_queue_mutex.unlock();
        }
      });
      for (int step=0;step<partitions;step++) {
        while (true) {
          recv_queue_mutex.lock();
          bool condition = (recv_queue_size<=step);
          recv_queue_mutex.unlock();
          if (!condition) break;
          __asm volatile ("pause" ::: "memory");
        }
        int i = recv_queue[step]; //���ն��л�ȡ��һ�ﵽ�Ļ�������
        MessageBuffer ** used_buffer;
        if (i==partition_id) {         //��ȡ����i���͹���������
          used_buffer = send_buffer[i];
        } else {
          used_buffer = recv_buffer[i];
        }
        for (int s_i=0;s_i<sockets;s_i++) {   //���numa�ڵ㴦����ÿ��numa�ڵ�Ķ������ݣ�������Ӧsocket�������߳̽��д����ʣ���ô��ʵ��numa������ǰһ��numaִ���꣬��ִ����һ��numa
          MsgUnit<M> * buffer = (MsgUnit<M> *)used_buffer[s_i]->data;
          size_t buffer_size = used_buffer[s_i]->count;
          for (int t_i=0;t_i<threads;t_i++) {   //���õ�ǰnuma�ڵ��е��̵߳Ĵ���Ķ�������͹���״̬
            // int s_i = get_socket_id(t_i);
            int s_j = get_socket_offset(t_i);
            VertexId partition_size = buffer_size;
            thread_state[t_i]->curr = partition_size / threads_per_socket  / basic_chunk * basic_chunk * s_j;
            thread_state[t_i]->end = partition_size / threads_per_socket / basic_chunk * basic_chunk * (s_j+1);
            if (s_j == threads_per_socket - 1) {
              thread_state[t_i]->end = buffer_size;
            }
            thread_state[t_i]->status = WORKING;
          }
          #pragma omp parallel reduction(+:reducer)  //����ִ�У�ÿ���߳�
          {
            R local_reducer = 0;
            int thread_id = omp_get_thread_num();
            int s_i = get_socket_id(thread_id);
            while (true) {
              VertexId b_i = __sync_fetch_and_add(&thread_state[thread_id]->curr, basic_chunk);
              if (b_i >= thread_state[thread_id]->end) break;
              VertexId begin_b_i = b_i;
              VertexId end_b_i = b_i + basic_chunk;
              if (end_b_i>thread_state[thread_id]->end) {
                end_b_i = thread_state[thread_id]->end;
              }
              for (b_i=begin_b_i;b_i<end_b_i;b_i++) {
                VertexId v_i = buffer[b_i].vertex;
                M msg_data = buffer[b_i].msg_data;           //��ȡ����������
                if (outgoing_adj_bitmap[s_i]->get_bit(v_i)) {  //�����ǰ�����г��ߣ���ִ��sparse_slot����
                  local_reducer += sparse_slot(v_i, msg_data, VertexAdjList<EdgeData>(outgoing_adj_list[s_i] + outgoing_adj_index[s_i][v_i], outgoing_adj_list[s_i] + outgoing_adj_index[s_i][v_i+1]));
                }
              }
            }
            thread_state[thread_id]->status = STEALING;
            for (int t_offset=1;t_offset<threads;t_offset++) {
              int t_i = (thread_id + t_offset) % threads;
              if (thread_state[t_i]->status==STEALING) continue;
              while (true) {
                VertexId b_i = __sync_fetch_and_add(&thread_state[t_i]->curr, basic_chunk);
                if (b_i >= thread_state[t_i]->end) break;
                VertexId begin_b_i = b_i;
                VertexId end_b_i = b_i + basic_chunk;
                if (end_b_i>thread_state[t_i]->end) {
                  end_b_i = thread_state[t_i]->end;
                }
                int s_i = get_socket_id(t_i);
                for (b_i=begin_b_i;b_i<end_b_i;b_i++) {
                  VertexId v_i = buffer[b_i].vertex; //sparseģʽ�´�master���㣬
                  M msg_data = buffer[b_i].msg_data; // master���㷢�͵�����
                  if (outgoing_adj_bitmap[s_i]->get_bit(v_i)) { //��mirror����ִ��slot����
                    local_reducer += sparse_slot(v_i, msg_data, VertexAdjList<EdgeData>(outgoing_adj_list[s_i] + outgoing_adj_index[s_i][v_i], outgoing_adj_list[s_i] + outgoing_adj_index[s_i][v_i+1]));
                  }
                }
              }
            }
            reducer += local_reducer;
          }
        }
      }
      send_thread.join();
      recv_thread.join();
      delete [] recv_queue;
    } else {
      // dense selective bitmap
      if (dense_selective!=nullptr && partitions>1) {
        double sync_time = 0;
        sync_time -= get_time();
        std::thread send_thread([&](){
          for (int step=1;step<partitions;step++) {
            int recipient_id = (partition_id + step) % partitions;
            MPI_Send(dense_selective->data + WORD_OFFSET(partition_offset[partition_id]), owned_vertices / 64, MPI_UNSIGNED_LONG, recipient_id, PassMessage, MPI_COMM_WORLD);
          }
        });
        std::thread recv_thread([&](){
          for (int step=1;step<partitions;step++) {
            int sender_id = (partition_id - step + partitions) % partitions;
            MPI_Recv(dense_selective->data + WORD_OFFSET(partition_offset[sender_id]), (partition_offset[sender_id + 1] - partition_offset[sender_id]) / 64, MPI_UNSIGNED_LONG, sender_id, PassMessage, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
          }
        });
        send_thread.join();
        recv_thread.join();
        MPI_Barrier(MPI_COMM_WORLD);
        sync_time += get_time();
        #ifdef PRINT_DEBUG_MESSAGES
        if (partition_id==0) {
          printf("sync_time = %lf\n", sync_time);
        }
        #endif
      }
      #ifdef PRINT_DEBUG_MESSAGES
      if (partition_id==0) {
        printf("dense mode\n");
      }
      #endif
      int * send_queue = new int [partitions];
      int * recv_queue = new int [partitions];
      volatile int send_queue_size = 0;
      volatile int recv_queue_size = 0;
      std::mutex send_queue_mutex;
      std::mutex recv_queue_mutex;

      std::thread send_thread([&](){
        for (int step=0;step<partitions;step++) {
          if (step==partitions-1) {
            break;
          }
          while (true) {
            send_queue_mutex.lock();
            bool condition = (send_queue_size<=step);
            send_queue_mutex.unlock();
            if (!condition) break;
            __asm volatile ("pause" ::: "memory");
          }
          int i = send_queue[step];
          for (int s_i=0;s_i<sockets;s_i++) {
            MPI_Send(send_buffer[i][s_i]->data, sizeof(MsgUnit<M>) * send_buffer[i][s_i]->count, MPI_CHAR, i, PassMessage, MPI_COMM_WORLD);
          }
        }
      });
      std::thread recv_thread([&](){
        std::vector<std::thread> threads;
        for (int step=1;step<partitions;step++) {
          int i = (partition_id - step + partitions) % partitions;
          threads.emplace_back([&](int i){
            for (int s_i=0;s_i<sockets;s_i++) {
              MPI_Status recv_status;
              MPI_Probe(i, PassMessage, MPI_COMM_WORLD, &recv_status);
              MPI_Get_count(&recv_status, MPI_CHAR, &recv_buffer[i][s_i]->count);
              MPI_Recv(recv_buffer[i][s_i]->data, recv_buffer[i][s_i]->count, MPI_CHAR, i, PassMessage, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
              recv_buffer[i][s_i]->count /= sizeof(MsgUnit<M>);
            }
          }, i);
        }
        for (int step=1;step<partitions;step++) {
          int i = (partition_id - step + partitions) % partitions;
          threads[step-1].join();
          recv_queue[recv_queue_size] = i;
          recv_queue_mutex.lock();
          recv_queue_size += 1;
          recv_queue_mutex.unlock();
        }
        recv_queue[recv_queue_size] = partition_id;
        recv_queue_mutex.lock();
        recv_queue_size += 1;
        recv_queue_mutex.unlock();
      });
      current_send_part_id = partition_id;
      for (int step=0;step<partitions;step++) {
        current_send_part_id = (current_send_part_id + 1) % partitions;
        int i = current_send_part_id;
        for (int t_i=0;t_i<threads;t_i++) {
          *thread_state[t_i] = tuned_chunks_dense[i][t_i];
        }
        #pragma omp parallel
        {
          int thread_id = omp_get_thread_num();
          int s_i = get_socket_id(thread_id);
          VertexId final_p_v_i = thread_state[thread_id]->end;
          while (true) {
            VertexId begin_p_v_i = __sync_fetch_and_add(&thread_state[thread_id]->curr, basic_chunk);
            if (begin_p_v_i >= final_p_v_i) break;
            VertexId end_p_v_i = begin_p_v_i + basic_chunk;
            if (end_p_v_i > final_p_v_i) {
              end_p_v_i = final_p_v_i;
            }
            for (VertexId p_v_i = begin_p_v_i; p_v_i < end_p_v_i; p_v_i ++) {
              VertexId v_i = compressed_incoming_adj_index[s_i][p_v_i].vertex;
              dense_signal(v_i, VertexAdjList<EdgeData>(incoming_adj_list[s_i] + compressed_incoming_adj_index[s_i][p_v_i].index, incoming_adj_list[s_i] + compressed_incoming_adj_index[s_i][p_v_i+1].index));
            }
          }
          thread_state[thread_id]->status = STEALING;
          for (int t_offset=1;t_offset<threads;t_offset++) {
            int t_i = (thread_id + t_offset) % threads;
            int s_i = get_socket_id(t_i);
            while (thread_state[t_i]->status!=STEALING) {
              VertexId begin_p_v_i = __sync_fetch_and_add(&thread_state[t_i]->curr, basic_chunk);
              if (begin_p_v_i >= thread_state[t_i]->end) break;
              VertexId end_p_v_i = begin_p_v_i + basic_chunk;
              if (end_p_v_i > thread_state[t_i]->end) {
                end_p_v_i = thread_state[t_i]->end;
              }
              for (VertexId p_v_i = begin_p_v_i; p_v_i < end_p_v_i; p_v_i ++) {
                VertexId v_i = compressed_incoming_adj_index[s_i][p_v_i].vertex;
                dense_signal(v_i, VertexAdjList<EdgeData>(incoming_adj_list[s_i] + compressed_incoming_adj_index[s_i][p_v_i].index, incoming_adj_list[s_i] + compressed_incoming_adj_index[s_i][p_v_i+1].index));
              }
            }
          }
        }
        #pragma omp parallel for
        for (int t_i=0;t_i<threads;t_i++) {
          flush_local_send_buffer<M>(t_i);
        }
        if (i!=partition_id) {
          send_queue[send_queue_size] = i;
          send_queue_mutex.lock();
          send_queue_size += 1;
          send_queue_mutex.unlock();
        }
      }
      for (int step=0;step<partitions;step++) {
        while (true) {
          recv_queue_mutex.lock();
          bool condition = (recv_queue_size<=step);
          recv_queue_mutex.unlock();
          if (!condition) break;
          __asm volatile ("pause" ::: "memory");
        }
        int i = recv_queue[step];
        MessageBuffer ** used_buffer;
        if (i==partition_id) {
          used_buffer = send_buffer[i];
        } else {
          used_buffer = recv_buffer[i];
        }
        for (int t_i=0;t_i<threads;t_i++) {
          int s_i = get_socket_id(t_i);
          int s_j = get_socket_offset(t_i);
          VertexId partition_size = used_buffer[s_i]->count;
          thread_state[t_i]->curr = partition_size / threads_per_socket  / basic_chunk * basic_chunk * s_j;
          thread_state[t_i]->end = partition_size / threads_per_socket / basic_chunk * basic_chunk * (s_j+1);
          if (s_j == threads_per_socket - 1) {
            thread_state[t_i]->end = used_buffer[s_i]->count;
          }
          thread_state[t_i]->status = WORKING;
        }
        #pragma omp parallel reduction(+:reducer)
        {
          R local_reducer = 0;
          int thread_id = omp_get_thread_num();
          int s_i = get_socket_id(thread_id);
          MsgUnit<M> * buffer = (MsgUnit<M> *)used_buffer[s_i]->data;
          while (true) {
            VertexId b_i = __sync_fetch_and_add(&thread_state[thread_id]->curr, basic_chunk);
            if (b_i >= thread_state[thread_id]->end) break;
            VertexId begin_b_i = b_i;
            VertexId end_b_i = b_i + basic_chunk;
            if (end_b_i>thread_state[thread_id]->end) {
              end_b_i = thread_state[thread_id]->end;
            }
            for (b_i=begin_b_i;b_i<end_b_i;b_i++) {
              VertexId v_i = buffer[b_i].vertex;
              M msg_data = buffer[b_i].msg_data;
              local_reducer += dense_slot(v_i, msg_data);
            }
          }
          thread_state[thread_id]->status = STEALING;
          reducer += local_reducer;
        }
      }
      send_thread.join();
      recv_thread.join();
      delete [] send_queue;
      delete [] recv_queue;
    }

    R global_reducer;
    MPI_Datatype dt = get_mpi_data_type<R>();
    MPI_Allreduce(&reducer, &global_reducer, 1, dt, MPI_SUM, MPI_COMM_WORLD);
    stream_time += MPI_Wtime();
    #ifdef PRINT_DEBUG_MESSAGES
    if (partition_id==0) {
      printf("process_edges took %lf (s)\n", stream_time);
    }
    #endif
    return global_reducer;
  }

};

#endif

