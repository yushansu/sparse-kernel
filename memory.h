#ifndef MEMORY_H_
#define MEMORY_H_

#include <numa.h>
#include <numaif.h>
#include <unordered_map>

#include "exception.h"

struct MMUtils {
    enum PageType {
        PAGE_NONE,
        PAGE_4K,
        PAGE_2M,
        PAGE_1G
    };

    enum MemoryNode {
        DEFAULT = -2,
        INTERLEAVE = -1,
        NODE0  = 0,
        NODE1  = 1,
        NODE2  = 2,
        NODE3  = 3,
        NODE4  = 4,
        NODE5  = 5,
        NODE6  = 6,
        NODE7  = 7,
        NODE8  = 8,
        NODE9  = 9,
        NODE10 = 10,
        NODE11 = 11,
        NODE12 = 12,
        NODE13 = 13,
        NODE14 = 14,
        NODE15 = 15
    };


    static std::unordered_map<void*, size_t> allocation_map;

    template <typename T>
    static T* Alloc(size_t len, PageType ptype = PAGE_NONE, MemoryNode mem_node = DEFAULT) {
        T* p = nullptr;
        if (ptype == PAGE_NONE) {
            p = static_cast<T *>(aligned_alloc(ALIGN_SIZE, sizeof(T) * len));
        } else {
            unsigned long* save_nodemask = (unsigned long*)malloc((numa_num_possible_nodes() + 7)/8);
            int save_mode = 0;
            RT_CHECK( get_mempolicy(&save_mode, save_nodemask, numa_num_possible_nodes(), NULL, 0) == 0);
            if (mem_node == INTERLEAVE) {
                unsigned long new_nodemask = ~0L;
                RT_CHECK( set_mempolicy(MPOL_INTERLEAVE, &new_nodemask, MAX_NODE) == 0 );
            } else if (mem_node == DEFAULT) {
                RT_CHECK( set_mempolicy(MPOL_DEFAULT, NULL, 0) == 0 );
            } else {
                unsigned long new_nodemask = (1L << mem_node);
                RT_CHECK( set_mempolicy(MPOL_BIND, &new_nodemask, sizeof(new_nodemask)*8) == 0 );
            }
            size_t bytes = sizeof(T) * len;
            if (ptype == PAGE_1G) {
                size_t pagesize = 1073741824UL;
                bytes = (bytes + pagesize - 1)/pagesize * pagesize;
                p = static_cast<T*>( mmap(NULL, bytes, PROT_READ | PROT_WRITE,
                                        MAP_ANONYMOUS | MAP_PRIVATE | MAP_HUGETLB | MAP_HUGE_1GB, -1, 0) );
            } else if (ptype == PAGE_2M) {
                size_t pagesize = 2097152UL;
                bytes = (bytes + pagesize - 1)/pagesize * pagesize;
                p = static_cast<T*>( mmap(NULL, bytes, PROT_READ | PROT_WRITE,
                                        MAP_ANONYMOUS | MAP_PRIVATE | MAP_HUGETLB, -1, 0) );
            } else if (ptype == PAGE_4K) {
                size_t pagesize = 4096UL;
                bytes = (bytes + pagesize - 1)/pagesize * pagesize;
                p = static_cast<T*>( mmap(NULL, bytes, PROT_READ | PROT_WRITE,
                                        MAP_ANONYMOUS | MAP_PRIVATE, -1, 0) );  
            }
            if (p == reinterpret_cast<void*>(-1)) {
                p = nullptr;
            }
            memset(p, 0, bytes);
            allocation_map[p] = bytes;
            RT_CHECK( set_mempolicy(save_mode, save_nodemask, numa_num_possible_nodes()) == 0 );
        }
        return p;
    }
    
    template <typename T>
    static void Free(T* p) {
        if (p != nullptr) {
            auto got = allocation_map.find(p);
            if (got == allocation_map.end()) {
                free(p);
            } else {
                size_t bytes = allocation_map[p];
                munmap(p, bytes);
            }
        }
    }

    static void PrintMemoryInfo() {
        std::cout << "Current memory status in GiB:" << std::endl;
        int num_numa_nodes = numa_num_configured_nodes();
        for (int i = 0; i < num_numa_nodes; ++i) {
            long long numa_free;
            long long numa_size;
            numa_size = numa_node_size64(i, &numa_free);
            std::cout << "  node " << i << ": ";
            std::cout << std::showpoint;
            std::cout << "size = " << numa_size/(1024.0*1024*1024);
            std::cout << " alloc = " << (numa_size - numa_free)/(1024.0*1024*1024);
            std::cout << " free = " << numa_free/(1024.0*1024*1024) << std::endl;
        }
    }

  private:
    enum {
        ALIGN_SIZE = 64
    };

    enum {
        MAX_NODE = sizeof(unsigned long) * 8,
    };
};

std::unordered_map<void*, size_t> MMUtils::allocation_map;

#endif // MEMORY_H_