#ifndef TIMER_H_
#define TIMER_H_

#include <cstdint>
#include <map>
#include <unordered_map>
#include <string>
#include <unistd.h>
#include <omp.h>
#include <iostream>
#include <iomanip>
#include <utility>
#include <vector>

#include "utils.h"

#define ENABLE_TIMER_

class Timer {
    struct BlockInfo {
        unsigned long long max_ticks;
        unsigned long long min_ticks = 0xFFFFFFFFFFFFFFFF;
        unsigned long long start_ticks;
        unsigned long long total_ticks;
        uint32_t starts;
        uint32_t stops;
        double ops;
    };

  public:
    Timer() {
        unsigned long long start_tick = ReadTSC();
        sleep(1);
        unsigned long long end_tick = ReadTSC();
        ticks_persecond_ =  static_cast<double>(end_tick - start_tick);
    }
    
    ~Timer() {};

    void Verbose(bool verbose = true) {
        verbose_ = true;
    }
    
    void Begin(const std::string &label) {
#ifdef ENABLE_TIMER_
        size_t idx = IndexFromLabel(label);
        blocks_[idx].starts += 1;
        blocks_[idx].start_ticks = ReadTSC();
#endif
    }
    
    void End(const std::string &label, double ops = 0.0) {
#ifdef ENABLE_TIMER_
        unsigned long long t_end = ReadTSC();
        size_t idx = IndexFromLabel(label);
        blocks_[idx].stops += 1;
        unsigned long long elapsed_ticks = t_end - blocks_[idx].start_ticks;
        blocks_[idx].total_ticks += elapsed_ticks;
        blocks_[idx].max_ticks = std::max(blocks_[idx].max_ticks, elapsed_ticks);
        blocks_[idx].min_ticks = std::min(blocks_[idx].min_ticks, elapsed_ticks);
        blocks_[idx].ops += ops;
        if (verbose_) {
            double elapsed_time = static_cast<double>(elapsed_ticks)/ticks_persecond_;
            double gops = ops/elapsed_time/1e9;
            std::cout << "\"" << label << "\"" << " called: ";
            std::cout << std::scientific;
            std::cout << elapsed_time << " secs, ";
            std::cout.unsetf(std::ios::scientific);
            std::cout << std::fixed;
            std::cout << gops << " GOPS" << std::endl;
            std::cout.unsetf(std::ios::fixed);
        }
#endif
    }
    
    void Reset() {
        blocks_.clear();
        label_map_.clear();
    }
    
    void Summary() {
#ifdef ENABLE_TIMER_
        const int kNameWidth = 32;
        const int kIntWidth = 9;
        const int kFPWidth = 16;
        std::cout << std::endl;
        std::cout << detail::RepeatString("-", kNameWidth +  kFPWidth * 5 + kIntWidth) << std::endl;
        std::cout << std::left;
        std::cout << std::setw(kNameWidth) << "Perf.  summary:";
        std::cout << std::right;
        std::cout << std::setw(kIntWidth) << "#calls";
        std::cout << std::setw(kFPWidth) << "total(secs)";
        std::cout << std::setw(kFPWidth) << "ave(secs)";
        std::cout << std::setw(kFPWidth) << "max(secs)";
        std::cout << std::setw(kFPWidth) << "min(secs)";
        std::cout << std::setw(kFPWidth) << "Perf.(GOPS)";
        std::cout << std::endl;
        std::cout << detail::RepeatString("-", kNameWidth +  kFPWidth * 5 + kIntWidth) << std::endl;
        double sum_total_time = 0.0;
        double sum_average_time = 0.0;
        double sum_max_time = 0.0;
        double sum_min_time = 0.0;
        for (auto it = label_map_.begin(); it != label_map_.end(); ++it){
            size_t idx = it->second;
            if (blocks_[idx].starts == blocks_[idx].stops) {
                // time
                double total_time = static_cast<double>(blocks_[idx].total_ticks)/ticks_persecond_;
                sum_total_time += total_time;
                double max_time = static_cast<double>(blocks_[idx].max_ticks)/ticks_persecond_;
                sum_max_time += max_time;
                double min_time = static_cast<double>(blocks_[idx].min_ticks)/ticks_persecond_;
                sum_min_time += min_time;
                double average_time = total_time/blocks_[idx].starts;
                sum_average_time += average_time;
                std::cout << std::left;
                std::cout << std::setw(kNameWidth) << it->first;
                std::cout << std::right;
                std::cout << std::setw(kIntWidth) << blocks_[idx].starts;
                std::cout << std::scientific;
                std::cout << std::setw(kFPWidth) << total_time;
                std::cout << std::setw(kFPWidth) << average_time;
                std::cout << std::setw(kFPWidth) << max_time;
                std::cout << std::setw(kFPWidth) << min_time;
                std::cout.unsetf(std::ios::scientific);
                if (blocks_[idx].ops != 0) {
                    double gops = blocks_[idx].ops/total_time/1e9;
                    std::cout << std::setw(kFPWidth) << gops;
                } else {
                    std::cout << std::setw(kFPWidth) << "NA";
                }
                std::cout << std::endl;
            } else {
                std::cout << "mismatch in starts/stops for code block \"" << it->first << "\""<< std::endl;
                std::cout <<"  starts = " << blocks_[idx].starts << std::endl;
                std::cout <<"  stops = " << blocks_[idx].stops << std::endl;
            } // if (block_starts[j] == block_stops[j])
        } // for (int j = 0; j< n_code_blocks; j++)
        std::cout << detail::RepeatString("-", kNameWidth +  kFPWidth * 5 + kIntWidth) << std::endl;
        std::cout << std::left;
        std::cout << std::setw(kNameWidth) << "Total";
        std::cout << std::right;
        std::cout << std::setw(kIntWidth) << "NA";
        std::cout << std::scientific;
        std::cout << std::setw(kFPWidth) << sum_total_time;
        std::cout << std::setw(kFPWidth) << sum_average_time;
        std::cout << std::setw(kFPWidth) << sum_max_time;
        std::cout << std::setw(kFPWidth) << sum_min_time;
        std::cout.unsetf(std::ios::scientific);
        std::cout << std::setw(kFPWidth) << "NA";
        std::cout << std::endl << std::endl;
#endif
    }

  private:
    bool verbose_ = false;
    double ticks_persecond_;
    std::vector<BlockInfo> blocks_;
    std::map<std::string, size_t> label_map_;

    size_t IndexFromLabel(const std::string &label){
        auto got = label_map_.find(label);
        size_t idx = 0;
        if (got == label_map_.end()) {
            blocks_.push_back(BlockInfo());
            idx = blocks_.size() - 1;
            label_map_.insert(std::pair<std::string, size_t>(label, idx));
        } else {
            idx = got->second;
        }
        return idx;
    }
    
    unsigned long long ReadTSC() {
#if defined(__i386__)
        unsigned long long x;
        __asm__ __volatile__(".byte 0x0f, 0x31" : "=A"(x));
        return x;
#elif defined(__x86_64__)
        uint32_t hi;
        uint32_t lo;
        __asm__ __volatile__("rdtsc" : "=a"(lo), "=d"(hi));
        return ((unsigned long long)lo) | (((unsigned long long)hi) << 32);
#elif defined(__powerpc__)
        unsigned long long result = 0;
        unsigned long long upper, lower, tmp;
        __asm__ __volatile__("0:                  \n"
                                "\tmftbu   %0           \n"
                                "\tmftb    %1           \n"
                                "\tmftbu   %2           \n"
                                "\tcmpw    %2,%0        \n"
                                "\tbne     0b         \n"
                                : "=r"(upper), "=r"(lower), "=r"(tmp));
        result = upper;
        result = result << 32;
        result = result | lower;
        return result;
#endif // defined(__i386__)
    }
}; // class Timer

#endif // TIMER_H_