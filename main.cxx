#include <utility>
#include <vector>
#include <string>
#include <iostream>
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include "inc/main.hxx"

using namespace std;




#pragma region CONFIGURATION
#ifndef TYPE
/** Type of edge weights. */
#define TYPE float
#endif
#ifndef MAX_THREADS
/** Maximum number of threads to use. */
#define MAX_THREADS 64
#endif
#ifndef REPEAT_METHOD
/** Number of times to repeat each method. */
#define REPEAT_METHOD 5
#endif
#pragma endregion




// HELPERS
// -------

template <class G, class R>
inline double getModularity(const G& x, const R& a, double M) {
  auto fc = [&](auto u) { return a.membership[u]; };
  return modularityByOmp(x, fc, M, 1.0);
}




// PERFORM EXPERIMENT
// ------------------

template <size_t NUM_SLOTS, class G>
inline auto runLeidenLowmem(const G& x, int repeat) {
  auto fh = [](auto c) { return c % NUM_SLOTS; };
  return leidenLowmemStaticOmp(x, {repeat, NUM_SLOTS}, fh);
}


template <class G>
void runExperiment(const G& x) {
  int repeat = REPEAT_METHOD;
  double   M = edgeWeightOmp(x)/2;
  // Follow a specific result logging format, which can be easily parsed later.
  auto flog = [&](const auto& ans, const char *technique, size_t numSlots=0) {
    printf(
      "{%09.1fms, %09.1fms mark, %09.1fms init, %09.1fms firstpass, %09.1fms locmove, %09.1fms refine, %09.1fms aggr, %.3e slots, %04d iters, %03d passes, %01.9f modularity} %s\n",
      ans.time, ans.markingTime, ans.initializationTime, ans.firstPassTime, ans.localMoveTime, ans.refinementTime, ans.aggregationTime,
      double(numSlots), ans.iterations, ans.passes, getModularity(x, ans, M), technique
    );
  };
  // Get community memberships on original graph (static).
  {
    auto b0 = leidenStaticOmp(x, {repeat});
    flog(b0, "leidenStaticOmp");
  }
  // Get community memberships on original graph (low memory).
  {
    auto b1 = runLeidenLowmem<(1 << 28)>(x, repeat);
    flog(b1, "leidenLowmemStaticOmp28", 1 << 28);
  }
  {
    auto b1 = runLeidenLowmem<(1 << 27)>(x, repeat);
    flog(b1, "leidenLowmemStaticOmp27", 1 << 27);
  }
  {
    auto b1 = runLeidenLowmem<(1 << 26)>(x, repeat);
    flog(b1, "leidenLowmemStaticOmp26", 1 << 26);
  }
  {
    auto b1 = runLeidenLowmem<(1 << 25)>(x, repeat);
    flog(b1, "leidenLowmemStaticOmp25", 1 << 25);
  }
  {
    auto b1 = runLeidenLowmem<(1 << 24)>(x, repeat);
    flog(b1, "leidenLowmemStaticOmp24", 1 << 24);
  }
  {
    auto b1 = runLeidenLowmem<(1 << 23)>(x, repeat);
    flog(b1, "leidenLowmemStaticOmp23", 1 << 23);
  }
  {
    auto b1 = runLeidenLowmem<(1 << 22)>(x, repeat);
    flog(b1, "leidenLowmemStaticOmp22", 1 << 22);
  }
  {
    auto b1 = runLeidenLowmem<(1 << 21)>(x, repeat);
    flog(b1, "leidenLowmemStaticOmp21", 1 << 21);
  }
  {
    auto b1 = runLeidenLowmem<(1 << 20)>(x, repeat);
    flog(b1, "leidenLowmemStaticOmp20", 1 << 20);
  }
  {
    auto b1 = runLeidenLowmem<(1 << 19)>(x, repeat);
    flog(b1, "leidenLowmemStaticOmp19", 1 << 19);
  }
  {
    auto b1 = runLeidenLowmem<(1 << 18)>(x, repeat);
    flog(b1, "leidenLowmemStaticOmp18", 1 << 18);
  }
  {
    auto b1 = runLeidenLowmem<(1 << 17)>(x, repeat);
    flog(b1, "leidenLowmemStaticOmp17", 1 << 17);
  }
  {
    auto b1 = runLeidenLowmem<(1 << 16)>(x, repeat);
    flog(b1, "leidenLowmemStaticOmp16", 1 << 16);
  }
  {
    auto b1 = runLeidenLowmem<(1 << 15)>(x, repeat);
    flog(b1, "leidenLowmemStaticOmp15", 1 << 15);
  }
  {
    auto b1 = runLeidenLowmem<(1 << 14)>(x, repeat);
    flog(b1, "leidenLowmemStaticOmp14", 1 << 14);
  }
  {
    auto b1 = runLeidenLowmem<(1 << 13)>(x, repeat);
    flog(b1, "leidenLowmemStaticOmp13", 1 << 13);
  }
  {
    auto b1 = runLeidenLowmem<(1 << 12)>(x, repeat);
    flog(b1, "leidenLowmemStaticOmp12", 1 << 12);
  }
  {
    auto b1 = runLeidenLowmem<(1 << 11)>(x, repeat);
    flog(b1, "leidenLowmemStaticOmp11", 1 << 11);
  }
  {
    auto b1 = runLeidenLowmem<(1 << 10)>(x, repeat);
    flog(b1, "leidenLowmemStaticOmp10", 1 << 10);
  }
  {
    auto b1 = runLeidenLowmem<(1 <<  9)>(x, repeat);
    flog(b1, "leidenLowmemStaticOmp09", 1 <<  9);
  }
  {
    auto b1 = runLeidenLowmem<(1 <<  8)>(x, repeat);
    flog(b1, "leidenLowmemStaticOmp08", 1 <<  8);
  }
  {
    auto b1 = runLeidenLowmem<(1 <<  7)>(x, repeat);
    flog(b1, "leidenLowmemStaticOmp07", 1 <<  7);
  }
  {
    auto b1 = runLeidenLowmem<(1 <<  6)>(x, repeat);
    flog(b1, "leidenLowmemStaticOmp06", 1 <<  6);
  }
  {
    auto b1 = runLeidenLowmem<(1 <<  5)>(x, repeat);
    flog(b1, "leidenLowmemStaticOmp05", 1 <<  5);
  }
  {
    auto b1 = runLeidenLowmem<(1 <<  4)>(x, repeat);
    flog(b1, "leidenLowmemStaticOmp04", 1 <<  4);
  }
  {
    auto b1 = runLeidenLowmem<(1 <<  3)>(x, repeat);
    flog(b1, "leidenLowmemStaticOmp03", 1 <<  3);
  }
  {
    auto b1 = runLeidenLowmem<(1 <<  2)>(x, repeat);
    flog(b1, "leidenLowmemStaticOmp02", 1 <<  2);
  }
  {
    auto b1 = runLeidenLowmem<(1 <<  1)>(x, repeat);
    flog(b1, "leidenLowmemStaticOmp01", 1 <<  1);
  }
}


int main(int argc, char **argv) {
  using K = uint32_t;
  using V = TYPE;
  install_sigsegv();
  char *file     = argv[1];
  bool symmetric = argc>2? stoi(argv[2]) : false;
  bool weighted  = argc>3? stoi(argv[3]) : false;
  omp_set_num_threads(MAX_THREADS);
  LOG("OMP_NUM_THREADS=%d\n", MAX_THREADS);
  LOG("Loading graph %s ...\n", file);
  DiGraph<K, None, V> x;
  readMtxOmpW(x, file, weighted); LOG(""); println(x);
  if (!symmetric) { symmetrizeOmpU(x); LOG(""); print(x); printf(" (symmetrize)\n"); }
  runExperiment(x);
  printf("\n");
  return 0;
}
