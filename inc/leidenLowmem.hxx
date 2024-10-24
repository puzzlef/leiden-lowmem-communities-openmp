#pragma once
#include <utility>
#include <limits>
#include <tuple>
#include <vector>
#include <algorithm>
#include <omp.h>
#include "_main.hxx"
#include "Graph.hxx"
#include "properties.hxx"
#include "csr.hxx"
#include "bfs.hxx"
#include "leiden.hxx"

using std::numeric_limits;
using std::tuple;
using std::vector;
using std::make_pair;
using std::move;
using std::swap;
using std::get;
using std::min;
using std::max;




#pragma region METHODS
#pragma region CHANGE COMMUNITY
/**
 * Scan an edge community connected to a vertex.
 * @param vcs communities vertex u is linked to (updated)
 * @param vcout total edge weight from vertex u to community C (updated)
 * @param u given vertex
 * @param v outgoing edge vertex
 * @param w outgoing edge weight
 * @param vcom community each vertex belongs to
 * @param vcob community bound each vertex belongs to
 * @param fh hash function mapping community to index
 */
template <bool SELF=false, bool REFINE=false, class K, class V, class W, class FH>
inline void leidenLowmemScanCommunityW(vector<K>& vcs, vector<W>& vcout, K u, K v, V w, const vector<K>& vcom, const vector<K>& vcob, FH fh) {
  if (!SELF && u==v) return;
  if (REFINE && vcob[u]!=vcob[v]) return;
  K c = vcom[v];
  if (!vcout[fh(c)]) vcs.push_back(c);
  vcout[fh(c)] += w;
}


/**
 * Scan an edge community connected to a vertex.
 * @param vcs communities vertex u is linked to (updated)
 * @param vcout total edge weight from vertex u to community C (updated)
 * @param u given vertex
 * @param v outgoing edge vertex
 * @param w outgoing edge weight
 * @param vcom community each vertex belongs to
 * @param fh hash function mapping community to index
 */
template <bool SELF=false, class K, class V, class W, class FH>
inline void leidenLowmemScanCommunityW(vector<K>& vcs, vector<W>& vcout, K u, K v, V w, const vector<K>& vcom, FH fh) {
  leidenLowmemScanCommunityW<SELF, false>(vcs, vcout, u, v, w, vcom, vcom, fh);
}


/**
 * Scan communities connected to a vertex.
 * @param vcs communities vertex u is linked to (updated)
 * @param vcout total edge weight from vertex u to community C (updated)
 * @param x original graph
 * @param u given vertex
 * @param vcom community each vertex belongs to
 * @param vcob community bound each vertex belongs to
 * @param fh hash function mapping community to index
 */
template <bool SELF=false, bool REFINE=false, class G, class K, class W, class FH>
inline void leidenLowmemScanCommunitiesW(vector<K>& vcs, vector<W>& vcout, const G& x, K u, const vector<K>& vcom, const vector<K>& vcob, FH fh) {
  x.forEachEdge(u, [&](auto v, auto w) { leidenLowmemScanCommunityW<SELF, REFINE>(vcs, vcout, u, v, w, vcom, vcob, fh); });
}


/**
 * Scan communities connected to a vertex.
 * @param vcs communities vertex u is linked to (updated)
 * @param vcout total edge weight from vertex u to community C (updated)
 * @param x original graph
 * @param u given vertex
 * @param vcom community each vertex belongs to
 * @param fh hash function mapping community to index
 */
template <bool SELF=false, class G, class K, class W, class FH>
inline void leidenLowmemScanCommunitiesW(vector<K>& vcs, vector<W>& vcout, const G& x, K u, const vector<K>& vcom, FH fh) {
  leidenLowmemScanCommunitiesW<SELF, false>(vcs, vcout, x, u, vcom, vcom, fh);
}


/**
 * Clear communities scan data.
 * @param vcs total edge weight from vertex u to community C (updated)
 * @param vcout communities vertex u is linked to (updated)
 * @param fh hash function mapping community to index
 */
template <class K, class W, class FH>
inline void leidenLowmemClearScanW(vector<K>& vcs, vector<W>& vcout, FH fh) {
  for (K c : vcs)
    vcout[fh(c)] = W();
  vcs.clear();
}


/**
 * Choose connected community with best delta modularity.
 * @param x original graph
 * @param u given vertex
 * @param d previous community of vertex u
 * @param vtot total edge weight of each vertex
 * @param ctot total edge weight of each community
 * @param vcs communities vertex u is linked to
 * @param vcout total edge weight from vertex u to community C
 * @param M total weight of "undirected" graph (1/2 of directed graph)
 * @param R resolution (0, 1]
 * @param fh hash function mapping community to index
 * @returns [best community, delta modularity]
 */
template <bool SELF=false, class G, class K, class W, class FH>
inline auto leidenLowmemChooseCommunity(const G& x, K u, K d, const vector<W>& vtot, const vector<W>& ctot, const vector<K>& vcs, const vector<W>& vcout, double M, double R, FH fh) {
  K cmax = K();
  W emax = W();
  for (K c : vcs) {
    if (!SELF && c==d) continue;
    W e = deltaModularity(vcout[fh(c)], vcout[fh(d)], vtot[u], ctot[c], ctot[d], M, R);
    if (e>emax) { emax = e; cmax = c; }
  }
  return make_pair(cmax, emax);
}
#pragma endregion




#pragma region LOCAL-MOVING PHASE
/**
 * Leiden algorithm's local moving phase.
 * @param vcom community each vertex belongs to (initial, updated)
 * @param ctot total edge weight of each community (precalculated, updated)
 * @param vaff is vertex affected flag (updated)
 * @param vcs communities vertex u is linked to (temporary buffer, updated)
 * @param vcout total edge weight from vertex u to community C (temporary buffer, updated)
 * @param x original graph
 * @param vcob community bound each vertex belongs to
 * @param vtot total edge weight of each vertex
 * @param M total weight of "undirected" graph (1/2 of directed graph)
 * @param R resolution (0, 1]
 * @param L max iterations
 * @param fc has local moving phase converged?
 * @param fa is vertex allowed to be updated?
 * @param fb track communities that need to be broken
 * @param fh hash function mapping community to index
 * @returns iterations performed (0 if converged already)
 */
template <bool REFINE=false, class G, class K, class W, class B, class FC, class FA, class FB, class FH>
inline int leidenLowmemMoveOmpW(vector<K>& vcom, vector<W>& ctot, vector<B>& vaff, vector<vector<K>*>& vcs, vector<vector<W>*>& vcout, const G& x, const vector<K>& vcob, const vector<W>& vtot, double M, double R, int L, FC fc, FA fa, FB fb, FH fh) {
  size_t S = x.span();
  int l = 0;
  W  el = W();
  for (; l<L;) {
    el = W();
    #pragma omp parallel for schedule(dynamic, 2048) reduction(+:el)
    for (K u=0; u<S; ++u) {
      int t = omp_get_thread_num();
      K   d = vcom[u];
      if (!x.hasVertex(u)) continue;
      if (!fa(u)) continue;
      if (!REFINE && !vaff[u]) continue;
      if ( REFINE && (d!=u || ctot[d]>vtot[u])) continue;
      leidenLowmemClearScanW(*vcs[t], *vcout[t], fh);
      leidenLowmemScanCommunitiesW<false, REFINE>(*vcs[t], *vcout[t], x, u, vcom, vcob, fh);
      auto [c, e] = leidenLowmemChooseCommunity(x, u, d, vtot, ctot, *vcs[t], *vcout[t], M, R, fh);
      if (e && leidenChangeCommunityOmpW<REFINE>(vcom, ctot, x, u, d, c, vtot)) {
        if (!REFINE) fb(d);
        if (!REFINE) x.forEachEdgeKey(u, [&](auto v) { vaff[v] = B(1); });
      }
      if (!REFINE) vaff[u] = B();
      el += e;  // l1-norm
    }
    if (REFINE || fc(el, l++)) break;
  }
  return l>1 || el? l : 0;
}


/**
 * Leiden algorithm's local moving phase.
 * @param vcom community each vertex belongs to (initial, updated)
 * @param ctot total edge weight of each community (precalculated, updated)
 * @param vaff is vertex affected flag (updated)
 * @param vcs communities vertex u is linked to (temporary buffer, updated)
 * @param vcout total edge weight from vertex u to community C (temporary buffer, updated)
 * @param x original graph
 * @param vcob community bound each vertex belongs to
 * @param vtot total edge weight of each vertex
 * @param M total weight of "undirected" graph (1/2 of directed graph)
 * @param R resolution (0, 1]
 * @param L max iterations
 * @param fc has local moving phase converged?
 * @param fa is vertex allowed to be updated?
 * @param fh hash function mapping community to index
 * @returns iterations performed (0 if converged already)
 */
template <bool REFINE=false, class G, class K, class W, class B, class FC, class FA, class FH>
inline int leidenLowmemMoveOmpW(vector<K>& vcom, vector<W>& ctot, vector<B>& vaff, vector<vector<K>*>& vcs, vector<vector<W>*>& vcout, const G& x, const vector<K>& vcob, const vector<W>& vtot, double M, double R, int L, FC fc, FA fa, FH fh) {
  auto fb = [](auto u) {};
  return leidenLowmemMoveOmpW<REFINE>(vcom, ctot, vaff, vcs, vcout, x, vcob, vtot, M, R, L, fc, fa, fb, fh);
}


/**
 * Leiden algorithm's local moving phase.
 * @param vcom community each vertex belongs to (initial, updated)
 * @param ctot total edge weight of each community (precalculated, updated)
 * @param vaff is vertex affected flag (updated)
 * @param vcs communities vertex u is linked to (temporary buffer, updated)
 * @param vcout total edge weight from vertex u to community C (temporary buffer, updated)
 * @param x original graph
 * @param vcob community bound each vertex belongs to
 * @param vtot total edge weight of each vertex
 * @param M total weight of "undirected" graph (1/2 of directed graph)
 * @param R resolution (0, 1]
 * @param L max iterations
 * @param fc has local moving phase converged?
 * @param fh hash function mapping community to index
 * @returns iterations performed (0 if converged already)
 */
template <bool REFINE=false, class G, class K, class W, class B, class FC, class FH>
inline int leidenLowmemMoveOmpW(vector<K>& vcom, vector<W>& ctot, vector<B>& vaff, vector<vector<K>*>& vcs, vector<vector<W>*>& vcout, const G& x, const vector<K>& vcob, const vector<W>& vtot, double M, double R, int L, FC fc, FH fh) {
  auto fa = [](auto u) { return true; };
  return leidenLowmemMoveOmpW<REFINE>(vcom, ctot, vaff, vcs, vcout, x, vcob, vtot, M, R, L, fc, fa, fh);
}
#pragma endregion




#pragma region AGGREGATION PHASE
/**
 * Aggregate outgoing edges of each community.
 * @param ydeg degree of each community (updated)
 * @param yedg vertex ids of outgoing edges of each community (updated)
 * @param ywei weights of outgoing edges of each community (updated)
 * @param vcs communities vertex u is linked to (temporary buffer, updated)
 * @param vcout total edge weight from vertex u to community C (temporary buffer, updated)
 * @param x original graph
 * @param vcom community each vertex belongs to
 * @param coff offsets for vertices belonging to each community
 * @param cedg vertices belonging to each community
 * @param yoff offsets for vertices belonging to each community
 * @param fh hash function mapping community to index
 */
template <int CHUNK_SIZE=2048, class G, class K, class W, class FH>
inline void leidenLowmemAggregateEdgesOmpW(vector<K>& ydeg, vector<K>& yedg, vector<W>& ywei, vector<vector<K>*>& vcs, vector<vector<W>*>& vcout, const G& x, const vector<K>& vcom, const vector<K>& coff, const vector<K>& cedg, const vector<size_t>& yoff, FH fh) {
  size_t C = coff.size() - 1;
  fillValueOmpU(ydeg, K());
  #pragma omp parallel for schedule(dynamic, CHUNK_SIZE)
  for (K c=0; c<C; ++c) {
    int t = omp_get_thread_num();
    K   n = csrDegree(coff, c);
    if (n==0) continue;
    leidenLowmemClearScanW(*vcs[t], *vcout[t], fh);
    csrForEachEdgeKey(coff, cedg, c, [&](auto u) {
      leidenLowmemScanCommunitiesW<true>(*vcs[t], *vcout[t], x, u, vcom, fh);
    });
    for (auto d : *vcs[t])
      csrAddEdgeU(ydeg, yedg, ywei, yoff, c, d, (*vcout[t])[fh(d)]);
  }
}


/**
 * Leiden algorithm's community aggregation phase.
 * @param yoff offsets for vertices belonging to each community (updated)
 * @param ydeg degree of each community (updated)
 * @param yedg vertex ids of outgoing edges of each community (updated)
 * @param ywei weights of outgoing edges of each community (updated)
 * @param bufs buffer for exclusive scan of size |threads| (scratch)
 * @param vcs communities vertex u is linked to (temporary buffer, updated)
 * @param vcout total edge weight from vertex u to community C (temporary buffer, updated)
 * @param x original graph
 * @param vcom community each vertex belongs to
 * @param coff offsets for vertices belonging to each community
 * @param cedg vertices belonging to each community
 * @param fh hash function mapping community to index
 */
template <int CHUNK_SIZE=2048, class G, class K, class W, class FH>
inline void leidenLowmemAggregateOmpW(vector<size_t>& yoff, vector<K>& ydeg, vector<K>& yedg, vector<W>& ywei, vector<size_t>& bufs, vector<vector<K>*>& vcs, vector<vector<W>*>& vcout, const G& x, const vector<K>& vcom, vector<K>& coff, vector<K>& cedg, FH fh) {
  size_t C = coff.size() - 1;
  leidenCommunityTotalDegreeOmpW(yoff, x, vcom);
  yoff[C] = exclusiveScanOmpW(yoff.data(), bufs.data(), yoff.data(), C);
  leidenLowmemAggregateEdgesOmpW<CHUNK_SIZE>(ydeg, yedg, ywei, vcs, vcout, x, vcom, coff, cedg, yoff, fh);
}
#pragma endregion




#pragma region ENVIRONMENT SETUP
/**
 * Setup and perform the Leiden algorithm.
 * @param x original graph
 * @param o leiden options
 * @param fi initializing community membership and total vertex/community weights (vcom, vtot, ctot)
 * @param fm marking affected vertices (vaff, vcs, vcout, vcom, vtot, ctot)
 * @param fa is vertex allowed to be updated? (u)
 * @param fh hash function mapping community to index
 * @returns leiden result
 */
template <bool DYNAMIC=false, bool SELSPLIT=false, int CHUNK_SIZE=2048, class G, class FI, class FM, class FA, class FH>
inline auto leidenLowmemInvokeOmp(const G& x, const LeidenOptions& o, FI fi, FM fm, FA fa, FH fh) {
  using  K = typename G::key_type;
  using  W = LEIDEN_WEIGHT_TYPE;
  using  B = char;
  // Options.
  double R = o.resolution;
  int    L = o.maxIterations, l = 0;
  int    P = o.maxPasses, p = 0;
  // Get graph properties.
  size_t X = x.size();
  size_t S = x.span();
  double M = edgeWeightOmp(x)/2;
  // Allocate buffers.
  int    T = omp_get_max_threads();
  vector<B> vaff(S);        // Affected vertex flag (any pass)
  vector<B> cchg, cspt(S);  // Community changed/split flag (first pass)
  vector<B> bufb;           // Buffer for splitting communities
  vector<K> bufc;           // Buffer for obtaining a vertex from each community
  vector<K> ucom, vcom(S);  // Community membership (first pass, current pass)
  vector<K> udom, vcob(S);  // Old community membership (first pass), Community bound (any pass)
  vector<W> utot, vtot(S);  // Total vertex weights (first pass, current pass)
  vector<W> ctot, dtot;     // Total community weights (any pass)
  vector<W> cdwt;           // Change in total weight of each community
  vector<K> bufk(T);        // Buffer for exclusive scan
  vector<size_t> bufs(T);   // Buffer for exclusive scan
  vector<vector<K>*> vcs(T);    // Hashtable keys
  vector<vector<W>*> vcout(T);  // Hashtable values
  vector<vector<K>*> us(T), vs(T);  // BFS scratch space
  for (int t=0; t<T; ++t) {
    us[t] = new vector<K>();
    vs[t] = new vector<K>();
    us[t]->reserve(S);
    vs[t]->reserve(S);
  }
  if (!DYNAMIC) ucom.resize(S);
  if (!DYNAMIC) utot.resize(S);
  if (!DYNAMIC) ctot.resize(S);
  if (!DYNAMIC) cdwt.resize(S);
  if ( DYNAMIC) udom.resize(S);
  if ( DYNAMIC) cchg.resize(S);
  if ( DYNAMIC) bufb.resize(S);
  if ( DYNAMIC) bufc.resize(S);
  if ( DYNAMIC) dtot.resize(S);
  leidenAllocateHashtablesW(vcs, vcout, o.numSlots);
  size_t Z = max(size_t(o.aggregationTolerance * X), X);
  size_t Y = max(size_t(o.aggregationTolerance * Z), Z);
  DiGraphCsr<K, None, None, K> cv(S, S);  // CSR for community vertices
  DiGraphCsr<K, None, W> y(S, Y);         // CSR for aggregated graph (input);  y(S, X)
  DiGraphCsr<K, None, W> z(S, Z);         // CSR for aggregated graph (output); z(S, X)
  // Perform Leiden algorithm.
  float tm = 0, ti = 0, tp = 0, tl = 0, ts = 0, tr = 0, ta = 0, tt = 0;  // Time spent in different phases
  float t  = measureDurationMarked([&](auto mark) {
    double E  = o.tolerance;
    auto   fc = [&](double el, int l) { return el<=E; };
    // Reset buffers, in case of multiple runs.
    fillValueOmpU(vaff, B());
    fillValueOmpU(cchg, B());
    fillValueOmpU(cspt, B());
    fillValueOmpU(ucom, K());
    fillValueOmpU(vcom, K());
    fillValueOmpU(udom, K());
    fillValueOmpU(vcob, K());
    fillValueOmpU(utot, W());
    fillValueOmpU(vtot, W());
    fillValueOmpU(ctot, W());
    fillValueOmpU(cdwt, W());
    cv.respan(S);
    y .respan(S);
    z .respan(S);
    // Time the algorithm.
    mark([&]() {
      size_t CCHG = 0;
      // Initialize community membership and total vertex/community weights.
      ti += measureDuration([&]() {
        fi(ucom, utot, ctot, cdwt);
        if (DYNAMIC) copyValuesOmpW(udom, ucom);
      });
      // Mark affected vertices.
      tm += measureDuration([&]() {
        CCHG = fm(vaff, cchg, cspt, cdwt, vcs, vcout, ucom, utot, ctot);
      });
      // Start timing first pass.
      auto t0 = timeNow(), t1 = t0;
      // Start local-moving, refinement, aggregation phases.
      // NOTE: In first pass, the input graph is a DiGraph.
      // NOTE: For subsequent passes, the input graph is a DiGraphCsr (optimization).
      for (l=0, p=0; M>0 && P>0;) {
        if (p==1) t1 = timeNow();
        bool isFirst = p==0;
        int m = 0;
        tl += measureDuration([&]() {
          auto fb = [&](auto c) { if (SELSPLIT) cspt[c] = B(1); };
          if (isFirst) m += leidenLowmemMoveOmpW(ucom, ctot, vaff, vcs, vcout, x, vcob, utot, M, R, L, fc, fa, fb, fh);
          else         m += leidenLowmemMoveOmpW(vcom, ctot, vaff, vcs, vcout, y, vcob, vtot, M, R, L, fc, fh);
        });
        size_t CSPT = SELSPLIT && isFirst? countValueOmp(cspt, B(1)) : 0;
        // Adjust community IDs.
        if (DYNAMIC && isFirst && (CSPT || CCHG)) {
          swap(ctot, dtot); swap(ucom, vcob); swap(cchg, vaff); swap(cspt, bufb);
          leidenSubsetRenameCommunitiesOmpW<SELSPLIT>(ucom, ctot, cchg, cspt, bufc, x, vcob, dtot, vaff, bufb);
        }
        ts += measureDuration([&]() {
          if (DYNAMIC && isFirst && (!SELSPLIT || CSPT)) {
            auto fs = [&](auto c) { return (!SELSPLIT || cspt[c]) && !cchg[c]; };
            splitDisconnectedCommunitiesBfsOmpW(vcom, bufb, vaff, us, vs, x, ucom, fs);
            swap(ucom, vcom);
          }
        });
        tr += measureDuration([&]() {
          if (!isFirst || !DYNAMIC || CCHG) {
            auto fr = [&](auto u) { return DYNAMIC? cchg[vcob[u]] : B(1); };
            if (isFirst) copyValuesOmpW(vcob.data(), ucom.data(), x.span());  // swap(vcob, ucom);
            else         copyValuesOmpW(vcob.data(), vcom.data(), y.span());  // swap(vcob, vcom);
            if (isFirst) leidenInitializeOmpW(ucom, ctot, x, utot, fr);
            else         leidenInitializeOmpW(vcom, ctot, y, vtot);
            // if (isFirst) fillValueOmpU(vaff.data(), x.order(), B(1));
            // else         fillValueOmpU(vaff.data(), y.order(), B(1));
            if (isFirst) m += leidenLowmemMoveOmpW<true>(ucom, ctot, vaff, vcs, vcout, x, vcob, utot, M, R, L, fc, fr, fh);
            else         m += leidenLowmemMoveOmpW<true>(vcom, ctot, vaff, vcs, vcout, y, vcob, vtot, M, R, L, fc, fh);
          }
        });
        l += max(m, 1); ++p;
        if ((m<=1 || p>=P) && (!isFirst || !CCHG)) break;
        size_t GN = isFirst? x.order() : y.order();
        size_t CN = 0;
        if (isFirst) CN = leidenCommunityExistsOmpW(cv.degrees, x, ucom);
        else         CN = leidenCommunityExistsOmpW(cv.degrees, y, vcom);
        if (double(CN)/GN >= o.aggregationTolerance) break;
        if (isFirst) leidenRenumberCommunitiesOmpW(ucom, cv.degrees, bufk, x);
        else         leidenRenumberCommunitiesOmpW(vcom, cv.degrees, bufk, y);
        if (isFirst) {}
        else         leidenLookupCommunitiesOmpU(ucom, vcom);
        ta += measureDuration([&]() {
          cv.respan(CN); z.respan(CN);
          if (isFirst) leidenCommunityVerticesOmpW(cv.offsets, cv.degrees, cv.edgeKeys, bufk, x, ucom);
          else         leidenCommunityVerticesOmpW(cv.offsets, cv.degrees, cv.edgeKeys, bufk, y, vcom);
          if (isFirst) leidenLowmemAggregateOmpW<CHUNK_SIZE>(z.offsets, z.degrees, z.edgeKeys, z.edgeValues, bufs, vcs, vcout, x, ucom, cv.offsets, cv.edgeKeys, fh);
          else         leidenLowmemAggregateOmpW<CHUNK_SIZE>(z.offsets, z.degrees, z.edgeKeys, z.edgeValues, bufs, vcs, vcout, y, vcom, cv.offsets, cv.edgeKeys, fh);
        });
        swap(y, z);
        // fillValueOmpU(vcob.data(), CN, K());
        // fillValueOmpU(vcom.data(), CN, K());
        // fillValueOmpU(ctot.data(), CN, W());
        fillValueOmpU(vtot.data(), CN, W());
        fillValueOmpU(vaff.data(), CN, B(1));
        leidenVertexWeightsOmpW(vtot, y);
        leidenInitializeOmpW(vcom, ctot, y, vtot);
        E /= o.toleranceDrop;
      }
      if (p<=1) {}
      else      leidenLookupCommunitiesOmpU(ucom, vcom);
      if (p<=1) t1 = timeNow();
      tp += duration(t0, t1);
      tt += measureDuration([&]() {
        if (DYNAMIC) leidenTrackCommunitiesOmpU(ucom, bufc, vtot, vcob, dtot, x, udom, utot);
      });
    });
  }, o.repeat);
  leidenFreeHashtablesW(vcs, vcout);
  for (int t=0; t<T; ++t) {
    delete us[t];
    delete vs[t];
  }
  return LeidenResult<K>(ucom, utot, ctot, cdwt, l, p, t, tm/o.repeat, ti/o.repeat, tp/o.repeat, tl/o.repeat, ts/o.repeat, tr/o.repeat, ta/o.repeat, tt/o.repeat, countValueOmp(vaff, B(1)));
}
#pragma endregion




#pragma region STATIC APPROACH
/**
 * Obtain the community membership of each vertex with Static Leiden.
 * @tparam SLOTS number of slots in each hashtable
 * @param x original graph
 * @param o leiden options
 * @param fh hash function mapping community to index
 * @returns leiden result
 */
template <class G, class FH>
inline auto leidenLowmemStaticOmp(const G& x, const LeidenOptions& o, FH fh) {
  using B = char;
  using W = LEIDEN_WEIGHT_TYPE;
  auto fi = [&](auto& vcom, auto& vtot, auto& ctot, auto& cdwt) {
    leidenVertexWeightsOmpW(vtot, x);
    leidenInitializeOmpW(vcom, ctot, x, vtot);
    fillValueOmpU(cdwt, W());
  };
  auto fm = [ ](auto& vaff, auto& cchg, auto& cspt, auto& cdwt, auto& vcs, auto& vcout, const auto& vcom, const auto& vtot, const auto& ctot) {
    fillValueOmpU(vaff, B(1));
    return size_t(1);
  };
  auto fa = [ ](auto u) { return true; };
  return leidenLowmemInvokeOmp(x, o, fi, fm, fa, fh);
}
#pragma endregion
#pragma endregion
