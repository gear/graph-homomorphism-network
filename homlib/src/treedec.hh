#pragma once

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <queue>
#include <tuple>
#include <algorithm>
#include "graph.hh"


/*
template <class T>
std::ostream &operator<<(std::ostream &os, const std::vector<T> &v) {
  os << "[";
  for (int i = 0; i < v.size(); os << v[i++]) 
    if (i > 0) os << " ";
  os << "]";
  return os;
}
template <class T>
std::ostream &operator<<(std::ostream &os, const std::vector<std::vector<T>> &v) {
  os << "[";
  for (int i = 0; i < v.size(); os << v[i++]) 
    if (i > 0) os << std::endl << " ";
  os << "]";
  return os;
}
*/

struct NiceTreeDecomposition {
  enum { INTRODUCE, FORGET, JOIN };
  std::vector<std::tuple<int,int,int>> nodes;
  int root;
  int type(int x)   const { return std::get<0>(nodes[x]); }
  int vertex(int x) const { return std::get<1>(nodes[x]); }
  int child(int x)  const { return std::get<2>(nodes[x]); }
  int left(int x)   const { return std::get<1>(nodes[x]); }
  int right(int x)  const { return std::get<2>(nodes[x]); }

  bool isLeaf(int x) const { return type(x) == INTRODUCE && child(x) == -1; }
  bool isIntroduce(int x) const { return type(x) == INTRODUCE; }
  bool isJoin(int x) const { return type(x) == JOIN; }
  bool isForget(int x) const { return type(x) == FORGET; }

  void display() {
    display(root, 0);
  }
  void display(int x, int tab) {
    if (x == -1) return;
    if (type(x) == INTRODUCE) {
      std::cout << std::string(tab, ' ') << x << ": Introduce " << vertex(x) << std::endl;
      display(child(x), tab+2);
    } else if (type(x) == FORGET) {
      std::cout << std::string(tab, ' ') << x << ": Forget " << vertex(x) << std::endl;
      display(child(x), tab+2);
    } else if (type(x) == JOIN) {
      std::cout << std::string(tab, ' ') << x << ": Join" << std::endl;
      display(left(x), tab+2);
      display(right(x), tab+2);
    }
  }
};

// Greedy Elimination Ordering
NiceTreeDecomposition niceTreeDecomposition(Graph G) {
  // utility functions
  auto set_difference = [&](std::vector<int> A, std::vector<int> B) {
    std::vector<int> C;
    std::set_difference(A.begin(), A.end(), B.begin(), B.end(), std::back_inserter(C));
    return C;
  };
  auto set_union = [&](std::vector<int> A, std::vector<int> B) {
    std::vector<int> C;
    std::set_union(A.begin(), A.end(), B.begin(), B.end(), std::back_inserter(C));
    return C;
  };

  // implementation
  int n = G.n;
  std::vector<std::vector<int>> nbh = G.adj;
  for (int u = 0; u < n; ++u) {
    nbh[u].push_back(u);
    sort(nbh[u].begin(), nbh[u].end());
    nbh[u].erase(unique(nbh[u].begin(), nbh[u].end()), nbh[u].end());
  }

  // greedy elimination ordering
  std::vector<int> order;
  std::vector<std::vector<int>> X;

  std::priority_queue<std::pair<int,int>, 
                     std::vector<std::pair<int,int>>,
                     std::greater<std::pair<int,int>>> que;
  for (int u = 0; u < n; ++u) {
    que.push(std::make_pair(nbh[u].size(), u));
  }
  while (!que.empty()) {
    auto [deg, u] = que.top();
    que.pop();
    if (deg != nbh[u].size()) continue;
    order.push_back(u);
    bool is_maximal = true;
    for (int i = 0; i < X.size(); ++i) {
      if (set_difference(nbh[u], X[i]).empty()) {
        is_maximal = false;
        break;
      }
    }
    if (is_maximal) X.push_back(nbh[u]);
    for (int v: nbh[u]) {
      if (u == v) continue;
      std::vector<int> U = set_union(nbh[u], nbh[v]);
      U.erase(std::remove(U.begin(), U.end(), u), U.end());
      U.swap(nbh[v]);
      if (nbh[v].size() < U.size()) que.push(std::make_pair(nbh[v].size(), v));
    }
  }
  /*
  for (int i = 0; i < X.size(); ++i) {
    std::cout << i << ": " << X[i] << std::endl;
  }
  */

  // from elimination ordering to nice tree decomposition
  NiceTreeDecomposition NTD;
  std::vector<std::vector<int>> child(X.size());
  std::vector<int> head(X.size(), -1);

  for (int i = 0; i < X.size(); ++i) {
    for (int j: child[i]) {
      for (int u: set_difference(X[j], X[i])) {
        NTD.nodes.push_back(std::make_tuple(NiceTreeDecomposition::FORGET, u, head[j]));
        head[j] = NTD.nodes.size() - 1;
      }
      for (int u: set_difference(X[i], X[j])) {
        NTD.nodes.push_back(std::make_tuple(NiceTreeDecomposition::INTRODUCE, u, head[j]));
        head[j] = NTD.nodes.size() - 1;
      }
      if (head[i] == -1) {
        head[i] = head[j];
      } else {
        NTD.nodes.push_back(std::make_tuple(NiceTreeDecomposition::JOIN, head[i], head[j]));
        head[i] = NTD.nodes.size() - 1;
      }
    }
    if (head[i] == -1) {
      for (int u: X[i]) {
        NTD.nodes.push_back(std::make_tuple(NiceTreeDecomposition::INTRODUCE, u, head[i]));
        head[i] = NTD.nodes.size() - 1;
      }
    }
    for (int j = i+1; j < X.size(); ++j) {
      if (set_difference(X[i], X[j]).size() == 1) {
        child[j].push_back(i);
        break;
      }
    }
  }
  for (int u: X.back()) {
    NTD.nodes.push_back(std::make_tuple(NiceTreeDecomposition::FORGET, u, head.back()));
    head.back() = NTD.nodes.size() - 1;
  }
  NTD.root = head.back();
  return NTD;
}
