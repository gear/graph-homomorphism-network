#include <iostream>
#include <algorithm>
#include "hom.hh"

void test_count_vertex() {
  Graph T(1);
  int n = 20;
  Graph G(n);
  for (int i = 0; i < n; ++i) {
    for (int j = i+1; j < n; ++j) {
      if (rand() % 2 == 0) {
        G.addEdge(i, j);
      }
    }
  }
  assert(hom<int>(T, G) == n);
}

void test_count_edge() {
  Graph T(2);
  T.addEdge(0, 1);
  int n = 20, m = 0;
  Graph G(n);
  for (int i = 0; i < n; ++i) {
    for (int j = i+1; j < n; ++j) {
      if (rand() % 2 == 0) {
        G.addEdge(i, j);
        ++m;
      }
    }
  }
  assert(hom<int>(T, G) == 2 * m);
}

void test_count_triangle() {
  Graph T(3);
  T.addEdge(0, 1);
  T.addEdge(1, 2);
  T.addEdge(2, 0);

  int n = 20;
  Graph G(n);
  for (int i = 0; i < n; ++i) {
    for (int j = i+1; j < n; ++j) {
      if (rand() % 2 == 0) {
        G.addEdge(i, j);
      }
    }
  }

  int triangles = 0;
  for (int u = 0; u < n; ++u) {
    for (int v: G.adj[u]) {
      if (v < u) continue;
      std::vector<int> common;
      std::set_intersection(
          G.adj[u].begin(), G.adj[u].end(), 
          G.adj[v].begin(), G.adj[v].end(),
          back_inserter(common));
      triangles += count_if(common.begin(), common.end(), [&](int a) { return a > v; });
    }
  }
  assert(hom<int>(T, G) == 6 * triangles);
}

void test2() {
  Graph T(4);
  T.addEdge(0, 1);
  T.addEdge(1, 2);
  T.addEdge(2, 3);
  T.addEdge(3, 0);
  Graph G(2);
  G.addEdge(0,1);
  assert(hom<int>(T, G) == 2);
}

void test_tree() {
  int t = 5;
  Graph T(t);
  for (int i = 1; i < t; ++i) {
    T.addEdge(rand() % i, i);
  }
  int n = 20;
  Graph G(n);
  for (int i = 0; i < n; ++i) {
    for (int j = i+1; j < n; ++j) {
      if (rand() % 2 == 0) {
        G.addEdge(i, j);
      }
    }
  }
  HomomorphismCounting<int> hom(T, G);
  HomomorphismCountingTree<int> homTree(T, G);
  assert(hom.run() == homTree.run());
}

int main() {
  srand(time(0));
  test_count_vertex();
  test_count_edge();
  test_count_triangle();
  test2();
  test_tree();
}
