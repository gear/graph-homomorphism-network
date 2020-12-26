#include <iostream>
#include <algorithm>
#include "hom.hh"

// === tick a time ===
#include <ctime>
double tick() {
  static clock_t oldtick;
  clock_t newtick = clock();
  double diff = 1.0*(newtick - oldtick) / CLOCKS_PER_SEC;
  oldtick = newtick;
  return diff;
}

void test_count_vertex() {
  Graph T(1);
  int n = 1000;
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
  int n = 1000, m = 0;
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

  int n = 100;
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

void test_count_tree() {
  int t = 10;
  Graph T(t);
  for (int i = 1; i < t; ++i) {
    T.addEdge(rand() % i, i);
  }

  int n = 10000, m = 1000000;
  Graph G(n);
  std::set<std::pair<int,int>> edges;
  while (edges.size() < m) {
    int u = 1 + (rand() % (n-1)), v = rand() % u;
    edges.insert(std::make_pair(u, v));
  }
  for (auto [u, v]: edges) {
    G.addEdge(u, v);
  }
  hom<int>(T, G);
}

int main() {
  srand(time(0));
  tick();
  test_count_vertex();
  std::cout << tick() << std::endl;
  test_count_edge();
  std::cout << tick() << std::endl;
  test_count_triangle();
  std::cout << tick() << std::endl;
  test_count_tree();
  std::cout << tick() << std::endl;
}
