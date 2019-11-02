// g++ -O3 -fmax-errors=1 -fsanitize=undefined -fsanitize=address -sanitize=leak\
-std=c++17
//
// https://core.ac.uk/download/pdf/82380292.pdf
//
#include <bits/stdc++.h>

using namespace std;

struct Graph {
  int n;
  vector<vector<int>> adj;
  Graph(int n) : n(n), adj(n) { }
  void addEdge(int u, int v) {
    adj[u].push_back(v);
    adj[v].push_back(u);
  }
};

struct NiceTreeDecomposition {
  enum { INTRODUCE, FORGET, JOIN };
  vector<tuple<int,int,int>> nodes;
  int root;
  int type(int x) { return get<0>(nodes[x]); }
  int vertex(int x) { return get<1>(nodes[x]); };
  int child(int x) { return get<2>(nodes[x]); };
  int left(int x) { return get<1>(nodes[x]); }
  int right(int x) { return get<2>(nodes[x]); }


  void fromTree(Graph T, int u) { 
    nodes.clear();
    nodes.push_back(make_tuple(FORGET, u, fromTree(T, u, -1)));
    root = nodes.size() - 1;
  }
  int fromTree(Graph T, int u, int p) {
    int x = -1;
    for (int v: T.adj[u]) {
      if (v == p) continue;
      int z = fromTree(T, v, u);
      int y = nodes.size();
      nodes.push_back(make_tuple(FORGET, v, y+1));
      nodes.push_back(make_tuple(INTRODUCE, u, z));
      if (x == -1) {
        x = y;
      } else {
        nodes.push_back(make_tuple(JOIN, x, y));
        x = nodes.size() - 1;
      }
    }
    if (x >= 0) return x;
    nodes.push_back(make_tuple(INTRODUCE, u, -1));
    return nodes.size() - 1;
  }

  void display() {
    display(root, 0);
  }
  void display(int x, int tab) {
    if (x == -1) return;
    if (type(x) == INTRODUCE) {
      cout << string(tab, ' ') << "Introduce " << vertex(x) << endl;
      display(child(x), tab+2);
    } else if (type(x) == FORGET) {
      cout << string(tab, ' ') << "Forget " << vertex(x) << endl;
      display(child(x), tab+2);
    } else if (type(x) == JOIN) {
      cout << string(tab, ' ') << "Join" << endl;
      display(left(x), tab+2);
      display(right(x), tab+2);
    }
  }
};

using PhiType = map<int,int>;

template <class It>
bool next_radix(It begin, It end, int base) {
  for (It cur = begin; cur != end; ++cur) {
    if ((*cur += 1) >= base) *cur = 0;
    else return true;
  }
  return false;
}
vector<PhiType> allMaps(vector<int> X, int n) {
  if (X.empty()) return {PhiType()};
  vector<PhiType> maps;
  vector<int> v(X.size()); 
  do {
    PhiType M;
    for (int i = 0; i < X.size(); ++i) M[X[i]] = v[i];
    maps.push_back(M);
  } while (next_radix(v.begin(), v.end(), n));
  return maps;
}

void display(map<PhiType,int> I) {
  return;
  for (auto [x,y]: I) {
    cout << "{";
    bool first = true;
    for (auto [u,v]: x) {
      if (!first) cout << ", ";
      first = false;
      cout << u << ": " << v;
    }
    cout << "} => " << y << "; ";
  }
  cout << endl;
}

// Thus far, F should be a tree
struct HomomorphismCounting {
  Graph F, G;
  NiceTreeDecomposition NTD;

  HomomorphismCounting(Graph F_, Graph G_) : F(F_), G(G_) {
    for (int a = 0; a < G.n; ++a) {
      sort(G.adj[a].begin(), G.adj[a].end());
    }
    for (int u = 0; u < F.n; ++u) {
      sort(F.adj[u].begin(), F.adj[u].end());
    }
    NTD.fromTree(F, 0);
    NTD.display();
  }
  vector<int> X;
  int run() {
    X.clear();
    auto [I, X] = run(NTD.root);
    return I[PhiType()];
  }
  pair<map<PhiType,int>, vector<int>> run(int x) {
    map<PhiType,int> I;
    if (NTD.type(x) == NiceTreeDecomposition::INTRODUCE && NTD.child(x) == -1) {
      vector<int> X = {NTD.vertex(x)};
      for (auto phi: allMaps(X, G.n)) {
        I[phi] = 1;
      }
      display(I);
      return make_pair(I, X);
    } 
    if (NTD.type(x) == NiceTreeDecomposition::INTRODUCE) {
      auto [J, X] = run(NTD.child(x));
      for (auto phi: allMaps(X, G.n)) {
        vector<int> nbh;
        for (int u: F.adj[NTD.vertex(x)]) {
          if (binary_search(X.begin(), X.end(), u)) {
            nbh.push_back(phi[u]);
          }
        }
        auto psi = phi;
        for (int a = 0; a < G.n; ++a) {
          psi[NTD.vertex(x)] = a;
          bool condition = true;
          for (int b: nbh) {
            if (!binary_search(G.adj[a].begin(), G.adj[a].end(), b)) {
              condition = false;
              break;
            }
          }
          if (condition) {
            I[psi] = J[phi];
          }
        }
      }
      int v = NTD.vertex(x);
      X.insert(lower_bound(X.begin(), X.end(), NTD.vertex(x)), NTD.vertex(x));
      display(I);
      return make_pair(I, X);
    } 
    if (NTD.type(x) == NiceTreeDecomposition::FORGET) {
      auto [J, X] = run(NTD.child(x));
      X.erase(lower_bound(X.begin(), X.end(), NTD.vertex(x)));
      for (auto phi: allMaps(X, G.n)) {
        auto psi = phi;
        for (int a = 0; a < G.n; ++a) {
          psi[NTD.vertex(x)] = a;
          I[phi] += J[psi];
        }
      }
      display(I);
      return make_pair(I, X);
    } 
    if (NTD.type(x) == NiceTreeDecomposition::JOIN) {
      auto [J, X] = run(NTD.left(x));
      auto [K, Y] = run(NTD.right(x));
      //assert(X == Y);
      for (auto phi: allMaps(X, G.n)) {
        I[phi] = J[phi] * K[phi];
      }
      return make_pair(I, X);
    }
    assert(false);
  }
};

long long hom(Graph F, Graph G) {
  HomomorphismCounting solver(F, G);
  return solver.run();
}


int main() {
  int t = 10;
  Graph T(t);
  for (int i = 1; i < t; ++i) 
    T.addEdge(rand() % i, i);

  int n = 100;
  Graph G(n);
  for (int i = 0; i < n; ++i) 
    for (int j = i+1; j < n; ++j) 
      if (rand() % 3 == 0) 
        G.addEdge(i, j);

  cout << hom(T, G) << endl;
}
