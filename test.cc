#include <cstdlib>
#include <cassert>
#include <iostream>
#include <fstream>
#include <chrono>
#include <functional>
#include <string>
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/cuthill_mckee_ordering.hpp>
#include <boost/graph/properties.hpp>
#include <boost/graph/bandwidth.hpp>
#include <cxxopts.hpp>
#include <xerus/algorithms/randomSVD.h>
#include "sptensor2tt.h"

using namespace std;
using namespace xerus;

void run_test(const std::function<TTTensor(const Tensor&)> &f, const Tensor &x, ostream &sout, ostream &vout) {
    auto begin = chrono::high_resolution_clock::now();
    auto tt = f(x);
    auto end = chrono::high_resolution_clock::now();
    auto xx = Tensor(tt);
    auto eps = (x - xx).frob_norm() / x.frob_norm();
    
    vout << "time: " << chrono::duration_cast<chrono::milliseconds>(end - begin).count() << "ms"  << endl;
    sout << chrono::duration_cast<chrono::milliseconds>(end - begin).count() << endl;
    vout << "eps: " << setprecision(10) << eps << endl;
    vout << "ranks: ";
    for (auto r : tt.ranks()) {
        vout << r << " ";
    }
    vout << endl;
}

int main(int argc, char *argv[]) {
    cxxopts::Options options("test", "Test fast T2TT.");
    options.add_options()
        ("f,file", "Input file name", cxxopts::value<std::string>())
        ("U, undirected", "If input graph is undirected ")
        ("R,random", "Use random generated n^d tesors as input instead")
        ("n", "Parameter n of the tensor", cxxopts::value<int>()->default_value("4"))
        ("d", "Parameter d of the tensor", cxxopts::value<int>()->default_value("10"))
        ("N,nnz", "The number of nonzero elements of the random generated tensor", cxxopts::value<int>()->default_value("500"))
        ("F,fixed_rank", "Generate fixed-rank tesors")
        ("r,rank", "The TT-ranks of generated tensors", cxxopts::value<int>()->default_value("50"))
        ("s,sparsity", "The sparsity of generated cores", cxxopts::value<double>()->default_value("0.02"))
        ("p", "Parameter p of FastTT", cxxopts::value<int>()->default_value("-1"))
        ("ttsvd", "Test TT-SVD")
        ("rttsvd", "Test Randomized TT-SVD")
        ("S,simple", "Output simple result")
        ;
    const auto args = [&options, &argc, &argv]() {
        try {
            return options.parse(argc, argv);
        }
        catch (cxxopts::OptionParseException &) {
            cout << options.help() << endl;
            exit(1);
        }
    } ();
    int n = 0;
    int d = 0;
    int N = 0;
    int r = 0;
    double sp = 0;
    Tensor x;
    if (args.count("random")) {
        n = args["n"].as<int>();
        d = args["d"].as<int>();
        N = args["N"].as<int>();
        r = args["r"].as<int>();
        sp = args["s"].as<double>();
        if (!(d > 0 && n > 0 && N >= 0 && r > 0 && sp > 0 && sp < 1)) {
            cerr << "Invalid args!" << endl;
            return 1;
        }
        if (!args.count("fixed_rank")) {
            x = Tensor::random(vector<size_t>(d, n), static_cast<size_t>(N));
        }
        else {
            x = Tensor::random({static_cast<size_t>(n), static_cast<size_t>(r)}, n * r * sp);
            for (int i = 1; i < d - 1; ++i) {
                auto y = Tensor::random({static_cast<size_t>(r), static_cast<size_t>(n), static_cast<size_t>(r)},
                                        r * n * r * sp);
                contract(x, x, y, 1);
            }
            auto y = Tensor::random({static_cast<size_t>(r), static_cast<size_t>(n)}, r * n * sp);
            contract(x, x, y, 1);
        }
        x.use_sparse_representation();
        N = x.get_sparse_data().size();
    }
    else {
        ifstream fin(args["file"].as<string>());
        if (!fin.is_open()) {
            cerr << "Cannot open file " << args["file"].as<string>() << endl;
            return -1;
        }
        n = args["n"].as<int>();
        d = args["d"].as<int>();
        size_t m = pow(n, d);
        x = Tensor(vector<size_t>(d * 2, n));
        
        vector<pair<size_t, size_t>> edges;
        auto index = [n, d](int a, int b) {
            vector<size_t> ret;
            for (int i = 0; i < d; ++i) {
                ret.push_back(b % n);
                ret.push_back(a % n);
                b /= n;
                a /= n;
                reverse(ret.begin(), ret.end());
            }
            return ret;
        };
        for (string line; getline(fin, line); ) {
            line.erase(line.begin(), std::find_if(line.begin(), line.end(), [](int ch) {
                return !std::isspace(ch);
            }));
            line.erase(std::find_if(line.rbegin(), line.rend(), [](int ch) {
                return !std::isspace(ch);
            }).base(), line.end());
            if (line.length() < 3 || line.front() == '#') {
                continue;
            }
            istringstream line_in(line);
            size_t a, b;
            line_in >> a >> b;
            if (a >= m || b >= m) {
                continue;
            }
            edges.emplace_back(a, b);
        }
        if (!args.count("undirected")) {
            for (const auto e : edges) {
                x[index(e.first, e.second)] = 1;
            }
        }
        else {
            using namespace boost;
            typedef adjacency_list<vecS, vecS, undirectedS,
                property<vertex_color_t, default_color_type,
                property<vertex_degree_t, int> > > Graph;
            typedef graph_traits<Graph>::vertex_descriptor Vertex;
            typedef graph_traits<Graph>::vertices_size_type size_type;
            Graph G(m);
            for (size_t i = 0; i < edges.size(); ++i) {
                add_edge(edges[i].first, edges[i].second, G);
            }
            std::vector<Vertex> inv_perm(num_vertices(G));
            std::vector<size_type> perm(num_vertices(G));
            cuthill_mckee_ordering(G, inv_perm.rbegin(), get(vertex_color, G), make_degree_map(G));
            property_map<Graph, vertex_index_t>::type index_map = get(vertex_index, G);
            for (size_type c = 0; c != inv_perm.size(); ++c) {
                perm[index_map[inv_perm[c]]] = c;
            }
            for (const auto &e : edges) {
                int a = perm[e.first];
                int b = perm[e.second];
                x[index(a, b)] = 1;
                x[index(b, a)] = 1;
            }
        }
        x.reinterpret_dimensions(vector<size_t>(d, n * n));
        x.use_sparse_representation();
        n = n * n;
        N = x.get_sparse_data().size();
    }
    const bool simple = args.count("simple");
    ofstream nout("/dev/null");
    ostream &sout = simple ? cout : nout;
    ostream &vout = !simple ? cout : nout;
    
    int vpos = args["p"].as<int>();
    if (vpos < 0) {
        vpos = d / 2;
    }
    vout << "n = " << n << ", d = " << d << ", N = " << N << endl;
    vout << "sparse: " << static_cast<double>(N) / pow(n, d) << endl;
    
    vout << "--------------------FLATT--------------------" << endl;
    run_test([vpos](auto &&x) { return sptensor2tt(x, vpos); }, x, sout, vout);

    if (args.count("ttsvd")) {
        vout << "--------------------TTSVD--------------------" << endl;
        auto y(x);
        y.use_dense_representation();
        run_test([](auto &&x) { return TTTensor(x); }, y, sout, vout);
    }
    else {
        sout << 0 << endl;
    }
    
    if (args.count("rttsvd")) {
        vout << "----------------Random TTSVD-----------------" << endl;
        auto y(x);
        y.use_dense_representation();
        run_test([d, r](auto &&x) { return randomTTSVD(x, vector<size_t>(d-1, r), vector<size_t>(d-1, 0)); }, y, sout, vout);
    }
    else {
        sout << 0 << endl;
    }
    
    return 0;
}

