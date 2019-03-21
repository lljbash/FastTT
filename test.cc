#include <cstdlib>
#include <cassert>
#include <iostream>
#include <fstream>
#include <chrono>
#include <functional>
#include <vector>
#include <string>
#include <random>
#include <boost/algorithm/string.hpp>
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/cuthill_mckee_ordering.hpp>
#include <boost/graph/properties.hpp>
#include <boost/graph/bandwidth.hpp>
#include <cxxopts.hpp>
#include <xerus/algorithms/randomSVD.h>
#include "sptensor2tt.h"

using namespace std;
using namespace xerus;

void error(string message = "Error!") {
    cerr << message << endl;
    exit(1);
}

void run_test(const std::function<TTTensor(const Tensor&)> &f, const Tensor &x, ostream &sout, ostream &vout) {
    auto c_begin = clock();
    auto begin = chrono::high_resolution_clock::now();
    auto tt = f(x);
    auto c_end = clock();
    auto end = chrono::high_resolution_clock::now();
    auto xx = Tensor(tt);
    auto eps = (x - xx).frob_norm() / x.frob_norm();
    
    vout << "time: " << chrono::duration_cast<chrono::milliseconds>(end - begin).count() << "ms"  << endl;
    vout << "cputime: " << 1000.0 * (c_end-c_begin) / CLOCKS_PER_SEC << "ms" << endl;
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
        ("f,file", "Input file name", cxxopts::value<string>())
        ("t,type", "Input file type: graph / image / tensor", cxxopts::value<string>()->default_value("unspecific"))
        ("U,undirected", "If input graph is undirected")
        ("O,obeserved", "The obeservation ratio of the image", cxxopts::value<double>()->default_value("0.01"))
        ("R,random", "Use random generated n^d tesors as input instead")
        ("n", "Parameter n of the tensor", cxxopts::value<int>()->default_value("4"))
        ("d", "Parameter d of the tensor", cxxopts::value<int>()->default_value("10"))
        ("l,n_list", "Use a list of n instead of n^d", cxxopts::value<string>())
        ("N,nnz", "The number of nonzero elements of the random generated tensor", cxxopts::value<int>()->default_value("500"))
        ("F,fixed_rank", "Generate fixed-rank tesors", cxxopts::value<int>()->default_value("0"))
        ("s,sparsity", "The sparsity of generated cores", cxxopts::value<double>()->default_value("0.02"))
        ("p", "Parameter p of FastTT", cxxopts::value<int>()->default_value("-1"))
        ("r,max_rank", "Max ranks of the target tensor train", cxxopts::value<int>()->default_value(to_string(numeric_limits<int>::max())))
        ("e,epsilon", "Desired tolerated relative error", cxxopts::value<double>()->default_value("1e-14"))
        ("ttsvd", "Test TT-SVD")
        ("rttsvd", "Test Randomized TT-SVD for given target rank", cxxopts::value<int>()->implicit_value("10"))
        ("nofasttt", "Do not test FastTT")
        ("S,simple", "Output simple result")
        ("save", "Save the random tensor as a csv file", cxxopts::value<string>()->default_value("backup.csv"))
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
    vector<size_t> n_list;
    if (args.count("n_list")) {
        string n_list_str = args["n_list"].as<string>();
        vector<string> n_str_list;
        boost::split(n_str_list, n_list_str, [](char c) { return c < '0' || c > '9'; });
        for (string n_str: n_str_list) {
            if (!n_str.empty()) {
                n_list.push_back(stoi(n_str));
            }
        }
    }
    else {
        int n = args["n"].as<int>();
        int d = args["d"].as<int>();
        if (!(d > 0 && n > 0)) {
            error("n and d must be positive integers!");
        }
        n_list = vector<size_t>(d, n);
    }
    int d = n_list.size();
    int m = 1;
    for (int n : n_list) {
        m *= n;
    }
    Tensor x;
    int N = 0;
    string type = args["type"].as<string>();
    if (args.count("random")) {
        N = args["N"].as<int>();
        double sp = args["s"].as<double>();
        int r = args["fixed_rank"].as<int>();
        if (!(N >= 0)) {
            error("N must be a positive integer!");
        }
        if (!(sp > 0 && sp < 1)) {
            error("sp must be a real number between 0 and 1!");
        }
        if (!(r > 0)) {
            x = Tensor::random(vector<size_t>(n_list), static_cast<size_t>(N));
        }
        else {
            x = Tensor::random({n_list.front(), static_cast<size_t>(r)}, static_cast<size_t>(n_list.front() * r * sp));
            for (int i = 1; i < d - 1; ++i) {
                auto n = n_list.at(i);
                auto y = Tensor::random({static_cast<size_t>(r), n, static_cast<size_t>(r)},
                                        static_cast<size_t>(r * n * r * sp));
                contract(x, x, y, 1);
            }
            auto y = Tensor::random({static_cast<size_t>(r), n_list.back()}, static_cast<size_t>(r * n_list.back() * sp));
            contract(x, x, y, 1);
        }
        x.use_sparse_representation();
        N = x.get_sparse_data().size();
        if (args.count("save")) {
            ofstream fout(args["save"].as<string>());
            fout.precision(std::numeric_limits<value_t>::digits10 + 3);
            for (auto [poistion, value] : x.get_sparse_data()) {
                fout << poistion << "\t" << value << endl;
            }
        }
    }
    else if (type == "graph") {
        ifstream fin(args["file"].as<string>());
        if (!fin.is_open()) {
            cerr << "Cannot open file " << args["file"].as<string>() << endl;
            exit(-1);
        }
        auto nn_list = n_list;
        nn_list.insert(nn_list.end(), n_list.begin(), n_list.end());
        x = Tensor(nn_list);
        
        vector<pair<size_t, size_t>> edges;
        auto index = [n_list, d](int a, int b) {
            vector<size_t> ret;
            for (int i = 0; i < d; ++i) {
                auto n = n_list[i];
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
            int a, b;
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
        for (auto &n : n_list) {
            n *= n;
        }
        m *= m;
        x.reinterpret_dimensions(n_list);
        x.use_sparse_representation();
        N = x.get_sparse_data().size();
    }
    else if (type == "image") {
        ifstream fin(args["file"].as<string>());
        if (!fin.is_open()) {
            cerr << "Cannot open file " << args["file"].as<string>() << endl;
            exit(-1);
        }
        double obeserved = args["obeserved"].as<double>();
        int channels = 3;
        int sz = m / channels;
        if (!(obeserved > 0 && obeserved < 1)) {
            error("The obeservation ratio must be a real number between 0 and 1!");
        }
        vector<bool> mask(sz, false);
        N = static_cast<int>(floor(static_cast<double>(sz) * obeserved));
        fill_n(mask.begin(), N, true);
        N *= channels;
        random_device rd;
        mt19937 gen(rd());
        shuffle(mask.begin(), mask.end(), gen);
        map<size_t, value_t> x_data;
        for (int i = 0; i < m; ++i) {
            int pixel;
            fin >> pixel;
            if (mask[i/channels]) {
                x_data.try_emplace(i, pixel);
            }
        }
        x = Tensor(n_list, Tensor::Representation::Sparse, Tensor::Initialisation::None);
        x.get_unsanitized_sparse_data() = move(x_data);
    }
    else if (type == "tensor") {
        ifstream fin(args["file"].as<string>());
        if (!fin.is_open()) {
            cerr << "Cannot open file " << args["file"].as<string>() << endl;
            exit(-1);
        }
        map<size_t, value_t> x_data;
        for (string line; getline(fin, line); ) {
            istringstream line_in(line);
            size_t position;
            value_t value;
            line_in >> position >> value;
            x_data.try_emplace(position, value);
        }
        N = x_data.size();
        x = Tensor(n_list, Tensor::Representation::Sparse, Tensor::Initialisation::None);
        x.get_unsanitized_sparse_data() = move(x_data);
    }
    else {
        error("You must specific a valid file type");
    }

    int r = args["r"].as<int>();
    if (r < 0) {
        error("Max ranks must be positive!");
    }
    double eps = args["e"].as<double>();
    if (eps <= 0) {
        error("Epsilon must be positive!");
    }

    const bool simple = args.count("simple");
    ofstream nout("/dev/null");
    ostream &sout = simple ? cout : nout;
    ostream &vout = !simple ? cout : nout;
    
    int vpos = args["p"].as<int>();
    string n_list_str = "[";
    for (int n : n_list) {
        n_list_str.append(to_string(n));
        n_list_str.append(", ");
    }
    n_list_str.erase(n_list_str.size() - 2);
    n_list_str.push_back(']');
    vout << "n = " << n_list_str << ", d = " << d << ", N = " << N << endl;
    vout << "sparse: " << static_cast<double>(N) / m << endl;
    
    if (!args.count("nofasttt")) {
        vout << "--------------------FastTT-------------------" << endl;
        run_test([vpos, r, eps](auto &&x) { return sptensor2tt(x, vpos, r, eps); }, x, sout, vout);
    }

    if (args.count("ttsvd")) {
        vout << "--------------------TTSVD--------------------" << endl;
        auto y(x);
        y.use_dense_representation();
        run_test([r, eps](auto &&x) { return TTTensor(x, eps, r); }, y, sout, vout);
    }
    else {
        sout << 0 << endl;
    }
    
    if (args.count("rttsvd")) {
        vout << "----------------Random TTSVD-----------------" << endl;
        auto y(x);
        y.use_dense_representation();
        int r = args["rttsvd"].as<int>();
        if (r < 0) {
            error("Target ranks must be positive!");
        }
        run_test([d, r](auto &&x) { return randomTTSVD(x, vector<size_t>(d-1, r), vector<size_t>(d-1, 0)); }, y, sout, vout);
    }
    else {
        sout << 0 << endl;
    }
    
    return 0;
}

