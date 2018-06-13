#include <cstdlib>
#include <cassert>
#include <iostream>
#include <fstream>
#include <chrono>
#include <functional>
#include <string>
#include <cxxopts.hpp>
#include <xerus/algorithms/randomSVD.h>
#include "tensor2tt_lossless.h"

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
        ("R,random", "Use random generated n^d tesors as input instead")
        ("n", "Parameter n of the random generated tensor", cxxopts::value<int>()->default_value("4"))
        ("d", "Parameter d of the random generated tensor", cxxopts::value<int>()->default_value("10"))
        ("N,nnz", "The number of nonzero elements of the random generated tensor", cxxopts::value<int>()->default_value("500"))
        ("F,fixed_rank", "Generate fixed-rank tesors")
        ("r,rank", "The TT-ranks of generated tensors", cxxopts::value<int>()->default_value("50"))
        ("s,sparsity", "The sparsity of generated cores", cxxopts::value<double>()->default_value("0.02"))
        ("ttsvd", "Test TT-SVD")
        ("rttsvd", "Test Randomized TT-SVD")
        ("S,simple", "Output simple result")
        ;
    const auto args = [&options, &argc, &argv]() {
        try {
            return options.parse(argc, argv);
        }
        catch (cxxopts::OptionParseException) {
            cout << options.help() << endl;
            exit(1);
        }
    } ();
    int n;
    int d;
    int N;
    int r;
    double sp;
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
        return 1;
    }
    const bool simple = args.count("simple");
    ofstream nout("/dev/null");
    ostream &sout = simple ? cout : nout;
    ostream &vout = !simple ? cout : nout;
    
    int vpos = d / 2;
    vout << "n = " << n << ", d = " << d << ", N = " << N << endl;
    vout << "sparse: " << static_cast<double>(N) / pow(n, d) << endl;
    
    vout << "--------------------FLATT--------------------" << endl;
    run_test([vpos](auto &&x) { return tensor2tt_lossless(x, vpos); }, x, sout, vout);

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

