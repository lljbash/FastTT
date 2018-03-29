#include <cstdlib>
#include <cassert>
#include <iostream>
#include <fstream>
#include <chrono>
#include <functional>
#include <string>
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
    
    //for (size_t i = 0; i < tt.degree(); ++i) {
        //auto comp = tt.get_component(i);
        //comp.reinterpret_dimensions({comp.dimensions.at(0), comp.dimensions.at(1) * comp.dimensions.at(2)});
        //printMatrix(comp);
    //}
}

int main(int argc, char *argv[]) {
    if (argc < 4 || argc > 5) {
        cerr << "Ussage: test n d N [vpos]" << endl;
        return 1;
    }
    int n = atoi(argv[1]);
    int d = atoi(argv[2]);
    int N = atoi(argv[3]);
    assert(d > 0 && n > 0 && N >= 0);
    int vpos;
    if (argc != 5) {
        vpos = d / 2;
    }
    else {
        vpos = atoi(argv[4]);
    }
    assert(vpos >= 0 && vpos < d);
    
    const char *env_TTSVD = getenv("TTSVD");
    const char *env_SIMPLE = getenv("SIMPLE");
    const bool simple = (env_SIMPLE && *env_SIMPLE == '1');
    ofstream nout("/dev/null");
    ostream &sout = simple ? cout : nout;
    ostream &vout = !simple ? cout : nout;
    
    vout << "n = " << n << ", d = " << d << ", N = " << N << endl;
    
    vector<size_t> dims(d, n);
    const auto x = Tensor::random(dims, static_cast<size_t>(N));
    
    vout << "--------------------FLATT--------------------" << endl;
    run_test([vpos](auto &&x) { return tensor2tt_lossless(x, vpos); }, x, sout, vout);

    if (env_TTSVD && *env_TTSVD == '1') {
        vout << "--------------------TTSVD--------------------" << endl;
        auto y(x);
        y.use_dense_representation();
        run_test([](auto &&x) { return TTTensor(x); }, y, sout, vout);
    }
    else {
        sout << 0 << endl;
    }
    
    return 0;
}

