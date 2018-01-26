#include <iostream>
#include <chrono>
#include <cassert>
#include "tensor2tt_lossless.h"

using namespace std;
using namespace xerus;

int main(int argc, char *argv[]) {
    if (argc < 4 || argc > 5) {
        cout << "Ussage: test n d N [vpos]" << endl;
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
    
    vector<size_t> dims(d, n);
    auto x = Tensor::random(dims, static_cast<size_t>(N));
    auto begin = chrono::high_resolution_clock::now();
    auto tt = tensor2tt_lossless(x, vpos);
    auto end = chrono::high_resolution_clock::now();
    
    cout << "n = " << n << ", d = " << d << ", N = " << N << endl;
    
    cout << "time: " << chrono::duration_cast<chrono::milliseconds>(end - begin).count() << "ms"  << endl;
    
    auto xx = Tensor(tt);
    auto eps = (x - xx).frob_norm() / x.frob_norm();
    cout << "eps: " << setprecision(10) << eps << endl;
    
    cout << "ranks: ";
    for (auto r : tt.ranks()) {
        cout << r << " ";
    }
    cout << endl;
    return 0;
}

