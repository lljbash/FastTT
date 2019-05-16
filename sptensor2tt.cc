#include "sptensor2tt.h"
#include <vector>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <memory>
#include <limits>
#include "xerus/misc/check.h"
#include "xerus/misc/internal.h"

using namespace std;

namespace xerus {

int count_subvector(Tensor &a, int vpos) {
    const size_t d = a.degree();
    TTTensor u(d);
    const auto &data = a.get_sparse_data();
    unordered_set<size_t> submats;
    const auto &n = a.dimensions;
    size_t pn = 1;
    for (size_t i = vpos + 1; i < d; ++i) {
        pn *= n.at(i);
    }
    for (const auto &[index, value] : data) {
        const size_t in_pos = index / pn % n.at(vpos);
        const size_t out_pos = index - in_pos * pn;
        submats.insert(out_pos);
    }
    return submats.size();
}

TTTensor extract_subvector(Tensor &a, int vpos) {
    const size_t d = a.degree();
    TTTensor u(d);
    const auto &data = a.get_sparse_data();
    unordered_map<size_t, vector<pair<size_t, value_t>>> submats;
    const auto &n = a.dimensions;
    size_t pn = 1;
    for (size_t i = vpos + 1; i < d; ++i) {
        pn *= n.at(i);
    }
    for (const auto &[index, value] : data) {
        const size_t in_pos = index / pn % n.at(vpos);
        const size_t out_pos = index - in_pos * pn;
        submats[out_pos].emplace_back(in_pos, value);
    }
    
    const size_t r = submats.size();
    u.set_component(0, Tensor({1, n.front(), r}));
    for (size_t i = 1; i < d - 1; ++i) {
        u.set_component(i, Tensor({r, n.at(i), r}));
    }
    u.set_component(d - 1, Tensor({r, n.back(), 1}));
    
    size_t index = 0;
    for (const auto &submat : submats) {
        const size_t out_pos = submat.first;
        const auto md = Tensor::position_to_multiIndex(out_pos, n);
        for (size_t i = 0; i < d; ++i) {
            size_t nleft = i == 0 ? 0 : index;
            size_t nright = i == d - 1 ? 0 : index;
            if (i != static_cast<size_t>(vpos)) {
                u.component(i).operator[]({nleft, md.at(i), nright}) = 1;
            }
            else {
                for (const auto &[index, value] : submat.second) {
                    u.component(i).operator[]({nleft, index, nright}) = value;
                }
            }
        }
        ++index;
    }

    return u;
}

auto depar01(Tensor a) {
    const size_t d = a.degree();
    REQUIRE(d == 2, "Input of depar01 must be a matrix");
    const size_t nrows = a.dimensions.at(0);
    const size_t ncols = a.dimensions.at(1);
    
    vector a_col_nze(ncols, numeric_limits<size_t>::max());
    a.use_sparse_representation();
    const auto &data = a.get_sparse_data();
    for (const auto &[index, value] : data) {
        auto indices = Tensor::position_to_multiIndex(index, a.dimensions);
        size_t index_row = indices.at(0);
        size_t index_col = indices.at(1);
        a_col_nze.at(index_col) = index_row;
    }
    
    vector hashmap(nrows, numeric_limits<size_t>::max());
    vector<vector<size_t>> b_indices;
    vector<vector<size_t>> t_indices;
    size_t cnt = 0;
    for (size_t i = 0; i < ncols; ++i) {
        const size_t &index_row = a_col_nze.at(i);
        if (index_row >= nrows) {
            continue;
        }
        size_t &newid = hashmap.at(index_row);
        if (newid >= ncols) {
            b_indices.push_back({index_row, cnt});
            newid = cnt++;
        }
        t_indices.push_back({newid, i});
    }
    
    vector b_size{nrows, cnt};
    vector t_size{cnt, ncols};
    map<size_t, value_t> b_data;
    map<size_t, value_t> t_data;
    for (const auto &entry : b_indices) {
        size_t pos = Tensor::multiIndex_to_position(entry, b_size);
        b_data.try_emplace(pos, 1);
    }
    for (const auto &entry : t_indices) {
        size_t pos = Tensor::multiIndex_to_position(entry, t_size);
        t_data.try_emplace(pos, 1);
    }
    
    auto b = Tensor(b_size, Tensor::Representation::Sparse, Tensor::Initialisation::None);
    b.get_unsanitized_sparse_data() = move(b_data);
    auto t = Tensor(t_size, Tensor::Representation::Sparse, Tensor::Initialisation::None);
    t.get_unsanitized_sparse_data() = move(t_data);
    
    return tuple(move(b), move(t));
}

void parrounding(TTTensor &a, size_t vpos) {
    const size_t d = a.degree();
    for (size_t i = 0; i < vpos; ++i) {
        auto &comp = a.component(i);
        const auto n = comp.dimensions;
        comp.reinterpret_dimensions({n.at(0) * n.at(1), n.at(2)});
        auto [b, t] = depar01(move(comp));
        b.reinterpret_dimensions({n.at(0), n.at(1), b.dimensions.back()});
        a.set_component(i, move(b));
        auto &succ = a.component(i+1);
        contract(t, t, succ, 1);
        a.set_component(i+1, move(t));
    }
    for (size_t i = d-1; i > vpos; --i) {
        auto &comp = a.component(i);
        const auto n = comp.dimensions;
        comp.reinterpret_dimensions({n.at(0), n.at(1) * n.at(2)});
        reshuffle(comp, comp, {1, 0});
        auto [b, t] = depar01(move(comp));
        reshuffle(b, b, {1, 0});
        b.reinterpret_dimensions({b.dimensions.front(), n.at(1), n.at(2)});
        a.set_component(i, move(b));
        auto &pre = a.component(i-1);
        contract(t, pre, false, t, true, 1);
        a.set_component(i-1, move(t));
    }
}

int64_t estimate_ttrounding_flops(/*int nnv*/vector<int> current_ranks, vector<size_t> dims, int vpos, int max_rank) {
    if (max_rank < 0) {
        max_rank = numeric_limits<int>::max();
    }
    const int d = dims.size();
    //vector current_ranks(d+1, nnv);
    //current_ranks[0] = 1;
    //current_ranks[d] = 1;
    vector final_ranks(current_ranks);
    for (int i = 0; i < d; ++i) {
        if (max_rank < final_ranks[i+1]) {
            final_ranks[i+1] = max_rank;
        }
    }
    for (auto [i, prod] = pair{0, 1LL}; i <= vpos-1; ++i) {
        prod *= dims[i];
        if (prod < current_ranks[i+1]) {
            current_ranks[i+1] = prod;
        }
    }
    for (auto [i, prod] = pair{d-2, 1LL}; i >= vpos; --i) {
        prod *= dims[i+1];
        if (prod < current_ranks[i+1]) {
            current_ranks[i+1] = prod;
        }
    }
    for (auto [i, prod] = pair{0, 1LL}; i < d-1; ++i) {
        prod *= dims[i];
        if (prod < final_ranks[i+1]) {
            final_ranks[i+1] = prod;
        }
    }
    for (auto [i, prod] = pair{d-2, 1LL}; i >= 0; --i) {
        prod *= dims[i+1];
        if (prod < final_ranks[i+1]) {
            final_ranks[i+1] = prod;
        }
    }
    for (auto r : current_ranks) {
        cout << r << " ";
    }
    cout << endl;
    for (auto r : final_ranks) {
        cout << r << " ";
    }
    cout << endl;
    auto svd_flops = [](int n, int m) {
        return static_cast<int64_t>(n) * static_cast<int64_t>(m)
            * static_cast<int64_t>(min(n, m));
    };
    int64_t flops = 0;
    if (vpos != d - 1) {
        flops += svd_flops(current_ranks[vpos]*dims[vpos], current_ranks[vpos+1]);
    }
    for (int i = vpos+1; i < d-1; ++i) {
        flops += svd_flops(final_ranks[i]*dims[i], current_ranks[i+1]);
    }
    for (int i = d-1; i > vpos; --i) {
        flops += svd_flops(final_ranks[i], dims[i]*final_ranks[i+1]) / 5; // QR flops
    }
    for (int i = vpos; i > 0; --i) {
        flops += svd_flops(final_ranks[i+1]*dims[i], current_ranks[i]);
    }
    return flops;
}

void ttrounding(TTTensor &a, size_t vpos, int max_rank, double eps) {
    constexpr double kAlpha = 1.5;
    static_assert(kAlpha >= 1);
    if (max_rank <= 0) {
        max_rank = 0;
    }
    const size_t d = a.degree();
    int rnsvd = d - vpos - 1;
    int lnsvd = vpos;
    double reps = eps * sqrt(rnsvd) / (sqrt(lnsvd) + sqrt(rnsvd));
    double leps = eps - reps;
    auto norm_a = a.component(vpos).frob_norm();
    //cout << lnsvd << " " << rnsvd << " " << leps << " " << reps << endl;
    for (size_t i = vpos; i < d-1; ++i) {
        Tensor U, S, Vt;
        a.component(i).use_dense_representation();
        double delta = rnsvd == 1 && lnsvd == 0 ? reps : min(leps + reps, reps * kAlpha / sqrt(rnsvd));
        delta = min(max(delta, 0.), 1. - numeric_limits<value_t>::epsilon());
        //cout << delta << endl;
        double svdeps = calculate_svd(U, S, Vt, a.component(i), 2, max_rank, max_rank == 0 ? delta : 0) / norm_a;
        //cout << svdeps << endl;
        --rnsvd;
        reps = reps * reps - svdeps * svdeps;
        reps = reps >= 0 ? sqrt(reps) : -sqrt(-reps);
        a.set_component(i, move(U));
        auto lhs = contract(S, Vt, 1);
        a.set_component(i+1, contract(lhs, a.component(i+1), 1));
        //cout << lnsvd << " " << rnsvd << " " << leps << " " << reps<< endl;
    }
    for (size_t i = d-1; i > vpos; --i) {
        Tensor Q, R;
        a.component(i).use_dense_representation();
        calculate_rq(R, Q, a.component(i), 1);
        a.set_component(i, move(Q));
        a.set_component(i-1, contract(a.component(i-1), R, 1));
    }
    leps += reps;
    reps = 0;
    //cout << lnsvd << " " << rnsvd << " " << leps << " " << reps << endl;
    for (size_t i = vpos; i > 0; --i) {
        Tensor U, S, Vt;
        a.component(i).use_dense_representation();
        double delta = rnsvd == 0 && lnsvd == 1 ? leps : min(leps + reps, leps * kAlpha / sqrt(lnsvd));
        delta = min(max(delta, 0.), 1. - numeric_limits<value_t>::epsilon());
        //cout << delta << endl;
        double svdeps = calculate_svd(U, S, Vt, a.component(i), 1, max_rank, max_rank == 0 ? delta : 0) / norm_a;
        //cout << svdeps << endl;
        --lnsvd;
        leps = leps * leps - svdeps * svdeps;
        leps = leps >= 0 ? sqrt(leps) : -sqrt(-leps);
        a.set_component(i, move(Vt));
        auto rhs = contract(U, S, 1);
        a.set_component(i-1, contract(a.component(i-1), rhs, 1));
        //cout << lnsvd << " " << rnsvd << " " << leps << " " << reps << endl;
    }
}

TTTensor sptensor2tt(Tensor a, int vpos, int max_rank, double eps) {
    a.use_sparse_representation();

    if (int d = a.degree(); vpos < 0 || vpos >= d) {
        vector temp_rank(a.degree()+1, 1);
        for (int d : vector{0, a.degree()-1}) {
            auto u = extract_subvector(a, d);
            parrounding(u, d);
            for (size_t dd = 1; dd < a.degree(); ++dd) {
                if (static_cast<int>(u.rank(dd-1)) > temp_rank[dd]) {
                    temp_rank[dd] = u.rank(dd-1);
                }
            }
        }
        for (auto r : temp_rank) {
            cout << r << " ";
        }
        cout << endl;
        auto min_flops = numeric_limits<int64_t>::max();
        for (int i = 0; i < d; ++i) {
            //int nnv = count_subvector(a, i);
            auto flops = estimate_ttrounding_flops(temp_rank, a.dimensions, i, max_rank);
            cout << flops << endl;
            if (flops < min_flops) {
                min_flops = flops;
                vpos = i;
            }
        }
        cout << vpos << endl;
    }

    auto u = extract_subvector(a, vpos);
    parrounding(u, vpos);
    ttrounding(u, vpos, max_rank, eps);
    
    return u;
}

}

