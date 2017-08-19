//
//  main.cpp
//  DigitRecognizer
//
//  Created by Anton Logunov on 6/12/14.
//  Copyright (c) 2014 Anton Logunov. All rights reserved.
//

#include <iostream>
#include <fstream>
#include <string>
#include <cstdlib>
#include <array>

#include <ant>

using namespace std;
using namespace ant;

const string root = "/Users/antoshkaplus/Documents"
                    "/Programming/Contests/Kaggle/DigitRecognizer/Scripts/";

using V_uch = vector<u_char>;
using VV_uch = vector<V_uch>;
using V_d = vector<double>;
using VV_d = vector<V_d>;
using VVV_d = vector<VV_d>;

vector<u_char> split(string str) {
    vector<u_char> res;
    Index i_0 = 0, i_1 = 0;
    while (++i_1 < str.size()) {
        // second condition for last number
        if (str[i_1] == ',') {
            str[i_1] = '\0';
            res.push_back(atoi(str.c_str()+i_0));
            str[i_1] = ',';
            i_0 = i_1+1;
            i_1 = i_0;
        }
    }
    res.push_back(atoi(str.c_str()+i_0));
    return res;
} 


V_uch neighbor(const VV_uch& tr_set, const VV_uch& t_set) {
    V_uch res(t_set.size());
    array<size_t, 10> count;
    for (auto i = 0; i < t_set.size(); ++i) {
        auto vs = ml::find_k_nearest_neighbors(tr_set, t_set[i], 10, 
        [](const vector<u_char>& train, const vector<u_char>& test) {
           double dist = 0;
           for (auto i = 0; i < test.size(); ++i) {
               dist += ((Int)train[i+1] - (Int)test[i])*((Int)train[i+1] - (Int)test[i]);
           }
           return sqrt(dist);
        });

        fill(count.begin(), count.end(), 0);
        for (auto v : vs) {
            assert(tr_set[v][0] >= 0 && tr_set[v][0] < 10);
            ++count[tr_set[v][0]]; 
            
        }
        res[i] = max_element(count.begin(), count.end())-count.begin();
        
        if ((i+1) % 100 == 0) cout << i+1 << " solved" << endl;  
    }
    return res;
}


V_uch probability(const VV_uch& tr_set, const VV_uch& t_set) {
    u_char value = 127;
    Count feature_count = t_set[0].size(); 
    vector<double> class_probs(10, 0.5);
    VVV_d value_probs(10);
    for (auto i = 0; i < 10; ++i) {
        value_probs[i].resize(feature_count);
        for (auto k = 0; k < feature_count; ++k) {
            value_probs[i][k].resize(2, 0);
        }
    }
    for (auto& t : tr_set) {
        Index cl = t[0]; 
        for (auto i = 0; i < feature_count; ++i) {
            Index k = t[i+1] > value ? 1 : 0;
            ++value_probs[cl][i][k]; 
        }
    }
    for (auto i = 0; i < 10; ++i) {
        for (auto k = 0; k < feature_count; ++k) {
            auto& vp = value_probs[i][k]; 
            double total = vp[0] + vp[1];
            vp[0] /= total;
            vp[1] /= total;
        }
    }
    
    auto value_probs_func = [&](Index cl, Index feat, u_char val) {
        return value_probs[cl][feat][val > value];
    };
    
    V_uch res(t_set.size());
    for (auto i = 0; i < t_set.size(); ++i) {
        res[i] = ml::naive_bayes(t_set[i], class_probs, value_probs_func);
    }
    return res;
}


int main(int argc, const char * argv[])
{
//    auto options = command_line_options(argv, argc);
//    ifstream tr_set(options["training_set"]);
//    ifstream t_set(options["test_set"]);
//    ofstream res(options["result"]);

    ifstream tr_set(root + "train.csv");
    ifstream t_set(root + "test.csv");
    ofstream res(root + "res.txt");
    
    
    vector<vector<u_char>> tr_set_mat;
    vector<vector<u_char>> t_set_mat;
    vector<u_char> r_mat;
    
    cout << "read train set" << endl;
    string str;
    bool first = true;
    while (tr_set >> str) {
        if (first) {
            first = false; 
            continue;       
        }
        
        tr_set_mat.push_back(split(str));
        assert(tr_set_mat.back().front() < 10); 
    }
    cout << tr_set_mat.size() << endl;
    
    cout << "read test set" << endl;
    first = true;
    while (t_set >> str) {
        if (first) {
            first = false;
            continue;
        }
        t_set_mat.push_back(split(str));
    }
    cout << t_set_mat.size() << endl;
    
    cout << "solving" << endl;
    
    auto condition = [](const vector<u_char>& raw, Index feature) {
        return raw[raw.size() == 785 ? feature : feature-1];
    };
//    ml::binary_decision_tree<decltype(tr_set_mat), decltype(condition), vector<u_char>> tree;
//    tree.construct(tr_set_mat, condition, 10);
//    for (auto& t : t_set_mat) {
//        r_mat.push_back(tree.category(t));
//    }

    ml::logistic_regression lg;
    ml::Mat<double> train_mat(tr_set_mat.size(), tr_set_mat[0].size()-1);
    for (auto r = 0; r < train_mat.row_count(); ++r) {
        for (auto c = 0; c < train_mat.col_count(); ++c) {
            train_mat(r, c) = tr_set_mat[r][c+1]/256.;
        }
    }
    ml::Mat<Index> cat_mat(tr_set_mat.size(), 1);
    for (auto r = 0; r < cat_mat.row_count(); ++r) {
        cat_mat(r, 0) = tr_set_mat[r][0];
    }
    ml::Mat<double> test_mat(t_set_mat.size(), t_set_mat[0].size());
    for (auto r = 0; r < test_mat.row_count(); ++r) {
        for (auto c = 0; c < test_mat.col_count(); ++c) {
            test_mat(r, c) = t_set_mat[r][c]/256.;
        }
    }
    lg.train(train_mat, cat_mat, 10);
    auto mat = lg.predict(test_mat);

    //r_mat = probability(tr_set_mat, t_set_mat);
    
    res << "ImageId,Label" << endl;
    for (auto i = 0; i < test_mat.row_count(); ++i) {
        //res << i+1 << "," << (Int)r_mat[i] << endl;
        res << i+1 << "," << (Int)mat(i, 0) << endl;
    }
    
    // insert code here...
    std::cout << "Hello, World!\n";
    return 0;
}

