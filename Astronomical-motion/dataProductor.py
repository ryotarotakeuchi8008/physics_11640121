import numpy as np
import sympy as sym
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from sympy.plotting import plot
import pandas as pd
import csv
from scipy.integrate import odeint
import scipy


def data_productor():
    # 惑星の運動方程式をシミュレーション
    # 惑星の2階微分運動方程式(無次元化済み)をrの1階微分方程式・vの1階微分方程式の連立微分方程式に分けてそれぞれにルンゲ=クッタ法を使う
    # numpy配列の各要素の桁数を一括設定
    np.set_printoptions(precision=3)

    # 初期条件 r(0),v(0) rはrベクトル={x,y} 無次元化した惑星の2階微分運動方程式が基準(地球を1とした時の火星の相対長半径=1.52・相対公転速度=0.81)
    r = np.array([1.52, 0.0])
    v = np.array([0.0, 0.81])

    t = 0.0

    # ルンゲ=クッタ法の幅dtと漸化式の繰り返し数
    dt = 0.01
    nt = 2301

    # アニメーションは漸化式の繰り返し20回ごとに記録
    nout = 20

    # x値を保存するlist
    gx = []

    # y値を保存するlist
    gy = []

    ims = []

    # グラフ描画 matplotlib
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    # ↓ r(t)とv(t)の各tによる勾配f(tn,r),f(tn,v)
    def fr(v):
        return v

    def fv(r):
        rnorm = np.linalg.norm(r)
        return -r / rnorm ** 3

    def round_int(x):
        return round(x, 3)

    # dtの幅でnt回だけ漸化式回す
    for i in range(nt):
        # 初期条件のr(0)とv(0)
        rp = r
        vp = v

        # r(t)とv(t)の各tによる勾配f(tn,r),f(tn,v)からt=0時点のk1を求める
        kr1 = fr(vp)
        kv1 = fv(rp)

        # k1と初期条件値からk2を求める
        rp = r + 0.5 * dt * kr1
        vp = v + 0.5 * dt * kv1
        kr2 = fr(vp)
        kv2 = fv(rp)

        # 同じくk3
        rp = r + 0.5 * dt * kr2
        vp = v + 0.5 * dt * kv2
        kr3 = fr(vp)
        kv3 = fv(rp)

        # k4も求める
        rp = r + dt * kr3
        vp = v + dt * kv3
        kr4 = fr(vp)
        kv4 = fv(rp)

        # k1~k4の加重平均を傾きとし、r(tn+dt)=r(tn)+傾き*dt
        r += dt * (kr1 + 2.0 * kr2 + 2.0 * kr3 + kr4) / 6.0
        v += dt * (kv1 + 2.0 * kv2 + 2.0 * kv3 + kv4) / 6.0
        t += dt

        sample1_list = list(range(5))

        # モデルへの入力t と 出力(==正解ラベル)であるr(t)ベクトル・v(t)ベクトルをcsvファイルへ出力
        with open('./data/data.csv', 'a') as f:
            writer = csv.writer(f)

            # 各要素を丸める(小数点)
            t = round_int(t)
            x = round_int(r[0])
            y = round_int(r[1])
            v_x = round_int(v[0])
            v_y = round_int(v[1])
            writer.writerow([t, x, y, v_x, v_y])
        import pdb;

        # グラフ描画は漸化式20回に1回
        if i % nout == 0:
            print(
                'i: {0:4d}, t: {1:6.2f}, x: {2:9.6f}, y: {3:9.6f} , vx: {4:9.6f}, vy: {5:9.6f}'.format(i, t, r[0], r[1],
                                                                                                       v[0], v[1]))
            gx.append(r[0])
            gy.append(r[1])

            # ➀これまでのx値・y値を全てプロット → 線の軌跡に
            im_line = ax.plot(gx, gy, 'b')

            # ➁現在のx値・y値を点としてプロット
            im_point = ax.plot(r[0], r[1], marker='.', color='b', markersize=10)

            # ③時間もプロット
            im_time = ax.text(0.0, 0.1, 'time = {0:5.2f}'.format(t))

            # ➀~③のaxsオブジェクトをまとめて配列に格納
            ims.append(im_line + im_point + [im_time])

    # 漸化式をnt回繰り返し終わったらグラフ描画

    # 太陽の位置を座標指定してプロット
    ax.plot(0.0, 0.0, marker='.', color='r', markersize=10)

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_aspect('equal', 'datalim')

    # 発表の時以外は使わない
    # figureオブジェクトとaxsオブジェクトの配列を渡してアニメーション
    anm = animation.ArtistAnimation(fig, ims, interval=50)
    anm.save('animation.gif', writer='pillow')
    plt.show()
