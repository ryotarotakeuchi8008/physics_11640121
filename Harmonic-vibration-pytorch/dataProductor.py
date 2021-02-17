import numpy as np
import sympy as sym
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from sympy.plotting import plot
import pandas as pd
import csv
from scipy.integrate import odeint


# 調和振動

def data_productor():
    sym.init_printing(use_unicode=True)
    t, m, k = sym.symbols("t m k")

    # 関数x=f(t)を定義
    x = sym.Function('x')(t)

    # 単振動 微分方程式 m=5,k=1とする
    eq = sym.Eq(5 * x.diff(t, 2) + 1 * x, 0)

    # 初期条件 t=0 → x=10 x'=0 で微分方程式を解く
    ans = sym.dsolve(eq, ics={x.subs(t, 0): 10, x.diff(t, 1).subs(t, 0): 0})

    # 微分方程式の解expr=x(t)
    expr = ans.rhs

    # グラフ描画 matplotlib
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    gx = []
    gt = []
    ims = []

    # 1~500だと40周期くらいになってしまってる(１周期 t=14)  1~3周期で小さく刻んでデータ数をとる。
    # 1~42(3周期分) 0.02ずつ細かくplot
    for i in [round(i * 0.07, 4) for i in range(1, 401, 1)]:
        # sympyの型からint型にしないとエラる
        x = round(float(expr.subs(t, i)), 5)

        with open('./data/data.csv', 'a') as f:
            writer = csv.writer(f)
            writer.writerow([i, x])

        # グラフ描画
        gt.append(i)
        gx.append(x)

        # ➀これまでのx値・y値を全てプロット → 線の軌跡に
        im_line = ax.plot(gt, gx, 'b')

        # ➁現在のx値・y値を点としてプロット
        im_point = ax.plot(i, x, marker='.', color='b', markersize=10)

        # ③時間もプロット
        im_time = ax.text(0.0, 0.1, 'time = {0:5.2f}'.format(i))

        # ➀~③のaxsオブジェクトをまとめて配列に格納
        ims.append(im_line + im_point + [im_time])

    # pd.read_csv('./data/data.csv', encoding="utf-8")

    ax.set_xlabel('t')
    ax.set_ylabel('x')
    ax.set_aspect('equal', 'datalim')

    anm = animation.ArtistAnimation(fig, ims, interval=50)
    anm.save('animation.gif', writer='pillow')
    plt.show()
