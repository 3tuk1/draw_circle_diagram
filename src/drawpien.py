import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams, font_manager

# フォント設定
font_path = os.path.join('..', 'Noto_Sans_JP', 'static', 'NotoSansJP-Regular.ttf')
if os.path.exists(font_path):
    font_prop = font_manager.FontProperties(fname=font_path)
    rcParams['font.family'] = font_prop.get_name()
else:
    # バックアップとして日本語対応フォントを設定
    plt.rcParams['font.family'] = ['sans-serif']
    plt.rcParams['font.sans-serif'] = ['IPAexGothic', 'Yu Gothic', 'Meiryo', 'Takao']


def draw_circle_diagram(V, I1, Is1, I2, Is2, R1, R2):
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_aspect('equal')

    # 基本ベクトルと点の計算
    O = np.array([0, 0])
    Np = np.array([0, I1])  # N'
    Sp = np.array([0, Is1])  # S'
    N = Np + np.array([I2, 0])  # N
    S = Sp + np.array([Is2, 0])  # S

    # 垂線の計算関数
    def perp_drop(p, a1, a2):
        a_vec = a2 - a1
        t = np.dot(p - a1, a_vec) / np.dot(a_vec, a_vec)
        return a1 + t * a_vec

    # 直線の交点計算関数
    def intersect(p1, d1, p2, d2):
        A = np.array([d1, -d2]).T
        try:
            return p1 + np.linalg.solve(A, p2 - p1)[0] * d1
        except np.linalg.LinAlgError:
            # 平行線の場合
            return None

    # (4) SよりN'Nの延長線上に垂線SUをおろす
    U = perp_drop(S, Np, N)

    # NSを結び（出力線）
    mid_NS = (N + S) * 0.5
    vec_NS = S - N
    perp = np.array([-vec_NS[1], vec_NS[0]])  # NSに垂直なベクトル

    # 修正後
    # Cの高さはN'と同じにする（y座標がI1）
    if perp[1] != 0:
        # 垂直二等分線上でy=I1となる点を計算
        t = (I1 - mid_NS[1]) / perp[1]
        C = mid_NS + t * perp
    else:
        # 垂直二等分線が水平の場合（稀なケース）
        C = np.array([mid_NS[0], I1])


    r = np.linalg.norm(C - N)  # 半径
    # 上半分の円弧のみ描画
    theta_arc = np.linspace(0, np.pi, 300)
    x_arc = C[0] + r * np.cos(theta_arc)
    y_arc = C[1] + r * np.sin(theta_arc)
    ax.plot(x_arc, y_arc, 'k', linewidth=2)

    # (6) SUをTで分割（UT:TS = R1:R2）
    alpha = R1 / (R1 + R2) if (R1 + R2) != 0 else 0
    T = U + alpha * (S - U)

    # (7) SNを左下方向に延長して横軸との交点をDとする
    NS_vec = N - S
    if NS_vec[1] != 0:  # 垂直でない場合
        t = -S[1] / NS_vec[1]
        D = S + t * NS_vec
    else:
        D = np.array([S[0], 0])  # 垂直の場合、横軸上の同じx座標

    # 垂線DF'を立てる
    DF_vec = np.array([0, 1])

    # 垂線NG'を立てる
    NG_vec = np.array([0, 1])

    # (8) DF'とSS'の交点をFとする
    F = intersect(D, DF_vec, Sp, S - Sp)

    # (9) Sを通り、NTと平行な直線を引く
    NT_vec = T - N
    NT_dir = NT_vec / np.linalg.norm(NT_vec)

    # NG'との交点をGとする
    G = intersect(S, NT_dir, N, NG_vec)

    # (10) 円の中心CよりNSおよびNTに垂線をおろす
    # NSに垂線をおろす
    NS_dir = vec_NS / np.linalg.norm(vec_NS)
    NS_perp = np.array([-NS_dir[1], NS_dir[0]])

    # 円周との交点Pmを求める
    Pm_vec = C + r * NS_perp
    Pm = perp_drop(Pm_vec, C, C + NS_perp)

    # NTに垂線をおろす
    NT_perp = np.array([-NT_dir[1], NT_dir[0]])

    # 円周との交点PTを求める
    PT_vec = C + r * NT_perp
    PT = perp_drop(PT_vec, C, C + NT_perp)

    # 垂線の基点Qm, QTを求める
    Qm = perp_drop(C, N, S)
    QT = perp_drop(C, N, T)

    # 線の描画
    # 出力線 NS
    ax.plot([N[0], S[0]], [N[1], S[1]], 'b-', label='出力線')

    # 回転力線 NT
    ax.plot([N[0], T[0]], [N[1], T[1]], 'g-', label='回転力線')

    # 効率線 FS
    ax.plot([F[0], S[0]], [F[1], S[1]], 'r-', label='効率線')

    # すべり線 SG
    ax.plot([S[0], G[0]], [S[1], G[1]], 'm-', label='すべり線')

    # 垂線と補助線の描画
    ax.plot([S[0], U[0]], [S[1], U[1]], 'k:', linewidth=1)  # 垂線SU
    ax.plot([D[0], F[0]], [D[1], F[1]], 'k:', linewidth=1)  # 垂線DF
    ax.plot([N[0], G[0]], [N[1], G[1]], 'k:', linewidth=1)  # 垂線NG
    ax.plot([C[0], Pm[0]], [C[1], Pm[1]], 'k:', linewidth=1)  # 垂線CPm
    ax.plot([C[0], PT[0]], [C[1], PT[1]], 'k:', linewidth=1)  # 垂線CPT
    ax.plot([Pm[0], Qm[0]], [Pm[1], Qm[1]], 'k:', linewidth=1)  # 垂線PmQm
    ax.plot([PT[0], QT[0]], [PT[1], QT[1]], 'k:', linewidth=1)  # 垂線PTQT

    # 水平補助線
    ax.plot([0, N[0]], [Np[1], N[1]], 'k:', linewidth=1)  # N'N
    ax.plot([0, S[0]], [Sp[1], S[1]], 'k:', linewidth=1)  # S'S

    # 点の描画と名前のラベル付け
    points = {
        'O': O, 'N\'': Np, 'S\'': Sp, 'N': N, 'S': S, 'U': U, 'C': C, 'T': T,
        'D': D, 'F': F, 'G': G, 'Pm': Pm, 'PT': PT, 'Qm': Qm, 'QT': QT
    }

    for lbl, pt in points.items():
        ax.plot(pt[0], pt[1], 'ko', markersize=4)
        offset_x = -8 if lbl in ['O', 'N\'', 'S\''] else 10
        offset_y = 10
        ha = 'right' if lbl in ['O', 'N\'', 'S\''] else 'left'
        ax.text(pt[0] + offset_x / 100, pt[1] + offset_y / 100, lbl, fontsize=9, ha=ha)

    # グラフの設定
    ax.set_title('L形円線図', fontsize=14)
    ax.grid(True)
    ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax.axvline(x=0, color='k', linestyle='-', alpha=0.3)
    ax.legend(loc='upper right')

    # マージンを広めに確保して全ての点が見えるようにする
    # 修正後:
    all_pts = np.array(list(points.values()))
    # 円の右端と上端も考慮
    max_x = max(max(all_pts[:, 0]), C[0] + r)
    max_y = max(max(all_pts[:, 1]), C[1] + r)

    # マージンを大きめに設定
    margin_x = max_x * 0.15
    margin_y = max_y * 0.15
    ax.set_xlim(0, max_x + margin_x)
    ax.set_ylim(0, max_y + margin_y)

    plt.tight_layout()
    plt.show()

    # 計算結果の表示
    print(f"入力パラメータ: V={V}, I1={I1}, Is1={Is1}, I2={I2}, Is2={Is2}, R1={R1}, R2={R2}")
    print(f"円の中心C: ({C[0]:.2f}, {C[1]:.2f})")
    print(f"円の半径: {r:.2f}")


# メイン実行部
if __name__ == '__main__':
    try:
        print("円線図作成のためのパラメータを入力してください:")
        V_val = float(input("V [V]: "))
        I1_val = float(input("I1 [A]: "))
        Is1_val = float(input("Is1 [A]: "))
        I2_val = float(input("I2 [A]: "))
        Is2_val = float(input("Is2 [A]: "))
        R1_val = float(input("R1 [Ω]: "))
        R2_val = float(input("R2 [Ω]: "))

        draw_circle_diagram(V_val, I1_val, Is1_val, I2_val, Is2_val, R1_val, R2_val)
    except ValueError:
        print("数値を入力してください。")
        print("テスト値を使用します: V=200, I1=0.291, Is1=2.77, I2=3.48, Is2=5.7, R1=0.7, R2=1.5")
        draw_circle_diagram(200, 0.291, 20.58, 3.48, 42.32, 0.7, 1.5)