import csv
from datetime import datetime
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


def draw_circle_diagram(V, I1, Is1, I2, Is2, R1, R2, P_1):
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

    # Cの高さはN'と同じにする（y座標がI1）
    if perp[1] != 0:
        t = (I1 - mid_NS[1]) / perp[1]
        C = mid_NS + t * perp
    else:
        C = np.array([mid_NS[0], I1])

    r = np.linalg.norm(C - N)  # 半径
    # 円弧描画用のデータ
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
    NS_dir = vec_NS / np.linalg.norm(vec_NS)
    NS_perp = np.array([-NS_dir[1], NS_dir[0]])
    Pm_vec = C + r * NS_perp
    Pm = perp_drop(Pm_vec, C, C + NS_perp)
    NT_perp = np.array([-NT_dir[1], NT_dir[0]])
    PT_vec = C + r * NT_perp
    PT = perp_drop(PT_vec, C, C + NT_perp)
    Qm = perp_drop(C, N, S)
    QT = perp_drop(C, N, T)

    # 追加要素の実装
    # (1) DFじょうにDH＝(P_1/3)/(V/√3)になるように点Hを定める
    DF_len = np.linalg.norm(F - D)
    DH_len = (P_1 / 3) / (V / np.sqrt(3))
    DF_dir = (F - D) / DF_len if DF_len != 0 else np.array([0, 1])
    H = D + DH_len * DF_dir

    # (2) HからNSに平行線を引き、円弧との交点をPとする
    # 円と直線の交点を計算
    H_dir = NS_dir  # NSと平行な方向ベクトル
    # 円と直線の交点を解析的に求める
    # 直線: H + t * H_dir
    # 円: |X - C|^2 = r^2
    a = np.dot(H_dir, H_dir)
    b = 2 * np.dot(H - C, H_dir)
    c = np.dot(H - C, H - C) - r ** 2
    discriminant = b ** 2 - 4 * a * c

    if discriminant >= 0:
        t1 = (-b + np.sqrt(discriminant)) / (2 * a)
        t2 = (-b - np.sqrt(discriminant)) / (2 * a)
        P1 = H + t1 * H_dir
        P2 = H + t2 * H_dir
        # xの値が小さい方をPとする
        P = P1 if P1[0] < P2[0] else P2
    else:
        # 交点がない場合のフォールバック
        P = H + H_dir * r

    # (3) Pから水平線を引いてx=0にある縦軸との交点をP'とする
    Pp = np.array([0, P[1]])  # P'

    # (4) PをO、D、Nに結び、DPの延長とFSの交点をYとし、NPの延長とGSの交点をRとする
    DP_dir = (P - D) / np.linalg.norm(P - D)
    NP_dir = (P - N) / np.linalg.norm(P - N)
    FS_dir = (S - F) / np.linalg.norm(S - F)
    GS_dir = (S - G) / np.linalg.norm(S - G)

    Y = intersect(D, DP_dir, F, FS_dir)
    R = intersect(N, NP_dir, G, GS_dir)

    # (5) Pから垂線をおろし、出力線NS、回転力線NT、NUおよび横軸との交点をそれぞれA、B、LおよびMとする
    # 垂線の方向ベクトル（下向き）
    perp_down = np.array([0, -1])

    # NS（出力線）との交点A
    NS_line = S - N
    A = intersect(P, perp_down, N, NS_line)

    # NT（回転力線）との交点B
    NT_line = T - N
    B = intersect(P, perp_down, N, NT_line)

    # NU線との交点L
    NU_line = U - N
    L = intersect(P, perp_down, N, NU_line)

    # 横軸との交点M
    M = np.array([P[0], 0])

    # 点の描画と線の描画（既存のコードは省略）
    # 新しい点と線の描画
    points = {
        'O': O, 'N\'': Np, 'S\'': Sp, 'N': N, 'S': S, 'U': U, 'C': C, 'T': T,
        'D': D, 'F': F, 'G': G, 'Pm': Pm, 'PT': PT, 'Qm': Qm, 'QT': QT,
        'H': H, 'P': P, 'P\'': Pp, 'Y': Y, 'R': R, 'A': A, 'B': B, 'L': L, 'M': M
    }

    # 線の描画
    ax.plot([N[0], S[0]], [N[1], S[1]], 'b-', label='出力線')
    ax.plot([N[0], T[0]], [N[1], T[1]], 'g-', label='回転力線')
    ax.plot([F[0], S[0]], [F[1], S[1]], 'r-', label='効率線')
    ax.plot([S[0], G[0]], [S[1], G[1]], 'm-', label='すべり線')

    # 追加する線
    ax.plot([D[0], H[0]], [D[1], H[1]], 'k:', linewidth=1)  # DH
    ax.plot([H[0], P[0]], [H[1], P[1]], 'k:', linewidth=1)  # HP
    ax.plot([P[0], Pp[0]], [P[1], Pp[1]], 'k:', linewidth=1)  # PP'
    ax.plot([O[0], P[0]], [O[1], P[1]], 'k-', linewidth=1)  # OP
    ax.plot([D[0], P[0]], [D[1], P[1]], 'k-', linewidth=1)  # DP
    ax.plot([D[0], Y[0]], [D[1], Y[1]], 'k:', linewidth=1)  # DY (DP延長)
    ax.plot([N[0], P[0]], [N[1], P[1]], 'k-', linewidth=1)  # NP
    ax.plot([N[0], R[0]], [N[1], R[1]], 'k:', linewidth=1)  # NR (NP延長)
    ax.plot([P[0], A[0]], [P[1], A[1]], 'k:', linewidth=1)  # PA
    ax.plot([P[0], B[0]], [P[1], B[1]], 'k:', linewidth=1)  # PB
    ax.plot([P[0], L[0]], [P[1], L[1]], 'k:', linewidth=1)  # PL
    ax.plot([P[0], M[0]], [P[1], M[1]], 'k:', linewidth=1)  # PM

    # その他の既存の線（省略）...
    ax.plot([S[0], U[0]], [S[1], U[1]], 'k:', linewidth=1)
    ax.plot([D[0], F[0]], [D[1], F[1]], 'k:', linewidth=1)
    ax.plot([N[0], G[0]], [N[1], G[1]], 'k:', linewidth=1)
    ax.plot([0, N[0]], [Np[1], N[1]], 'k:', linewidth=1)
    ax.plot([0, S[0]], [Sp[1], S[1]], 'k:', linewidth=1)

    # 点の描画
    for lbl, pt in points.items():
        ax.plot(pt[0], pt[1], 'ko', markersize=4)
        offset_x = -8 if lbl in ['O', 'N\'', 'S\'', 'P\''] else 10
        offset_y = 10
        ha = 'right' if lbl in ['O', 'N\'', 'S\'', 'P\''] else 'left'
        ax.text(pt[0] + offset_x / 100, pt[1] + offset_y / 100, lbl, fontsize=9, ha=ha)

    # グラフ設定
    ax.set_title('L形円線図', fontsize=14)
    ax.grid(True)
    ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax.axvline(x=0, color='k', linestyle='-', alpha=0.3)
    ax.legend(loc='upper right')

    # マージン設定
    all_pts = np.array(list(points.values()))
    max_x = max(max(all_pts[:, 0]), C[0] + r)
    max_y = max(max(all_pts[:, 1]), C[1] + r)
    margin_x = max_x * 0.15
    margin_y = max_y * 0.15
    ax.set_xlim(0, max_x + margin_x)
    ax.set_ylim(0, max_y + margin_y)


    # 計算結果の表示
    print(f"入力パラメータ: V={V}, I1={I1}, Is1={Is1}, I2={I2}, Is2={Is2}, R1={R1}, R2={R2}, P_1={P_1}")
    print(f"円の中心C: ({C[0]:.2f}, {C[1]:.2f})")
    print(f"円の半径: {r:.2f}")

    # 図の表示
    plt.tight_layout()
    output_dir = os.path.join('..', 'output')
    # 出力ディレクトリの作成
    os.makedirs(output_dir, exist_ok=True)

    # ファイル名の生成（タイムスタンプ付き）
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"circle_diagram_V{V}_P{P_1}_{timestamp}.png"
    filepath = os.path.join(output_dir, filename)

    # PNG形式で保存
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.show()

    # 指定した線分の長さを計算してCSVファイルに出力
    def calculate_distance(point1, point2):
        return np.linalg.norm(np.array(point1) - np.array(point2))

    # 計測する線分のリスト
    segments = [
        ('OP', 'O', 'P'),
        ('OP\'', 'O', 'P\''),
        ('FY', 'F', 'Y'),
        ('FS', 'F', 'S'),
        ('GR', 'G', 'R'),
        ('GS', 'G', 'S'),
        ('PB', 'P', 'B'),
        ('PA', 'P', 'A'),
        ('AB', 'A', 'B'),
        ('BL', 'B', 'L'),
        ('LM', 'L', 'M')
    ]

    # 線分の長さを計算
    lengths = []
    for label, start, end in segments:
        if start in points and end in points:
            length = calculate_distance(points[start], points[end])
            lengths.append((label, length))
        else:
            print(f"警告: 点 {start} または {end} が見つかりません")

    # CSVファイルに書き込み
    csv_filename = f"segment_lengths_V{V}_P{P_1}_{timestamp}.csv"
    csv_filepath = os.path.join(output_dir, csv_filename)

    with open(csv_filepath, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['線分', '長さ'])
        for label, length in lengths:
            writer.writerow([label, f"{length:.4f}"])

    print(f"線分の長さをCSVファイルに保存しました: {csv_filepath}")

    # 計算結果の表示
    print(f"入力パラメータ: V={V}, I1={I1}, Is1={Is1}, I2={I2}, Is2={Is2}, R1={R1}, R2={R2}, P_1={P_1}")
    print(f"円の中心C: ({C[0]:.2f}, {C[1]:.2f})")
    print(f"円の半径: {r:.2f}")
    print(f"円線図をPNG形式で保存しました: {filepath}")


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
        P1_val = float(input("P_1 [W]: "))

        draw_circle_diagram(V_val, I1_val, Is1_val, I2_val, Is2_val, R1_val, R2_val, P1_val)
    except ValueError:
        print("数値を入力してください。")
        print("テスト値を使用します: V=200, I1=0.291, Is1=2.77, I2=3.48, Is2=5.7, R1=0.7, R2=1.5, P_1=2200")
        draw_circle_diagram(200, 0.291, 20.58, 3.48, 42.32, 0.7, 1.5, 2200)