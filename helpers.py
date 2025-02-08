# helpers.py

import cv2
import numpy as np
import math
from colormath.color_objects import sRGBColor, LabColor
from colormath.color_conversions import convert_color
from skimage.color import deltaE_ciede2000
from skimage import color as sk_color
from sklearn.cluster import KMeans
from config import COLOR_MAP

def hex_to_lab(hex_color):
    """
    将 HEX 颜色转换为 LAB 颜色空间，并返回 (L, a, b) 数值元组，而不是 LabColor 对象。
    """
    rgb = sRGBColor.new_from_rgb_hex(hex_color)
    lab = convert_color(rgb, LabColor)
    return np.array([lab.lab_l, lab.lab_a, lab.lab_b])  # 返回 NumPy 数组

def hex_to_rgb(hex_color):
    """
    将十六进制颜色转换为 RGB 元组，值范围 0-255
    """
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

def rgb_to_lab(rgb):
    """
    将 RGB 颜色（0-255）转换为 Lab 数组
    """
    rgb_norm = np.array(rgb) / 255.0  # 归一化到 [0,1]
    # skimage 的 rgb2lab 需要输入形状为 (M, N, 3)，这里构造 (1, 1, 3)
    lab = sk_color.rgb2lab(np.array([[rgb_norm]]))
    return lab[0, 0]


def velocity_to_dynamic(velocity):
    """
    简单地把 [0~127] 的力度数值映射成 LilyPond 动态标记
    """
    if velocity >= 110:
        return "\\f"    # forte
    elif velocity >= 90:
        return "\\mf"   # mezzo-forte
    elif velocity >= 70:
        return "\\mp"   # mezzo-piano
    else:
        return "\\p"    # piano
    

def find_closest_tone_key(color_features, COLOR_MAP):
    """
    根据中心区域提取的颜色特征 color_features（字典，key 为 HEX 字符串，value 为比例），
    在 COLOR_MAP 中寻找与该 dominant 颜色最接近的 Tone key。
    返回一个元组：
        (tone_label, dominant_hex, best_dist, target_lab)
    其中：
        - tone_label: 匹配到的 Tone 标签（例如 "V", "b", "s", ...）
        - dominant_hex: 从 color_features 中选出的主导颜色的 HEX 字符串
        - best_dist: 最小的 CIEDE2000 距离
        - target_lab: 主导颜色转换到 Lab 空间的结果
    """
    if not color_features:
        return ("C-b", None, None, None)  # 若无颜色信息，则返回默认 Tone
    
    # 取出权重最大的颜色作为主导颜色
    dominant_hex = max(color_features, key=color_features.get)
    dom_rgb = hex_to_rgb(dominant_hex)
    target_lab = rgb_to_lab(dom_rgb)
    best_dist = float('inf')
    best_tone = None

    # 遍历 COLOR_MAP，比较每个 Tone 下所有颜色与 target_lab 的距离
    for tone_label, hex_list in COLOR_MAP.items():
        for hex_color in hex_list:
            candidate_rgb = hex_to_rgb(hex_color)
            candidate_lab = rgb_to_lab(candidate_rgb)
            dist = deltaE_ciede2000(target_lab, candidate_lab)
            if dist < best_dist:
                best_dist = dist
                best_tone = tone_label

    return (best_tone, dominant_hex, best_dist, target_lab)


def extract_color_features(block):
    """
    从给定图像块（numpy 数组）中提取颜色特征。
    返回一个字典：key 为颜色的 HEX 字符串，value 为该颜色所占比例。
    为简便起见，这里直接统计每个像素的颜色出现频率。
    """
    pixels = block.reshape(-1, 3)  # 将 (h, w, 3) 重构为 (h*w, 3)
    color_counts = {}
    total = pixels.shape[0]
    for pixel in pixels:
        # 假定图像使用 RGB 顺序，转换为 HEX 表示
        hex_color = '#{:02X}{:02X}{:02X}'.format(int(pixel[0]), int(pixel[1]), int(pixel[2]))
        color_counts[hex_color] = color_counts.get(hex_color, 0) + 1
    # 计算比例
    for k in color_counts:
        color_counts[k] /= total
    return color_counts



def signed_normalize(c, score_min, score_max):
    """
    将 c 映射到 [-1, 1]:
      - 若 c=0 => 0
      - 若 c>0 => c / score_max (若 score_max!=0)
      - 若 c<0 => c / abs(score_min) (若 score_min!=0)
    """
    if abs(c) < 1e-9:
        return 0.0
    
    # 如果只有正数或只有负数，可能出现 min=0 或 max=0 的特殊情况
    if c > 0:
        if abs(score_max) < 1e-9:  # 避免除0
            return 1.0  # 或者 0.0，看你需求
        else:
            return c / score_max
    else:  # c < 0
        if abs(score_min) < 1e-9:
            return -1.0
        else:
            return c / abs(score_min)
        
def find_closest_color(base_hex, color_tone_dict):
    """
    根据中心区域的HEX颜色，在 color_tone_dict 中寻找距离最近的颜色，
    返回对应的tone值，例如 "C-b"。
    """
    base_lab = hex_to_lab(base_hex)
    best_color = None
    best_distance = float('inf')
    for hex_val in color_tone_dict.keys():
        lab = hex_to_lab(hex_val)
        # print(base_lab, lab)
        d = deltaE_ciede2000(base_lab, lab)
        if d < best_distance:
            best_distance = d
            best_color = hex_val
    return color_tone_dict[best_color]


def find_closest_tone_key_in_COLOR_MAP(hex_color, COLOR_MAP):
    """
    给定单个 hex_color，遍历 COLOR_MAP { tone_label: [hex1, hex2,...], ... }
    找到与之距离最近的 tone_label (可复用你已有的 color->lab, deltaE 逻辑)
    """
    # 这里演示最简单: 先 hex->lab
    color_lab = hex_to_lab(hex_color)
    best_label = None
    best_dist = float('inf')
    for tone_label, hex_list in COLOR_MAP.items():
        for c_hex in hex_list:
            c_lab = hex_to_lab(c_hex)
            d = deltaE_ciede2000(color_lab, c_lab)
            if d < best_dist:
                best_dist = d
                best_label = tone_label
    return best_label


def average_color(block):
    """
    计算给定图像块的平均颜色，输入为numpy数组（BGR顺序）。
    """
    avg = np.mean(block.reshape(-1, 3), axis=0)
    return avg


def extract_dominant_colors(block, num_colors=4):
    """
    利用KMeans聚类提取给定图像块的前 num_colors 个主导颜色（HEX字符串）。
    """
    pixels = block.reshape(-1, 3)
    kmeans = KMeans(n_clusters=num_colors, n_init="auto", random_state=42).fit(pixels)
    centers = kmeans.cluster_centers_
    labels, counts = np.unique(kmeans.labels_, return_counts=True)
    sorted_idx = np.argsort(-counts)
    sorted_centers = centers[sorted_idx]
    dominant_colors = []
    for center in sorted_centers:
        # center为BGR顺序，转换为RGB
        rgb = [int(x) for x in center[::-1]]
        hex_color = '#{:02X}{:02X}{:02X}'.format(*rgb)
        dominant_colors.append(hex_color)
    return dominant_colors

def compute_block_complexity(block_img):
    """
    对给定 block_img 做简单的边缘+纹理检测，返回一个 complexity_score。
    """
    h, w, _ = block_img.shape
    half_h = h // 2
    half_w = w // 2
    top_left_img = block_img[0:half_h, 0:half_w]
    top_right_img = block_img[0:half_h, half_w:w]
    bottom_left_img = block_img[half_h:h, 0:half_w]
    bottom_right_img = block_img[half_h:h, half_w:w]

    # 灰度
    tl_gray = cv2.cvtColor(top_left_img, cv2.COLOR_BGR2GRAY)
    tr_gray = cv2.cvtColor(top_right_img, cv2.COLOR_BGR2GRAY)
    bl_gray = cv2.cvtColor(bottom_left_img, cv2.COLOR_BGR2GRAY)
    br_gray = cv2.cvtColor(bottom_right_img, cv2.COLOR_BGR2GRAY)

    # Canny
    tl_edges = cv2.Canny(tl_gray, 50, 150)
    tr_edges = cv2.Canny(tr_gray, 50, 150)
    bl_edges = cv2.Canny(bl_gray, 50, 150)
    br_edges = cv2.Canny(br_gray, 50, 150)
    total_edge_count = (tl_edges>0).sum() + (tr_edges>0).sum() \
                     + (bl_edges>0).sum() + (br_edges>0).sum()

    # Laplacian
    tl_lap = cv2.Laplacian(tl_gray, cv2.CV_64F).var()
    tr_lap = cv2.Laplacian(tr_gray, cv2.CV_64F).var()
    bl_lap = cv2.Laplacian(bl_gray, cv2.CV_64F).var()
    br_lap = cv2.Laplacian(br_gray, cv2.CV_64F).var()
    total_lap = tl_lap + tr_lap + bl_lap + br_lap

    complexity_score = total_edge_count * 0.01 + total_lap * 0.05
    return complexity_score

def bgr_to_hsv(bgr):
    color = np.uint8([[bgr]])
    hsv = cv2.cvtColor(color, cv2.COLOR_BGR2HSV)[0, 0]
    return hsv

def select_notes_from_block_with_kmeans(block, base_scale):
    """
    对单个2×2格子（图像块）提取 4 个主导颜色，
    和平均颜色对比，得到4个音符。
    """
    avg = average_color(block)
    avg_hsv = bgr_to_hsv(avg)
    dominant_hexes = extract_dominant_colors(block, num_colors=4)

    notes = []
    for hex_color in dominant_hexes:
        # 将 hex_color -> BGR -> HSV
        rgb = [int(hex_color[i:i+2],16) for i in (1,3,5)]
        bgr = rgb[::-1]
        dom_hsv = bgr_to_hsv(np.array(bgr,dtype=np.uint8))
        hue_diff = float(dom_hsv[0]) - float(avg_hsv[0])
        sat_diff = float(dom_hsv[1]) - float(avg_hsv[1])

        diff = hue_diff * 1.0 + sat_diff * 0.5
        factor = 30.0
        offset = int(round(diff/factor))
        base_index = len(base_scale)//2
        note_index = base_index+offset
        if note_index<0: note_index=0
        if note_index>len(base_scale)-1:
            note_index = len(base_scale)-1
        notes.append(base_scale[note_index])
    return notes



def split_duration(duration, epsilon=0.06):
    """
    将一个时值（拍数）拆分为合法的 LilyPond 时值列表。
    允许的时值（拍数）从大到小排列：
      4.0, 2.0, 1.0, 0.5, 0.25, 0.125, 0.0625
    例如：1.25 -> [1.0, 0.25], 3.75 -> [2.0, 1.0, 0.5, 0.25]
    """
    allowed = [4.0, 2.0, 1.0, 0.5, 0.25, 0.125, 0.0625]
    parts = []
    rem = duration
    for d in allowed:
        while rem >= d - 1e-9:
            parts.append(d)
            rem -= d

    # 如果最后剩余 rem 很小，比如 < epsilon，则视作浮点数误差，忽略或附加给最后一块
    if abs(rem) < epsilon:
        # 要么直接忽略
        # pass
        # 或把这点误差并给最后一个时值(若存在)
        if parts:
            parts[-1] += rem
        rem = 0.0
    
    if abs(rem) > epsilon:
        raise ValueError(f"无法精确拆分时值：{duration}，剩余：{rem}")
    
    return parts


# 2. 将单个 (音符, 时值) 转换为 LilyPond 兼容字符串
def convert_note_old(note, duration):
    """
    将 (音符, 时值) 转换为 LilyPond 兼容格式字符串。
    如果时值正好在映射中，则直接转换；
    如果时值不在映射中，则拆分为多个合法时值，并用 tie ("~") 连接。
    """
    # 完整的音符映射（支持升、降、自然）
    note_map = {
        # 自然音
        "C": "c", "D": "d", "E": "e", "F": "f", "G": "g", "A": "a", "B": "b",
        # 升音 (#)
        "C#": "cis", "D#": "dis", "E#": "eis", "F#": "fis", "G#": "gis", "A#": "ais", "B#": "bis",
        # 降音（可以用 "-" 表示，也可以用 "b"）
        "C-": "ces", "D-": "des", "E-": "ees", "F-": "fes", "G-": "ges", "A-": "aes", "B-": "bes",
        "Cb": "b",  "Db": "des", "Eb": "ees", "Fb": "e",  "Gb": "ges", "Ab": "aes", "Bb": "bes"
    }
    
    # 将输入音符转换为 LilyPond 格式（若未在映射中则直接转小写）
    lilypond_note = note_map.get(note, note.lower())
    
    # 定义时值映射：单位为拍，4.0 拍代表全音符（LilyPond 时值 "1"）
    duration_map = {
        4.0: "1",    # 全音符
        2.0: "2",    # 二分音符
        1.0: "4",    # 四分音符
        0.5: "8",    # 八分音符
        0.25: "16",  # 十六分音符
        0.125: "32", # 三十二分音符
        0.0625: "64" # 六十四分音符
    }
    
    # 如果时值正好在映射中，直接转换
    if duration in duration_map:
        return f"{lilypond_note}{duration_map[duration]}"
    else:
        # 否则拆分时值
        parts = split_duration(duration)
        part_strs = [f"{lilypond_note}{duration_map[p]}" for p in parts]
        return " ~ ".join(part_strs)
    
def convert_note(note, duration, octave):
    """
    将 (音符, 时值, 八度) 转换为 LilyPond 兼容格式字符串。
    
    - note: 例如 "C", "D#", "B-", "A#" 等，不包含八度信息。若为休止符 "r"/"R"/"rest" 则输出休止符格式。
    - duration: 拍数(如 1.0, 0.5, 2.0)，若不在映射中会拆分并用 "~" 连接
    - octave: LilyPond 中 c' = C4(中央C)。 octave=5 => c''；octave=3 => c；octave=2 => c, ...
    
    返回: 类似 "c'4", "des''8", "fis4 ~ fis8"、或 "r4" (休止符)等 LilyPond 字符串。
    """

    # ========== 若是休止符，不要加逗号/撇号 ==========
    # 你可根据项目中约定的休止符写法( "r", "R", "rest", "sil" 等)自行扩展
    if note.lower() in ["r", "rest"]:
        # 只需映射时值，不加 octave
        duration_map = {
            4.0: "1",    # 全音符
            2.0: "2",    # 二分音符
            1.0: "4",    # 四分音符
            0.5: "8",    # 八分音符
            0.25: "16",  # 十六分音符
            0.125: "32", # 三十二分音符
            0.0625: "64" # 六十四分音符
        }
        if duration in duration_map:
            return f"r{duration_map[duration]}"
        else:
            parts = split_duration(duration)
            part_strs = [f"r{duration_map[p]}" for p in parts]
            return " ~ ".join(part_strs)

    # ========== 普通音符时，执行你原先的逻辑 ==========

    # 1) 定义音名到 LilyPond 音名(不含撇号/逗号)的基本映射
    note_map = {
        # 自然音
        "C": "c", "D": "d", "E": "e", "F": "f", "G": "g", "A": "a", "B": "b",
        # 升音 (#)
        "C#": "cis", "D#": "dis", "E#": "eis", "F#": "fis", "G#": "gis", "A#": "ais", "B#": "bis",
        # 降音（可用 "-" 或 "b"）
        "C-": "ces", "D-": "des", "E-": "ees", "F-": "fes", "G-": "ges", "A-": "aes", "B-": "bes",
        "Cb": "ces", "Db": "des", "Eb": "ees", "Fb": "fes", "Gb": "ges", "Ab": "aes", "Bb": "bes"
    }
    
    # 如果 note 不在字典，就尝试直接用小写
    lily_base = note_map.get(note, note.lower())

    # 2) 根据传入的 octave，给 LilyPond 名字加撇号(')或逗号(,)。
    #    LilyPond 的约定： c = C3, c' = C4, c'' = C5, c, = C2 ...
    difference = octave - 3
    if difference > 0:
        lily_base += "'" * difference
    elif difference < 0:
        lily_base += "," * (-difference)

    # 3) 定义拍数到 LilyPond 时值的映射
    duration_map = {
        4.0: "1",    # 全音符
        2.0: "2",    # 二分音符
        1.0: "4",    # 四分音符
        0.5: "8",    # 八分音符
        0.25: "16",  # 十六分音符
        0.125: "32", # 三十二分音符
        0.0625: "64" # 六十四分音符
    }

    # 如果时值正好在映射中，直接转换
    if duration in duration_map:
        return f"{lily_base}{duration_map[duration]}"
    else:
        # 4) 如果时值不在映射中，拆分 (例如 1.5 => [1.0, 0.5]) 并用 "~" 连接
        parts = split_duration(duration)
        part_strs = []
        for p in parts:
            if p in duration_map:
                dur_str = duration_map[p]
            else:
                # fallback: 找最近keys
                best = min(duration_map.keys(), key=lambda x: abs(x - p))
                dur_str = duration_map[best]
            part_strs.append(f"{lily_base}{dur_str}")
        return " ~ ".join(part_strs)

def convert_notes_tempo_velocity(notes_4tuple, octave, insert_tempo=False, insert_velco=False):
    """
    将 [(note, duration, tempo, velocity), ...] 转换为 LilyPond 字符串。
    当 tempo 改变(与上一音符不同)时，插入 \\tempo 4=xx 标记。
    
    返回一个 LilyPond 片段字符串，如: "\\tempo 4=88 c4 d8 e8 \\tempo 4=120 f4 ..."
    注意: 不处理 velocity -> 动态记号; 如果需要可自行扩展
    """
    lily_str_parts = []
    last_tempo = None

    for measure_idx, notes_in_measure in enumerate(notes_4tuple):
        
        if not notes_in_measure:
            continue  # 跳过空小节
        
        # 取该小节第一个音符的 tempo, velocity
        first_note_tempo = notes_in_measure[0][2]     # tempo
        first_note_velocity = notes_in_measure[0][3]  # velocity
        
        # 如果要插入 tempo，并且与上一个小节不同，就插入 \tempo
        if insert_tempo:
            if last_tempo is None or not math.isclose(first_note_tempo, last_tempo, rel_tol=1e-7):
                tempo_int = int(round(first_note_tempo))
                lily_str_parts.append(f"\\tempo 4={tempo_int}")
                last_tempo = first_note_tempo
        
        # print(notes_in_measure)
        # 遍历该小节内的所有音符
        for i, (note, duration, _, _) in enumerate(notes_in_measure):
            # 将 (note, duration) 转为 LilyPond 字符串，如 "c4", "d16"
            lily_note_str = convert_note(note, duration, octave)

            # 只在**小节的第一个音符**后追加动态标记
            # (如果想让每小节的力度都统一，以第一个音符的 velocity 为准即可)
            if i == 0 and insert_tempo:
                dyn_mark = velocity_to_dynamic(first_note_velocity)
                lily_note_str += dyn_mark

            lily_str_parts.append(lily_note_str)

        # 如果想每小节结尾加竖线或小节线：
        # lily_str_parts.append("|")  # 简单例子
        lily_str_parts.append("\\bar \"|\"")

    return " ".join(lily_str_parts)