#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
music_gen_deeplearning_advanced.py

本系统实现了从图像自动生成音乐及乐谱的全流程：
1. 利用 ResNet18 提取 512 维深度特征，并根据特征均值决定调性、模式和节奏速度。
2. 同时提取图像 HSV 均值，通过两种色彩映射函数确定调性（参考 Scriabin/Kandinsky 的色彩-调性对应）。
3. 利用 Markov 链生成和弦序列，并通过两种生成方法：
   - "dual" 模式：采用对位规则生成左右声部（使用 generate_dual_voice_measure），
   - "pattern" 模式：采用预定义伴奏模式（generate_accompaniment）生成左声部，而右声部仍采用对位方法生成。
4. 最终生成 MIDI 文件、转换为 LilyPond 记谱文本，并调用 LilyPond 生成 PDF 乐谱。
5. 关键参数（img_path、length、method、pattern_name、left_program_index、right_program_index）均可由外部自定义传入。

作者：Yao  
"""

import os
import cv2
import random
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
import torchvision.models as models
from torchvision.models import MobileNet_V2_Weights
import subprocess
import pretty_midi
import hashlib




#########################################################
# A. 深度特征提取与简单分类/回归 (示例)
#########################################################

def load_resnet18_model():
    """
    加载预训练的 ResNet18, 将 fc 替换为 Identity, 输出 512 维特征向量
    """
    model = models.resnet18(pretrained=True)
    model.fc = nn.Identity()
    model.eval()
    scripted_model = torch.jit.script(model)
    return scripted_model

def load_mobilenet_v2():
    model = models.mobilenet_v2()
    # model = models.mobilenet_v2(weights='DEFAULT')
    model.load_state_dict(torch.load("mobilenet_v2.pth", map_location="cpu"))
    model.classifier = nn.Identity()
    model.eval()
    scripted_model = torch.jit.script(model)
    return scripted_model

def extract_deep_features_bgr(image_bgr, model):
    """
    从 BGR 图片提取 512 维深度特征。
    """
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image_rgb = cv2.resize(image_rgb, (224, 224))
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225])
    ])
    tensor_img = transform(image_rgb).unsqueeze(0)  # (1,3,224,224)
    with torch.no_grad():
        feats = model(tensor_img)  # (1,512)
    vec = feats.squeeze(0).numpy()  # (512,)
    return vec

def decide_deep_params(deep_vec):
    """
    利用12个调性 (C, C#, D, D#, E, F, F#, G, G#, A, A#, B) 作为 cluster center，
    用欧式距离选择最合适的 root_note；
    若 dv_mean > 0 则为 major，否则为 minor；
    同时采用 dv_mean 的线性映射计算 tempo：
      factor = (dv_mean + 2) / 4，tempo = 60 + 80 * factor，
    其中 tempo 范围为 [60, 140] BPM。
    """
    all_keys = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"]
    cluster_centers = {}
    offsets = np.linspace(-0.5, 0.5, len(all_keys))
    for i, k in enumerate(all_keys):
        center = np.random.randn(1280) * 0.1 + offsets[i]
        cluster_centers[k] = center

    best_key = None
    best_dist = 1e9
    for k, center in cluster_centers.items():
        dist = np.linalg.norm(deep_vec - center)
        if dist < best_dist:
            best_dist = dist
            best_key = k

    dv_mean = np.mean(deep_vec)
    scale_mode = "major" if dv_mean > 0 else "minor"
    factor = (dv_mean + 2) / 4
    factor = max(0, min(1, factor))
    tempo = 60 + int(80 * factor)
    return best_key, scale_mode, tempo

#########################################################
# B. HSV 与色彩映射
#########################################################

def color_to_tonality(h_mean, s_mean, v_mean):
    """
    将 hue (0~180) 等分为12个区间（每15度），返回对应的调性（例如 "C", "C#", "D", ...）。
    """
    all_keys = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"]
    idx = int(h_mean // 15)
    if idx >= 12:
        idx = 11
    return all_keys[idx]

def color_to_tonality_new(h_mean, s_mean, v_mean):
    """
    根据 Scriabin 所倡导的十二大调色彩分配：
      每15度一个区间，返回一个三元组：(key, color_desc, mystic_set)
    例如：
      0~15   -> ("C", "red, plain", [0,6,10,4,9,2])
      15~30  -> ("G", "orange (red-yellow, fiery)", [7,1,5,11,4,9])
      ...
      165~180-> ("F", "dark red", [5,11,3,9,2,7])
    """
    data = [
        (15, "C",  "red, plain",                [0,6,10,4,9,2]),
        (30, "G",  "orange (red-yellow, fiery)",[7,1,5,11,4,9]),
        (45, "D",  "sunny yellow",              [2,8,0,6,11,4]),
        (60, "A",  "green, grass green",        [9,3,7,1,6,11]),
        (75, "E",  "blue greenish (light blue)",[4,10,2,8,1,6]),
        (90, "B",  "dark blue with light blue", [11,5,9,3,8,1]),
        (105,"F#", "dark blue with violet shade", [6,0,4,10,3,8]),
        (120,"Db", "pure violet",               [1,7,11,5,10,3]),
        (135,"Ab", "lily-colored, reddish",     [8,2,6,0,5,10]),
        (150,"Eb", "steely blue, metallic",     [3,9,1,7,0,5]),
        (165,"Bb", "metallic leaden grey",      [10,4,8,2,7,0]),
        (180,"F",  "dark red",                  [5,11,3,9,2,7]),
    ]
    for boundary, key, color_desc, mystic_set in data:
        if h_mean < boundary:
            return key, color_desc, mystic_set
    return "F", "dark red", [5,11,3,9,2,7]

#########################################################
# C. 随机性设置（确保同一图像生成相同结果）
#########################################################

def set_deterministic_seed(deep_vec):
    """
    利用 deep_vec 的字节哈希值设置 random.seed，确保同一图像产生相同的随机序列
    """
    # 使用 hashlib.md5 得到一个确定性的 128 位哈希值
    h = hashlib.md5(deep_vec.tobytes()).hexdigest()
    # 取前 8 位转为 int（32 位种子）
    s = int(h[:8], 16)
    random.seed(s)
    np.random.seed(s)

#########################################################
# D. 对位与和弦生成相关函数
#########################################################

# 传统功能和弦映射
major_map = {
    "I": [0, 4, 7],
    "ii": [2, 5, 9],
    "iii": [4, 7, 11],
    "IV": [5, 9, 0],
    "V": [7, 11, 2],
    "vi": [9, 0, 4],
    "vii°": [11, 2, 5]
}
minor_map = {
    "i": [0, 3, 7],
    "ii°": [2, 5, 8],
    "III": [4, 7, 11],
    "iv": [5, 8, 0],
    "v": [7, 10, 2],
    "VI": [8, 0, 3],
    "VII": [10, 2, 5]
}

NOTE_BASE_MAP = {
    "C": 0, "C#": 1, "Db": 1, "D": 2, "D#": 3, "Eb": 3, "E": 4,
    "F": 5, "F#": 6, "Gb": 6, "G": 7, "G#": 8, "Ab": 8,
    "A": 9, "A#": 10, "Bb": 10, "B": 11, "Cb": 11
}

def chord_pcs_in_scale(chord_label, root_note, scale_mode="major", use_scriabin=False):
    """
    生成和弦音集合：
      - 当 use_scriabin 为 True 时，直接返回 Scriabin 的 mystic chord 集合；
      - 否则采用传统的 major_map/minor_map，并进行移调。
    """
    scriabin_sets = {
        "C":  [0,6,10,4,9,2],
        "G":  [7,1,5,11,4,9],
        "D":  [2,8,0,6,11,4],
        "A":  [9,3,7,1,6,11],
        "E":  [4,10,2,8,1,6],
        "B":  [11,5,9,3,8,1],
        "F#": [6,0,4,10,3,8],
        "Db": [1,7,11,5,10,3],
        "Ab": [8,2,6,0,5,10],
        "Eb": [3,9,1,7,0,5],
        "Bb": [10,4,8,2,7,0],
        "F":  [5,11,3,9,2,7]
    }
    if use_scriabin:
        return scriabin_sets.get(root_note, [0,6,10,4,9,2])
    else:
        if scale_mode == "major":
            base_ints = major_map.get(chord_label, [0, 4, 7])
        else:
            base_ints = minor_map.get(chord_label, [0, 3, 7])
        root_val = NOTE_BASE_MAP.get(root_note, 0)
        return [(root_val + i) % 12 for i in base_ints]

def build_markov_states(scale_mode):
    if scale_mode == "major":
        return ["I", "ii", "iii", "IV", "V", "vi", "vii°"]
    else:
        return ["i", "ii°", "III", "iv", "v", "VI", "VII"]

def build_markov_transition(states, s_mean, v_mean):
    base_trans = {}
    for st in states:
        base_trans[st] = {}
        for nxt in states:
            base_trans[st][nxt] = 0.1
        if "V" in base_trans[st]:
            base_trans[st]["V"] += 0.2 * (s_mean / 255)
        if "vi" in base_trans[st]:
            base_trans[st]["vi"] += 0.1 * (1 - v_mean / 255)
    for st in states:
        total = sum(base_trans[st].values())
        for nxt in states:
            base_trans[st][nxt] /= total
    return base_trans

def generate_chord_sequence(states, transition, length=8):
    current = random.choice(states)
    seq = [current]
    for _ in range(length - 1):
        probs = transition[current]
        nxt = random.choices(list(probs.keys()), list(probs.values()))[0]
        seq.append(nxt)
        current = nxt
    return seq

def generate_dual_voice_measure(chord_pcs, beats=4, last_interval=None, hist_notes=(None, None), max_jump=7, step_threshold=2):
    """
    为一个小节生成左右声部音符，并遵循对位规则：
      - 避免平行五度/八度；
      - 大跳后接向级进行（要求反向且步幅不超过 2）。
    返回 (right_list, left_list, new_interval, new_hist)。
    """
    right_list = []
    left_list = []
    second_last_r, last_r = hist_notes

    # 左声部采用传统方案：这里你可以改为调用 pick_bass_in_chord 实现多样化
    bass_pc = chord_pcs[0]
    bass_midi = 12 * (2 + 1) + bass_pc  # octave = 2

    current_interval = last_interval
    current_second_last = second_last_r
    current_last_r = last_r

    for _ in range(beats):
        trials = 0
        candidate_midi = None
        while trials < 30:
            pc = random.choice(chord_pcs)
            octv = 4 if random.random() < 0.85 else 5
            mel_midi = 12 * (octv + 1) + pc
            new_interval = (bass_midi, mel_midi)
            # 这里调用 is_parallel_perfect 与 check_big_leap_direction（需保证这两个函数已实现）
            # 简单模拟检查
            if current_interval is not None and abs(new_interval[1]-new_interval[0]) in (7,12):
                trials += 1
                continue

            # 检查与前一个右声部音符的跳跃幅度，若已有前音，则要求不超过max_jump
            if current_last_r is not None and abs(mel_midi - current_last_r) > max_jump:
                trials += 1
                continue

            candidate_midi = mel_midi
            break

        if candidate_midi is None:
            if current_last_r is not None:
                # 搜索当前和弦在较低和较高八度中与前一个音接近的候选音
                possible_candidates = []
                for pc in chord_pcs:
                    for octv in [4, 5]:
                        candidate = 12 * (octv + 1) + pc
                        if abs(candidate - current_last_r) <= max_jump:
                            possible_candidates.append(candidate)
                if possible_candidates:
                    # 选择与前一个音距离最小的候选
                    candidate_midi = min(possible_candidates, key=lambda x: abs(x - current_last_r))
                else:
                    candidate_midi = bass_midi + 12  # 最后备用
            else:
                candidate_midi = bass_midi + 12

        right_list.append(candidate_midi)
        left_list.append(bass_midi)

        current_second_last = current_last_r
        current_last_r = candidate_midi
        current_interval = (bass_midi, candidate_midi)

    return right_list, left_list, current_interval, (current_second_last, current_last_r)

#########################################################
# E. 伴奏模式相关函数
#########################################################

# 常见伴奏模式库
ACCOMPANIMENT_PATTERNS = {
    # 1) Alberti Bass (4/4) => 下-上-中-上
    #   每小节 4 拍, pattern中记录[(拍子, 音位)]:
    "alberti_4_4": [
        (0.0, "lowest"),   # 拍0 下
        (1.0, "highest"),  # 拍1 上
        (2.0, "middle"),   # 拍2 中
        (3.0, "highest"),  # 拍3 上
    ],

    # 2) Boom-chick (4/4) => 第0拍弹低音, 第2拍弹和弦
    "boomchick_4_4": [
        (0.0, "lowest"),   # Bass
        (2.0, "chord"),    # Chord block
    ],

    # 3) Waltz (3/4) => 第0拍低音, 第1拍和第2拍弹和弦
    "waltz_3_4": [
        (0.0, "lowest"),   # Bass
        (1.0, "chord"),    # Chord
        (2.0, "chord"),    # Chord
    ],

    # 4) Simple arpeggio up (4/4), 在每拍演奏 chord 的 不同音
    "arp_up_4_4": [
        (0.0, "lowest"),
        (1.0, "next"),
        (2.0, "next"),
        (3.0, "next"),
    ],

    # 5) Ostinato 16 (4/4), 假设每小节 8次16分(只是示例)
    #   例如 pattern写 [ (0.0, "lowest"), (0.5,"next"), (1.0,"next")... ]
    #   user可再自定义
    "ostinato_16": [
        (0.0, "lowest"),
        (0.5, "next"),
        (1.0, "middle"),
        (1.5, "next"),
        (2.0, "lowest"),
        (2.5, "next"),
        (3.0, "middle"),
        (3.5, "next"),
    ],
    # ... 可添加更多 pattern ...
}



def pick_note_from_chord(chord_pcs_sorted, note_position="lowest", lowestBass=36, highestBass=60, 
                         velocity=80, prev_note=None, max_jump=7, cyc_idx_dict=None):
    """
    根据 note_position 从 chord_pcs_sorted 中选择一个音符，并保证生成的音符固定在一个八度内，
    同时检查与前一个左声部音符的跳跃幅度不超过 max_jump，避免出现突然的高八度。
    note_position 可为 "lowest", "highest", "middle", "random", "next", "chord" 等。
    返回 (pitch_list, velocity_list)。
    
    参数说明：
      - chord_pcs_sorted: 和弦音的音高（以音高类，即 0~11 表示音名）列表，已排序。
      - lowestBass, highestBass: 指定允许的音高范围，但这里我们将音符固定在一个八度内，
            所以我们会选用一个合适的八度（例如固定为 octv = 3）。
      - prev_note: 前一个左声部音符，用于平滑跳跃。
      - max_jump: 允许的最大跳跃音程（半音数）。
      - cyc_idx_dict: 用于 "next" 模式循环索引。
    """
    # 固定使用的八度：例如设为 3，即构造音符时用 12*(3+1)=48 作为基础，这样生成的音在 [48, 59] 内
    fixed_octv = 3

    def build_candidate(pc):
        octave = (prev_note // 12) if prev_note else fixed_octv + 1  # 计算出一个合理的八度
        return (octave * 12) + (pc % 12)

    def get_base_pitch():
        # 按照 note_position 选择一个和弦内的音（仅返回音名 0~11）
        if not chord_pcs_sorted:
            return 0
        # if note_position == "chord":
             # 如果是 "chord"，返回整个列表（注意：这种情况在左声部中一般不用）
        #     return chord_pcs_sorted[:]  
        if note_position == "lowest":
            return chord_pcs_sorted[0]
        elif note_position == "highest":
            return chord_pcs_sorted[-1]
        elif note_position == "middle":
            mid = len(chord_pcs_sorted) // 2
            return chord_pcs_sorted[mid]
        elif note_position == "random":
            return random.choice(chord_pcs_sorted)
        elif note_position == "next":
            if cyc_idx_dict is None:
                cyc_idx_dict = {"arp_index": 0}
            idx = cyc_idx_dict["arp_index"]
            pitch = chord_pcs_sorted[idx % len(chord_pcs_sorted)]
            cyc_idx_dict["arp_index"] = idx + 1
            return pitch
        else:
            return chord_pcs_sorted[0]
    
    candidate = None
    trials = 0
    while trials < 30:
        base_pitch = get_base_pitch()
        cand = build_candidate(base_pitch)
        # 检查是否在允许范围内（虽然固定八度一般都在范围内）
        if cand < lowestBass or cand > highestBass:
            trials += 1
            continue

        # 检查与前一个音符的跳跃幅度，若已有前音则要求不超过 max_jump
        if prev_note is not None:
            if abs(cand - prev_note) > max_jump or (cand // 12 != prev_note // 12):
                trials += 1
                continue

        # 可在这里加入其它规则检查，例如避免与右声部产生平行完美（需传入对应右声部信息）
        candidate = cand
        break

    # 如果多次尝试后没有找到合适的候选，则采用回退策略：
    if candidate is None:
        # 在固定八度内，对所有和弦音计算候选值
        possible = [build_candidate(pc) for pc in chord_pcs_sorted if lowestBass <= build_candidate(pc) <= highestBass]
        if prev_note is not None and possible:
            # 选择与 prev_note 差距最小的候选
            # candidate = min(possible, key=lambda x: abs(x - prev_note))
            candidate = min(possible, key=lambda x: (abs(x - prev_note), abs((x // 12) - (prev_note // 12))))
        elif possible:
            candidate = possible[0]
        else:
            # 如果所有候选都不合适，则直接返回固定八度内最低的音
            candidate = build_candidate(chord_pcs_sorted[0])
    
    return ([candidate], [velocity])

def generate_accompaniment(
    chord_pcs_sorted, 
    pattern_name,
    start_time=0.0,
    beats_per_bar=4,
    tempo=120,
    lowestBass=36,
    highestBass=60,
    velocityBase=80,
    cyc_idx_dict=None
):
    """
    生成一个小节的伴奏事件（单声部），采用指定的伴奏模式。
    参数:
      chord_pcs_sorted: 升序和弦音列表 (例如 [60,64,67])
      pattern_name: 在 ACCOMPANIMENT_PATTERNS 里查找的模式名称
      start_time: 小节起始时间（秒）
      beats_per_bar: 每小节拍数
      tempo: BPM
      lowestBass, highestBass: 音域限制
      velocityBase: 基础力度
      cyc_idx_dict: 用于 "next" 模式的循环索引（可选）
    返回:
      note_events: 列表，每个元素为 (on_time, off_time, pitch_list, velocity_list)
    """
    if pattern_name not in ACCOMPANIMENT_PATTERNS:
        print(f"Warning: pattern {pattern_name} not found, fallback to 'alberti_4_4'")
        pattern_name = "alberti_4_4"
    pattern = ACCOMPANIMENT_PATTERNS[pattern_name]

    note_events = []
    spb = 60.0 / tempo  # 每拍秒数

    for entry in pattern:
        if len(entry) == 2:
            beat_offset, note_position = entry
            duration_factor = 0.5
        else:
            beat_offset, note_position, duration_factor = entry

        event_on = start_time + beat_offset * spb
        event_off = event_on + duration_factor * spb

        pitches, velocities = pick_note_from_chord(
            chord_pcs_sorted,
            note_position=note_position,
            lowestBass=lowestBass,
            highestBass=highestBass,
            velocity=velocityBase,
            prev_note=None, 
            max_jump=7,
            cyc_idx_dict=cyc_idx_dict
        )
        note_events.append((event_on, event_off, pitches, velocities))
    return note_events

#########################################################
# F. MIDI 与 LilyPond 辅助函数
#########################################################

def merge_consecutive_notes(notes, epsilon=1e-9):
    sorted_notes = sorted(notes, key=lambda x: x.start)
    merged = []
    for note in sorted_notes:
        if not merged:
            merged.append(note)
        else:
            last = merged[-1]
            if (abs(note.start - last.end) < epsilon and
                note.pitch == last.pitch and
                note.velocity == last.velocity):
                last.end = note.end
            else:
                merged.append(note)
    return merged

def merge_measure(measure):
    merged = []
    if not measure:
        return merged
    current = measure[0]
    count = 1
    for n in measure[1:]:
        if n == current:
            count += 1
        else:
            merged.append((current, count))
            current = n
            count = 1
    merged.append((current, count))
    return merged

def merge_measures(measures):
    return [merge_measure(m) for m in measures]

def duration_token(count):
    mapping = {1: "4", 2: "2", 3: "2.", 4: "1"}
    return mapping.get(count, "4")

def midi_to_lily_pitch(midi_val):
    pitch_class = midi_val % 12
    octave = (midi_val // 12) - 1
    names = ["c", "cis", "d", "dis", "e", "f", "fis", "g", "gis", "a", "ais", "b"]
    base = names[pitch_class]
    shift = octave - 3
    if shift > 0:
        base += "'" * shift
    elif shift < 0:
        base += "," * (-shift)
    return base

def measures_to_lily_merged(merged_measures):
    lines = []
    for measure in merged_measures:
        tokens = []
        for (midiv, count) in measure:
            note_str = midi_to_lily_pitch(midiv) + duration_token(count)
            tokens.append(note_str)
        line = " ".join(tokens) + " |"
        lines.append(line)
    return "\n".join(lines)

def convert_key_for_lily(key):
    mapping = {
        "c#": "cis",
        "d#": "dis",
        "f#": "fis",
        "g#": "gis",
        "a#": "ais",
        "bb": "bes",
        "eb": "ees",
        "gb": "ges",
        "ab": "aes",
        "cb": "ces"
    }
    k = key.lower()
    return mapping.get(k, k)

def convert_ly_to_pdf(ly_file, output_dir=None):
    # 将输入路径转换为绝对路径
    ly_file_abs = os.path.abspath(ly_file)
    if not os.path.isfile(ly_file_abs):
        raise FileNotFoundError(f"找不到 {ly_file_abs}")
    
    if output_dir:
        # 将输出目录也转换为绝对路径
        output_dir_abs = os.path.abspath(output_dir)
        os.makedirs(output_dir_abs, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(ly_file_abs))[0]
        out_prefix = os.path.join(output_dir_abs, base_name)
        command = ["lilypond", "-fpdf", "-o", out_prefix, ly_file_abs]
        pdf_file = f"{out_prefix}.pdf"
    else:
        base_name = os.path.splitext(os.path.basename(ly_file_abs))[0]
        pdf_file = f"{base_name}.pdf"
        command = ["lilypond", "-fpdf", ly_file_abs]

    try:
        subprocess.run(command, check=True)
        print(f"[INFO] PDF生成成功: {pdf_file}")
    except subprocess.CalledProcessError:
        print("LilyPond转换失败.")
        return None
    return pdf_file


def from_midi_to_mp3(midi_file):
    """
    将 MIDI 文件转换为 WAV/MP3 的逻辑示例
    假设你通过 fluidsynth + ffmpeg/ffmpeg 来实现
    """
    wav_file = midi_file.replace(".mid", ".wav")
    mp3_file = midi_file.replace(".mid", ".mp3")

    # 1) 用 fluidsynth 生成 WAV
    subprocess.run([
        "fluidsynth",
        "-ni",
        "/usr/share/sounds/sf2/FluidR3_GM.sf2", 
        midi_file,
        "-F", wav_file,
        "-r", "44100"
    ], check=True)

    # 2) 用 ffmpeg 将 WAV 转成 MP3
    subprocess.run([
        "ffmpeg",
        "-y",
        "-i", wav_file,
        "-acodec", "libmp3lame",
        mp3_file
    ], check=True)

    print(f"已生成 MP3 文件: {mp3_file}")

def convert_to_type0(pretty_midi_obj, output_filename, left_program_index, right_program_index):
    """
    Convert a PrettyMIDI object to Type 0 while preserving instrument information
    for left and right parts.
    左手乐器（名称中含 "Left"）将设为 channel 1 和 left_program_index，
    右手乐器（名称中含 "Right"）设为 channel 2 和 right_program_index.
    """

    type0_midi = pretty_midi.PrettyMIDI()
    merged_instr = pretty_midi.Instrument(program=0, name="MergedTrack", is_drum=False)

    for instr in pretty_midi_obj.instruments:
        print(f"Copying notes from {instr.name} ({len(instr.notes)} notes)")
        if "Left" in instr.name:
            # 保留左手乐器信息
            instr.program = left_program_index
            for note in instr.notes:
                note.channel = 1  # 左声部设为 channel 1
        elif "Right" in instr.name:
            # 保留右手乐器信息
            instr.program = right_program_index
            for note in instr.notes:
                note.channel = 2  # 右声部设为 channel 2
        else:
            # 如果不是左右声部，默认保持不变
            pass

        merged_instr.notes.extend(instr.notes)

    if len(merged_instr.notes) == 0:
        print("[ERROR] No notes found in converted MIDI!")
    else:
        print(f"[INFO] Successfully merged {len(merged_instr.notes)} notes.")

    merged_instr.notes.sort(key=lambda note: note.start)
    type0_midi.instruments.append(merged_instr)
    type0_midi.write(output_filename)
    print(f"[INFO] Converted to Type 0 MIDI: {output_filename}")

    print(f"Number of tracks: {len(type0_midi.instruments)}")
    for i, instrument in enumerate(type0_midi.instruments):
        print(f"Track {i}: {instrument.name}, Program: {instrument.program}")


def pick_left_voice_note(chord_pcs_sorted, note_position="lowest", 
                         fixed_octv=3,
                         lowestBass=36, highestBass=60, 
                         velocity=80, prev_note=None, max_jump=7, cyc_idx_dict=None,
                         prev_right=None, current_right=None):
    """
    从 chord_pcs_sorted 中选择一个左声部音符，使其固定在指定八度内，
    并检查与前一个左声部音符的跳跃不超过 max_jump，且避免与右声部产生平行完美。
    
    chord_pcs_sorted 中的数字视为音名（0～11），候选音通过固定八度构造为：base_octave_base + (pc % 12)
    
    新增参数：
      - prev_right: 前一拍右声部音符（若有）
      - current_right: 当前拍右声部音符
    返回 (pitch_list, velocity_list)。
    """
    # 固定八度内的基础音，例如 fixed_octv=3 对应 12*(3+1)=48
    base_octave_base = 12 * (fixed_octv + 1)  # 例如 48

    def build_candidate(pc):
        # 强制将 pc 限定在 0～11 内，再加上固定八度基准
        return base_octave_base + pc

    def get_base_pitch():
        if not chord_pcs_sorted:
            return 0
        if note_position == "lowest":
            return chord_pcs_sorted[0]
        elif note_position == "highest":
            return chord_pcs_sorted[-1]
        elif note_position == "middle":
            mid = len(chord_pcs_sorted) // 2
            return chord_pcs_sorted[mid]
        elif note_position == "random":
            return random.choice(chord_pcs_sorted)
        elif note_position == "next":
            if cyc_idx_dict is None:
                cyc_idx_dict = {"arp_index": 0}
            idx = cyc_idx_dict["arp_index"]
            pitch = chord_pcs_sorted[idx % len(chord_pcs_sorted)]
            cyc_idx_dict["arp_index"] = idx + 1
            return pitch
        else:
            return chord_pcs_sorted[0]
    
    candidate = None
    trials = 0
    while trials < 30:
        base_pitch = get_base_pitch()
        cand = build_candidate(base_pitch)
        
        # 检查是否在允许范围内
        if cand < lowestBass or cand > highestBass:
            trials += 1
            continue

        # 检查与前一个左声部音符的跳跃幅度，若有前音则要求不超过 max_jump
        if prev_note is not None and abs(cand - prev_note) > max_jump:
            trials += 1
            continue

        # 新增：避免与右声部产生平行完美
        # 如果存在上一拍和当前拍右声部音符，则判断
        if prev_note is not None and prev_right is not None and current_right is not None:
            interval_prev = abs(prev_right - prev_note)
            interval_candidate = abs(current_right - cand)
            if interval_prev in (7, 12) and interval_candidate in (7, 12):
                # 产生平行完美，放弃此候选
                trials += 1
                continue

        candidate = cand
        break
    # 如果 30 次尝试后仍未找到合适候选，则采用回退策略
    if candidate is None:
        possible = [build_candidate(pc) for pc in chord_pcs_sorted]
        # 过滤出在允许范围内的
        possible = [p for p in possible if lowestBass <= p <= highestBass]
        if prev_note is not None and possible:
            candidate = min(possible, key=lambda x: abs(x - prev_note))
        elif possible:
            candidate = possible[0]
        else:
            candidate = build_candidate(chord_pcs_sorted[0])
    
    return ([candidate], [velocity])

#########################################################
# G. 主入口: 生成 MIDI, LY, PDF
#########################################################

def generate_music(
    img_path,
    length=24,
    out_midi="final_out.mid",
    out_ly="final_out.ly",
    out_pdf_dir="outputs",
    method="dual",              # "dual" 或 "pattern"
    pattern_name="alberti_4_4",   # 当 method=="pattern" 时使用
    left_program_index=32,        # 左声部 MIDI 程序号，默认 32 (Acoustic Bass)
    right_program_index=0         # 右声部 MIDI 程序号，默认 0 (Acoustic Grand Piano)
):
    """
    生成 MIDI、LilyPond 和 PDF 文件。
    
    流程：
      1. 从图像中提取 HSV 均值和深度特征，确定调性、模式和节奏速度。
      2. 利用 Markov 链生成和弦序列。
      3. 根据 method 选择生成方式：
         - method="dual": 使用 generate_dual_voice_measure 生成左右声部。
         - method="pattern": 右声部采用对位模式，左声部采用伴奏模式（generate_accompaniment）。
      4. 将生成的音符写入 MIDI，并转换为 LilyPond 文本，调用 LilyPond 生成 PDF。
    
    参数:
      img_path: 图像文件路径
      length: 小节数
      out_midi, out_ly, out_pdf_dir: 输出文件路径配置
      method: "dual" 或 "pattern"
      pattern_name: 当 method=="pattern" 时，为左声部伴奏选用的模式名称
      left_program_index: 左声部的 MIDI 程序号（如 32 表示 Acoustic Bass）
      right_program_index: 右声部的 MIDI 程序号（如 0 表示 Acoustic Grand Piano）
    """
    # a) 读取图像与 HSV 均值
    img_bgr = cv2.imread(img_path)
    img_name = os.path.splitext(os.path.basename(img_path))[0]

    if img_bgr is None:
        raise FileNotFoundError(f"无法读取 {img_path}")
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    h_mean = np.mean(hsv[:, :, 0])
    s_mean = np.mean(hsv[:, :, 1])
    v_mean = np.mean(hsv[:, :, 2])
    
    # b) 提取深度特征
    model = load_mobilenet_v2()
    # model = load_resnet18_model()
    deep_vec = extract_deep_features_bgr(img_bgr, model)
    print(deep_vec)
    
    # 固定随机种子，确保相同图像生成相同随机序列
    set_deterministic_seed(deep_vec)
    
    # c) 根据深度特征确定调性与节奏参数
    deep_root, deep_scale, deep_tempo = decide_deep_params(deep_vec)
    
    # 结合色彩映射（注意 color_to_tonality_new 返回三元组，此处只取第一个元素）
    color_root = color_to_tonality(h_mean, s_mean, v_mean)
    color_root2, _, _ = color_to_tonality_new(h_mean, s_mean, v_mean)
    final_root = color_root2 if random.random() < 0.5 else color_root
    if random.random() < 0.5:
        deep_root = final_root
    
    # d) 构建 Markov 状态与和弦序列
    states = build_markov_states(deep_scale)
    transition = build_markov_transition(states, s_mean, v_mean)
    chord_seq = generate_chord_sequence(states, transition, length=length)
    print(f"[INFO] chord_seq= {chord_seq}, root= {deep_root}, scale= {deep_scale}, tempo= {deep_tempo}")
    
    # e) 根据 method 选择生成方式
    right_all = []
    left_all  = []
    
    if method == "dual":
        last_interval = None
        hist_notes = (None, None)
        for chord_label in chord_seq:
            chord_pcs = chord_pcs_in_scale(chord_label, deep_root, deep_scale, use_scriabin=False)
            r_ms, l_ms, new_int, new_hist = generate_dual_voice_measure(
                chord_pcs, beats=4,
                last_interval=last_interval, hist_notes=hist_notes
            )
            right_all.append(r_ms)
            left_all.append(l_ms)
            last_interval = new_int
            hist_notes = new_hist
    else:
        # method == "pattern"
        # 右声部仍采用对位模式
        
        last_interval_r = None
        hist_notes_r = (None, None)
        spb = 60.0 / deep_tempo
        bar_duration = 4 * spb
        current_time = 0.0
        cyc_dict = {}
        for chord_label in chord_seq:
            chord_pcs = chord_pcs_in_scale(chord_label, deep_root, deep_scale, use_scriabin=False)
            # 右声部对位生成
            r_ms, _, new_int_r, new_hist_r = generate_dual_voice_measure(
                chord_pcs, beats=4,
                last_interval=last_interval_r,
                hist_notes=hist_notes_r
            )
            right_all.append(r_ms)
            last_interval_r = new_int_r
            hist_notes_r = new_hist_r

            
            # 左声部使用伴奏模式生成
            events = generate_accompaniment(
                chord_pcs_sorted=chord_pcs,
                pattern_name=pattern_name,
                start_time=current_time,
                beats_per_bar=4,
                tempo=deep_tempo,
                lowestBass=36,
                highestBass=60,
                velocityBase=80,
                cyc_idx_dict=cyc_dict
            )
            current_time += bar_duration
            # 将每小节 events 简化为 4 拍音高列表（这里只取每拍第一个事件的音符）
            measure_pitch_list = [60, 60, 60, 60]
            for (on_t, off_t, pitches, velocities) in events:
                rel_time = on_t - (current_time - bar_duration)
                beat_idx = int(rel_time // spb)
                if beat_idx < 0:
                    beat_idx = 0
                if beat_idx > 3:
                    beat_idx = 3
                if pitches:
                    measure_pitch_list[beat_idx] = pitches[0]
            left_all.append(measure_pitch_list)
        
        """
        # 遍历每个小节的 right_all（例如，每个元素都是一个有 4 个拍音高的列表）
        
    
        for measure_right in right_all:
            prev_left = None   # 用于记录前一拍左声部音符，用于平滑过渡
            measure_left = []   # 存放当前小节每拍的左声部音符
            # 对当前小节的每一拍循环
            for beat_idx, current_right in enumerate(measure_right):
                # 如果当前拍不是第一拍，则上一拍右声部音符取当前小节中上一拍，否则为空
                prev_right = measure_right[beat_idx-1] if beat_idx > 0 else None
                    
                # 调用 pick_left_voice_note 生成当前拍的左声部音符
                left_voice = pick_left_voice_note(
                    chord_pcs_sorted=chord_pcs,         # 和弦音列表（例如：[0, 4, 7]）
                    note_position="lowest",
                    fixed_octv=3,                       # 固定八度，例如固定在 48～59 之间
                    lowestBass=36,
                    highestBass=60,
                    velocity=80,
                    prev_note=prev_left,                # 上一拍左声部音符（如果有）
                    max_jump=7,
                    prev_right=prev_right,              # 上一拍右声部音符（如果有）
                    current_right=current_right         # 当前拍右声部音符
                )
                # left_voice 返回 ([pitch], [velocity])，取第一个音高作为当前拍的左声部
                pitch = left_voice[0][0]
                measure_left.append(pitch)
                # 更新 prev_left 为当前拍的左声部音符，供下一拍使用
                prev_left = pitch

            left_all.append(measure_left)
            """

    # f) 写 MIDI 文件
    pm = pretty_midi.PrettyMIDI()

    right_instr = pretty_midi.Instrument(program=right_program_index, name="RightHand")
    left_instr  = pretty_midi.Instrument(program=left_program_index, name="LeftHand")
    sec_per_beat = 60.0 / deep_tempo
    current_time = 0.0
    for measure_i in range(len(right_all)):
        for b in range(4):  # 每小节 4 拍
            start_t = current_time
            end_t = current_time + sec_per_beat
            r_midi = right_all[measure_i][b]
            l_midi = left_all[measure_i][b]
            nr = pretty_midi.Note(velocity=100, pitch=r_midi, start=start_t, end=end_t)
            nl = pretty_midi.Note(velocity=80, pitch=l_midi, start=start_t, end=end_t)
            right_instr.notes.append(nr)
            left_instr.notes.append(nl)
            current_time = end_t
    right_instr.notes = merge_consecutive_notes(right_instr.notes)
    left_instr.notes = merge_consecutive_notes(left_instr.notes)
    pm.instruments.append(right_instr)
    pm.instruments.append(left_instr)
    pm.write(out_midi)
    from_midi_to_mp3(out_midi)

    convert_to_type0(pm, out_midi, left_program_index, right_program_index)

    # g) 生成 LilyPond 文本并转换为 PDF
    merged_right = merge_measures(right_all)
    merged_left = merge_measures(left_all)
    right_lily = measures_to_lily_merged(merged_right)
    left_lily = measures_to_lily_merged(merged_left)
    

    def add_dyn(lily):
        # 先按照小节分割，假设每个小节以 "|" 结尾
        measures = [m.strip() for m in lily.split("|") if m.strip()]
        total_measures = len(measures)

        # 定义分段：前 25% 的小节为 seg1，后 25% 为 seg3，中间为 seg2
        seg1_measures = measures[: max(1, total_measures // 4)]
        seg3_measures = measures[-max(1, total_measures // 4):]
        seg2_measures = measures[max(1, total_measures // 4): total_measures - max(1, total_measures // 4)]

        # 重新拼接为字符串，每个小节后加上竖线
        seg1 = " | ".join(seg1_measures) + " |"
        seg2 = " | ".join(seg2_measures) + " |"
        seg3 = " | ".join(seg3_measures) + " |"


        segment1_with_dyn = f"{{ \\p {seg1} r4 \\< }}"
        segment2_with_dyn = f"{{ \\! \\f {seg2} r4 }}"
        segment3_with_dyn = f"{{ \\> {seg3} r4 \\! \\mp }}"

        # 最终的右手音符串
        final_lily = segment1_with_dyn + segment2_with_dyn + segment3_with_dyn
        return final_lily
    
    final_right_lily = add_dyn(right_lily)
    final_left_lily = add_dyn(left_lily)

    print(len(final_right_lily), len(final_left_lily))


    lily_root = convert_key_for_lily(deep_root)

    lily_content = f"""\\version "2.24.1"
    \\header {{
        title = "{img_name}"
        % composer = "Yao."
    }}
    \\score {{
        \\new PianoStaff <<
            \\new Staff = "right" {{
                \\clef treble
                \\key {lily_root} \\{deep_scale}
                \\tempo 4={deep_tempo}
                {right_lily}
                \\bar "|."
            }}
            \\new Staff = "left" {{
                \\clef bass
                \\key {lily_root} \\{deep_scale}
                {left_lily}
                \\bar "|."
            }}
        >>
        \\layout {{}}
        \\midi {{}}
    }}
    """
    with open(out_ly, "w", encoding="utf-8") as f:
        f.write(lily_content)
    print("[INFO] LilyPond 写入:", out_ly)
    
    pdf_file = convert_ly_to_pdf(out_ly, output_dir=out_pdf_dir)
    if pdf_file:
        print("[INFO] 生成PDF:", pdf_file)

    png_file = pdf_file[:-4] + ".png"
    subprocess.run(
        ["convert", 
        "-density", "300", 
        pdf_file, 
        png_file
    ],check=True)
    if png_file:
        print("[INFO] 生成PNG:", png_file)


    
if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser(
        description="Generate music from image using deep learning-based algorithm."
    )
    parser.add_argument("--img_path", type=str, required=True, help="Path to the input image.")
    parser.add_argument("--length", type=int, default=24, help="Number of measures (chord sequence length).")
    parser.add_argument("--method", type=str, choices=["dual", "pattern"], default="dual",
                        help="Generation method: 'dual' for dual-voice counterpoint, 'pattern' for accompaniment pattern mode.")
    parser.add_argument("--pattern_name", type=str, default="alberti_4_4",
                        help="When method is 'pattern', the accompaniment pattern name to use.")
    parser.add_argument("--left_program_index", type=int, default=32,
                        help="MIDI program index for the left-hand instrument (e.g., 32 for Acoustic Bass).")
    parser.add_argument("--right_program_index", type=int, default=0,
                        help="MIDI program index for the right-hand instrument (e.g., 0 for Acoustic Grand Piano).")
    parser.add_argument("--out_midi", type=str, default="final_out.mid", help="Output MIDI file path.")
    parser.add_argument("--out_ly", type=str, default="final_out.ly", help="Output LilyPond file path.")
    parser.add_argument("--out_pdf_dir", type=str, default="outputs", help="Output directory for the PDF file.")
    
    args = parser.parse_args()
    img_name = args.img_path.split('/')[-1].split('.')[0]
    
    generate_music(
        img_path=args.img_path,
        length=args.length,
        out_midi=os.path.join(args.out_pdf_dir, f"{img_name}.mid"),
        out_ly=os.path.join(args.out_pdf_dir, f"{img_name}.ly"),
        out_pdf_dir=args.out_pdf_dir,
        method=args.method,
        pattern_name=args.pattern_name,
        left_program_index=args.left_program_index,
        right_program_index=args.right_program_index
    )
    
    print("[DONE]")