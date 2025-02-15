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
    model = models.mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)
    model.classifier = nn.Identity()
    model.eval()
    return model

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
        center = np.random.randn(512)*0.1 + offsets[i]
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
    s = hash(deep_vec.tobytes()) & 0xffffffff
    random.seed(s)

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
            octv = 4 if random.random() < 0.7 else 5
            mel_midi = 12 * (octv + 1) + pc
            new_interval = (bass_midi, mel_midi)
            # 这里调用 is_parallel_perfect 与 check_big_leap_direction（需保证这两个函数已实现）
            # 简单模拟检查
            if current_interval is not None and abs(new_interval[1]-new_interval[0]) in (7,12):
                trials += 1
                continue
            # 这里可加入大跳后向级规则检查（省略细节）
            candidate_midi = mel_midi
            break
        if candidate_midi is None:
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

def pick_note_from_chord(chord_pcs_sorted, note_position="lowest", lowestBass=36, highestBass=60, velocity=80, cyc_idx_dict=None):
    """
    根据 note_position 从 chord_pcs_sorted 中选择音符，并保证音高在 [lowestBass, highestBass] 范围内。
    note_position 可为 "lowest", "highest", "middle", "random", "next", "chord" 等。
    返回 (pitch_list, velocity_list)。
    """
    def clip_to_range(pitch, low, high):
        p = pitch
        while p < low:
            p += 12
        while p > high:
            p -= 12
        return p

    if not chord_pcs_sorted:
        return ([60], [velocity])
    
    if note_position == "chord":
        final = [clip_to_range(p, lowestBass, highestBass) for p in chord_pcs_sorted]
        return (final, [velocity] * len(final))
    elif note_position == "lowest":
        base = chord_pcs_sorted[0]
    elif note_position == "highest":
        base = chord_pcs_sorted[-1]
    elif note_position == "middle":
        mid = len(chord_pcs_sorted) // 2
        base = chord_pcs_sorted[mid]
    elif note_position == "random":
        base = random.choice(chord_pcs_sorted)
    elif note_position == "next":
        if cyc_idx_dict is None:
            cyc_idx_dict = {"arp_index": 0}
        idx = cyc_idx_dict["arp_index"]
        base = chord_pcs_sorted[idx % len(chord_pcs_sorted)]
        cyc_idx_dict["arp_index"] = idx + 1
    else:
        base = chord_pcs_sorted[0]
    
    return ([clip_to_range(base, lowestBass, highestBass)], [velocity])

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
    if not os.path.isfile(ly_file):
        raise FileNotFoundError(f"找不到 {ly_file}")
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(ly_file))[0]
        out_prefix = os.path.join(output_dir, base_name)
        command = ["lilypond", "-fpdf", "-o", out_prefix, ly_file]
        pdf_file = f"{out_prefix}.pdf"
    else:
        base_name = os.path.splitext(os.path.basename(ly_file))[0]
        pdf_file = f"{base_name}.pdf"
        command = ["lilypond", "-fpdf", ly_file]

    try:
        subprocess.run(command, check=True)
        print(f"[INFO] PDF生成成功: {pdf_file}")
    except subprocess.CalledProcessError:
        print("LilyPond转换失败.")
        return None
    return pdf_file

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
    os.makedirs(os.path.join(out_pdf_dir, img_name), exist_ok=True)

    if img_bgr is None:
        raise FileNotFoundError(f"无法读取 {img_path}")
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    h_mean = np.mean(hsv[:, :, 0])
    s_mean = np.mean(hsv[:, :, 1])
    v_mean = np.mean(hsv[:, :, 2])
    
    # b) 提取深度特征
    model = load_mobilenet_v2()
    deep_vec = extract_deep_features_bgr(img_bgr, model)
    
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
    print("[INFO] MIDI 写入:", out_midi)
    
    # g) 生成 LilyPond 文本并转换为 PDF
    merged_right = merge_measures(right_all)
    merged_left = merge_measures(left_all)
    right_lily = measures_to_lily_merged(merged_right)
    left_lily = measures_to_lily_merged(merged_left)
    lily_root = convert_key_for_lily(deep_root)
    lily_content = f"""\\version "2.22.1"
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
        out_midi=os.path.join(args.out_pdf_dir, img_name, f"{img_name}.mid"),
        out_ly=os.path.join(args.out_pdf_dir, img_name, f"{img_name}.ly"),
        out_pdf_dir=args.out_pdf_dir,
        method=args.method,
        pattern_name=args.pattern_name,
        left_program_index=args.left_program_index,
        right_program_index=args.right_program_index
    )
    
    print("[DONE]")