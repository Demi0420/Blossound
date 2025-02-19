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
    # print(f"deep_vec: mean={np.mean(deep_vec):.4f}, std={np.std(deep_vec):.4f}")


    all_keys = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"]
    cluster_centers = {}
    offsets = np.linspace(-0.5, 0.5, len(all_keys))
    for i, k in enumerate(all_keys):
        center = np.random.randn(1280) * 0.1 + offsets[i]
        cluster_centers[k] = center
        # print(f"{k}: mean={np.mean(cluster_centers[k]):.4f}, std={np.std(cluster_centers[k]):.4f}")

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

ALL_MAJOR_KEYS_CHORDS = {
    "C":  {"I": [0, 4, 7], "ii": [2, 5, 9], "iii": [4, 7, 11], "IV": [5, 9, 0], "V": [7, 11, 2], "vi": [9, 0, 4], "vii°": [11, 2, 5]},
    "C#": {"I": [1, 5, 8], "ii": [3, 6, 10], "iii": [5, 8, 0], "IV": [6, 10, 1], "V": [8, 0, 3], "vi": [10, 1, 5], "vii°": [0, 3, 6]},
    "D":  {"I": [2, 6, 9], "ii": [4, 7, 11], "iii": [6, 9, 1], "IV": [7, 11, 2], "V": [9, 1, 4], "vi": [11, 2, 6], "vii°": [1, 4, 7]},
    "D#": {"I": [3, 7, 10], "ii": [5, 8, 0], "iii": [7, 10, 2], "IV": [8, 0, 3], "V": [10, 2, 5], "vi": [0, 3, 7], "vii°": [2, 5, 8]},
    "E":  {"I": [4, 8, 11], "ii": [6, 9, 1], "iii": [8, 11, 3], "IV": [9, 1, 4], "V": [11, 3, 6], "vi": [1, 4, 8], "vii°": [3, 6, 9]},
    "F":  {"I": [5, 9, 0], "ii": [7, 10, 2], "iii": [9, 0, 4], "IV": [10, 2, 5], "V": [0, 4, 7], "vi": [2, 5, 9], "vii°": [4, 7, 10]},
    "F#": {"I": [6, 10, 1], "ii": [8, 11, 3], "iii": [10, 1, 5], "IV": [11, 3, 6], "V": [1, 5, 8], "vi": [3, 6, 10], "vii°": [5, 8, 11]},
    "G":  {"I": [7, 11, 2], "ii": [9, 0, 4], "iii": [11, 2, 6], "IV": [0, 4, 7], "V": [2, 6, 9], "vi": [4, 7, 11], "vii°": [6, 9, 0]},
    "G#": {"I": [8, 0, 3], "ii": [10, 1, 5], "iii": [0, 3, 7], "IV": [1, 5, 8], "V": [3, 7, 10], "vi": [5, 8, 0], "vii°": [7, 10, 1]},
    "A":  {"I": [9, 1, 4], "ii": [11, 2, 6], "iii": [1, 4, 8], "IV": [2, 6, 9], "V": [4, 8, 11], "vi": [6, 9, 1], "vii°": [8, 11, 2]},
    "Bb": {"I": [10, 2, 5], "ii": [0, 3, 7], "iii": [2, 5, 9], "IV": [3, 7, 10], "V": [5, 9, 0], "vi": [7, 10, 2], "vii°": [9, 0, 3]},
    "B":  {"I": [11, 3, 6], "ii": [1, 4, 8], "iii": [3, 6, 10], "IV": [4, 8, 11], "V": [6, 10, 1], "vi": [8, 11, 3], "vii°": [10, 1, 4]},
}

ALL_MINOR_KEYS_CHORDS = {
    "A":  {"i": [9, 0, 4], "ii°": [11, 2, 5], "III": [0, 4, 7], "iv": [2, 5, 9], "v": [4, 7, 11], "VI": [5, 9, 0], "VII": [7, 11, 2]},
    "A#": {"i": [10, 1, 5], "ii°": [0, 3, 6], "III": [1, 5, 8], "iv": [3, 6, 10], "v": [5, 8, 0], "VI": [6, 10, 1], "VII": [8, 0, 3]},
    "B":  {"i": [11, 2, 6], "ii°": [1, 4, 8], "III": [2, 6, 9], "iv": [4, 8, 11], "v": [6, 9, 1], "VI": [8, 11, 2], "VII": [10, 1, 4]},
    "C":  {"i": [0, 3, 7], "ii°": [2, 5, 8], "III": [3, 7, 10], "iv": [5, 8, 0], "v": [7, 10, 2], "VI": [8, 0, 3], "VII": [10, 2, 5]},
    "C#": {"i": [1, 4, 8], "ii°": [3, 6, 9], "III": [4, 8, 11], "iv": [6, 9, 1], "v": [8, 11, 3], "VI": [9, 1, 4], "VII": [11, 3, 6]},
    "D":  {"i": [2, 5, 9], "ii°": [4, 7, 10], "III": [5, 9, 0], "iv": [7, 10, 2], "v": [9, 0, 4], "VI": [10, 2, 5], "VII": [0, 4, 7]},
    "D#": {"i": [3, 6, 10], "ii°": [5, 8, 11], "III": [6, 10, 1], "iv": [8, 11, 3], "v": [10, 1, 5], "VI": [11, 3, 6], "VII": [1, 5, 8]},
    "E":  {"i": [4, 7, 11], "ii°": [6, 9, 0], "III": [7, 11, 2], "iv": [9, 0, 4], "v": [11, 2, 6], "VI": [0, 4, 7], "VII": [2, 6, 9]},
    "F":  {"i": [5, 8, 0], "ii°": [7, 10, 1], "III": [8, 0, 3], "iv": [10, 1, 5], "v": [0, 3, 7], "VI": [1, 5, 8], "VII": [3, 7, 10]},
    "F#": {"i": [6, 9, 1], "ii°": [8, 11, 2], "III": [9, 1, 4], "iv": [11, 2, 6], "v": [1, 4, 8], "VI": [2, 6, 9], "VII": [4, 8, 11]},
    "G":  {"i": [7, 10, 2], "ii°": [9, 0, 3], "III": [10, 2, 5], "iv": [0, 3, 7], "v": [2, 5, 9], "VI": [3, 7, 10], "VII": [5, 9, 0]},
    "G#": {"i": [8, 11, 3], "ii°": [10, 1, 4], "III": [11, 3, 6], "iv": [1, 4, 8], "v": [3, 6, 10], "VI": [4, 8, 11], "VII": [6, 10, 1]},
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
            chord_map = ALL_MAJOR_KEYS_CHORDS
        else:
            chord_map = ALL_MINOR_KEYS_CHORDS
        
        if root_note in chord_map and chord_label in chord_map[root_note]:
            return chord_map[root_note][chord_label]
        else:
            return []  # 返回空列表，避免错误

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


def generate_dual_voice_measure_old(chord_pcs,
                                    chord_label,
                                    time_signature=(4,4), 
                                    last_interval=None, 
                                    hist_notes=(None, None), 
                                    max_jump=7, 
                                    step_threshold=2):
    """
    time_signature: 一个元组 (numerator, denominator)，例如 (4,4)、(3,4)、(6,8)、(9,8) 等。
                      注意：对于复合拍（denominator==8 且 numerator 为6、9、12），我们按每组3个八分音符作为一拍来处理，
                      即 (6,8)→2 拍，(9,8)→3 拍，(12,8)→4 拍。
    确保 **生成的音符时值总和符合节拍**，并遵循对位规则：
      - 避免平行五度/八度；
      - 大跳后接向级进行（要求反向且步幅不超过 2）。
    """
    """ 优化和弦进行，使用二次属和弦增强吸引力 """
    # 1️⃣ **尝试加入二次属和弦**
    # sec_dom_pcs = get_secondary_dominant(chord_label)
    # if sec_dom_pcs and random.random() <= 1.0:  # 30% 概率使用二次属和弦
    #     chord_pcs = sec_dom_pcs  
    num, den = time_signature
    if den == 8 and num in (6, 9, 12):
        beats_per_measure = num // 3  # 6/8 → 2 拍, 9/8 → 3 拍, 12/8 → 4 拍
        beat_unit_duration = 3/8  # 每拍 3 个八分音符
    else:
        beats_per_measure = num  # 例如 3/4 → 3, 4/4 → 4
        beat_unit_duration = 1/4  # 每拍 2 个八分音符

    expected_notes_per_measure = num  # **确保生成 num 个音符（即 9/8 → 9 个音符）**
    right_list = []
    left_list = []
    second_last_r, last_r = hist_notes

    # **左声部采用固定低音**
    bass_pc = chord_pcs[0]
    bass_midi = 12 * (3 + 1) + bass_pc  # octave = 3

    current_interval = last_interval
    current_second_last = second_last_r
    current_last_r = last_r

    for _ in range(expected_notes_per_measure):  # ✅ **确保生成 9 个音符**
        trials = 0
        candidate_midi = None
        while trials < 30:
            pc = random.choice(chord_pcs)
            octv = 4 if random.random() <= 1.0 else 5
            mel_midi = 12 * (octv + 1) + pc
            new_interval = (bass_midi, mel_midi)

            # ✅ **避免平行五度/八度**
            if (current_interval is not None and last_interval is not None and
                abs(new_interval[1] - new_interval[0]) in (7,12) and
                abs(last_interval[1] - last_interval[0]) in (7,12)):
                trials += 1
                continue

            # ✅ **大跳后向级进行（反向步幅≤2）**
            if current_second_last is not None and abs(current_last_r - current_second_last) > step_threshold:
                if (mel_midi - current_last_r) * (current_last_r - current_second_last) > 0:
                    trials += 1
                    continue

            candidate_midi = mel_midi
            break

        if candidate_midi is None:
            candidate_midi = bass_midi + 12  # 备用音符

        right_list.append(candidate_midi)
        left_list.append(bass_midi)

        current_second_last = current_last_r
        current_last_r = candidate_midi
        current_interval = (bass_midi, candidate_midi)

    return right_list, left_list, current_interval, (current_second_last, current_last_r)

def get_secondary_dominant(chord_label, root_note, scale_mode="major"):
    """
    根据当前和弦，返回可能的二次属和弦的音程列表，确保符合给定的调性。
    """
    if scale_mode == "major":
        key_chords = ALL_MAJOR_KEYS_CHORDS.get(root_note, {})
    else:
        key_chords = ALL_MINOR_KEYS_CHORDS.get(root_note, {})

    # 目标和弦的根音
    target_chord = key_chords.get(chord_label, None)
    if target_chord is None:
        return None  # 没有匹配的和弦
    
    # 计算 V7/目标和弦
    target_root_pc = target_chord[0]  # 目标和弦的根音
    dominant_root_pc = (target_root_pc + 7) % 12  # 属七和弦的根音（V7）

    # 生成完整的 V7 和弦
    secondary_dominant = [
        dominant_root_pc,  # 根音
        (dominant_root_pc + 4) % 12,  # 大三度
        (dominant_root_pc + 7) % 12,  # 完全五度
        (dominant_root_pc + 10) % 12  # 小七度
    ]

    return secondary_dominant

def generate_rhythm_pattern(time_signature):
    """
    生成符合 time_signature 的节奏模式，确保总和等于 time_signature[0]
    """
    num, den = time_signature
    total_duration = num  # 总时长应该等于 num
    
    if den == 8 and num in (6, 9, 12):  
        base_unit = 3  # 复合拍 (6/8, 9/8, 12/8)
    else:
        base_unit = 1  # 单拍 (4/4, 3/4, 5/4)
    
    # 确保节奏模式总时长等于 time_signature[0]
    while True:
        rhythm_pattern = []
        remaining_duration = total_duration
        
        while remaining_duration > 0:
            if remaining_duration <= base_unit:
                rhythm_pattern.append(remaining_duration)
                break
            max_note = min(base_unit + 1, remaining_duration)  # 限制最大值
            note_duration = random.randint(1, max_note)
            rhythm_pattern.append(note_duration)
            remaining_duration -= note_duration
        
        if sum(rhythm_pattern) == total_duration:
            return rhythm_pattern

def generate_dual_voice_measure(chord_pcs, 
                                chord_label, 
                                root_note,
                                scale_mode="major",
                                time_signature=(4, 4), 
                                last_interval=None, 
                                hist_notes=(None, None), 
                                max_jump=7, 
                                step_threshold=2):
    """
    Generates an optimized melody using secondary dominants, passing tones, rhythmic patterns, and directional control.
    Ensures the melody follows the given time signature.
    """
    # Attempt to add a secondary dominant chord
    # ✅ 1️⃣ 确保二次属和弦符合当前调性
    sec_dom_pcs = get_secondary_dominant(chord_label, root_note, scale_mode)
    if sec_dom_pcs and random.random() < 0.0:
        valid_notes = set(sum(ALL_MAJOR_KEYS_CHORDS.get(root_note, {}).values(), []))
        # 过滤掉不属于当前调性的二次属和弦音符
        chord_pcs = [pc for pc in sec_dom_pcs if pc in valid_notes]

    # ✅ 2️⃣ 确保 scale_notes 包含当前调性的所有和弦音符
    if scale_mode == "major":
        valid_notes = set(sum(ALL_MAJOR_KEYS_CHORDS[root_note].values(), []))  # 获取所有B大调的音符
    else:
        valid_notes = set(sum(ALL_MINOR_KEYS_CHORDS[root_note].values(), []))  # 获取所有B小调的音符

    scale_notes = [pc for pc in chord_pcs if pc in valid_notes]

    print("[DEBUG] scale_notes: ", scale_notes)

    # Generate rhythm pattern dynamically
    rhythm_pattern = generate_rhythm_pattern(time_signature)
    print("[DEBUG] rhythm_pattern: ", rhythm_pattern)
    expected_notes_per_measure = sum(rhythm_pattern)

    # Initialize melody lists
    right_hand = []  # Melody (higher voice)
    left_hand = []   # Bassline (lower voice)
    second_last_r, last_r = hist_notes
    bass_pc = chord_pcs[0]
    bass_midi = 48 + bass_pc  # Keep bass stable
    
    current_interval = last_interval
    current_second_last = second_last_r
    current_last_r = last_r

    for duration in rhythm_pattern:  # 确保每个音符匹配节奏模式
        trials = 0
        candidate_midi = None
        
        while trials < 30:
            if current_last_r and random.random() < 0.7:
                stepwise_choices = [(current_last_r + 2) % 12, (current_last_r - 2) % 12]
                stepwise_choices = [n for n in stepwise_choices if n in scale_notes]
                candidate_pc = random.choice(stepwise_choices) if stepwise_choices else random.choice(scale_notes)
            else:
                candidate_pc = random.choice(scale_notes)

            candidate_midi = 60 + candidate_pc  # Map pitch class to MIDI note
            new_interval = (bass_midi, candidate_midi)

            # Avoid parallel fifths and octaves
            if current_interval and last_interval:
                if abs(new_interval[1] - new_interval[0]) in (7, 12) and abs(last_interval[1] - last_interval[0]) in (7, 12):
                    trials += 1
                    continue  # Try again if parallel 5ths/8ves detected
            
            break  
        
        if candidate_midi is None:
            candidate_midi = bass_midi + 12  # Default fallback
        
        # 根据节奏模式的数值分配音符
        for _ in range(duration):
            right_hand.append(candidate_midi)
            left_hand.append(bass_midi)
        
        # Update previous note references
        current_second_last = current_last_r
        current_last_r = candidate_midi
        current_interval = (bass_midi, candidate_midi)
    
    return right_hand, left_hand, current_interval, (current_second_last, current_last_r)

#########################################################
# E. 伴奏模式相关函数
#########################################################

ACCOMPANIMENT_PATTERNS_4_4 = {
    # 流行 (Pop)
    # 常采用简洁的扫弦伴奏：第一拍以低音强调，后面用明亮的和弦扫出
    "pop_4_4": [
        (0.0, "bass"),    # 第1拍：强调低音或根音
        (1.0, "chord"),   # 第2拍：简单扫弦和弦
        (2.0, "chord"),   # 第3拍：保持和弦连贯
        (3.0, "chord"),   # 第4拍：轻扫和弦，形成流畅律动
    ],

    # 摇滚 (Rock)
    # 摇滚伴奏常强调2和4拍（反拍），通常采用强劲的低音与失真吉他和弦
    "rock_4_4": [
        (0.0, "bass"),    # 第1拍：重低音或鼓点，稳固节奏
        (1.0, "chord"),   # 第2拍：用失真或较强的和弦强调反拍
        (2.0, "bass"),    # 第3拍：再次突出低音或打击乐效果
        (3.0, "chord"),   # 第4拍：反拍继续，形成明显的节奏律动
    ],

    # 电子舞曲 (EDM)
    # 典型的“四四拍”伴奏，即所谓的“four on the floor”，每拍均有低音鼓，并辅以军鼓或高帽打击
    "edm_4_4": [
        (0.0, "kick"),       # 第1拍：低音鼓强击
        (1.0, "snare/hihat"),# 第2拍：军鼓或高帽声，制造律动感
        (2.0, "kick"),       # 第3拍：低音鼓继续，确保律动均衡
        (3.0, "snare/hihat"),# 第4拍：重复第二拍的打击，形成持续驱动效果
    ],

    # 爵士 (Jazz)
    # 爵士伴奏常采用摇摆感的分解和弦（comping），重音灵活，注重和声与律动的交互
    "jazz_4_4": [
        (0.0, "bass"),     # 第1拍：低音根音，稳固节奏
        (1.0, "comping"),  # 第2拍：轻扫或分解和弦，带有跳跃感
        (2.0, "comping"),  # 第3拍：较强的和弦补充，可加装饰音
        (3.0, "comping"),  # 第4拍：延音或扫弦，形成连贯的摇摆感
    ],
}


ACCOMPANIMENT_PATTERNS_3_4 = {
    # 华尔兹 (Waltz)
    # 典型的“oom–pah–pah”：第一拍为强音（低音或根音），第二、第三拍用和弦扫弦，整体营造出轻盈流畅的圆舞节奏
    "waltz_3_4": [
        (0.0, "bass"),   # 强拍：突出低音（参考传统华尔兹重拍设定  [oai_citation_attribution:0‡360doc.com](https://www.360doc.com/content/21/0125/22/73531903_958925601.shtml)）
        (1.0, "chord"),  # 弱拍：扫弦和弦
        (2.0, "chord"),  # 弱拍：扫弦和弦
    ],
    
    # 浪漫古典音乐 (Romantic Classical)
    # 这种风格常采用较流畅、富有连贯性的分解和弦（琶音）伴奏，
    # 第一拍突出低音，后两拍可以用琶音展开和弦，营造出柔美且富有延音的效果
    "romantic_3_4": [
        (0.0, "bass"),      # 强拍：清晰的低音根音
        (1.0, "arpeggio"),  # 中间拍：琶音式分解和弦
        (2.0, "sustain"),   # 末拍：和弦延音，增加柔和感
    ],
    
    # 抒情民谣 (Lyrical Folk)
    # 民谣伴奏通常较为简单，采用下-上扫弦型，第一拍强调后两拍轻柔扫弦，
    # 既保持旋律的流畅又不抢主旋律的风头
    "folk_3_4": [
        (0.0, "bass"),         # 强拍：强调低音或根音
        (1.0, "light_strum"),  # 第二拍：轻柔扫弦
        (2.0, "light_strum"),  # 第三拍：轻柔扫弦
    ],
}


ACCOMPANIMENT_PATTERNS_2_4 = {
    # 1) 进行曲 (March)
    # 进行曲通常采用简单直接的“强－弱”模式，
    # 第一拍以低音或大鼓突出重音，第二拍用军鼓或轻击补充，
    # 形成稳健、整齐的进行节奏。
    "march_2_4": [
        (0.0, "bass"),   # 第一拍：重低音（或大鼓），突出下拍
        (1.0, "snare"),  # 第二拍：用军鼓或轻击强调上拍
    ],

    # 2) 探戈 (Tango)
    # 探戈风格的2/4伴奏通常具有戏剧性和短促感，
    # 第一拍以强调和弦（或低音）为主，第二拍则采用断音或弱化处理，
    # 形成强烈的对比和紧凑节奏。
    "tango_2_4": [
        (0.0, "accented_chord"),  # 第一拍：重音和弦，短促有力
        (1.0, "muted_chord"),     # 第二拍：断音或轻扫，营造出探戈特有的冷峻感
    ],

    # 3) 乡村舞曲 (Country Dance)
    # 乡村舞曲的伴奏风格较为简洁明快，
    # 第一拍用清晰的低音强调节奏，第二拍则以轻柔扫弦衬托出“走路式”的律动，
    # 使整体感觉轻松自然。
    "country_2_4": [
        (0.0, "bass"),          # 第一拍：强烈低音或根音
        (1.0, "light_strum"),   # 第二拍：柔和扫弦，保持律动流畅
    ],
}


ACCOMPANIMENT_PATTERNS_6_8 = {
    # 1) 布鲁斯 (Blues)
    # 在6/8拍中，布鲁斯常以两大拍为基础（下拍在0和3），并在中间加入略带摇摆的“切分”律动。
    # 这里设置在第1拍和第4拍用低音（bass）强调，下拍附近（第3个八分音符的位置）加入和弦（chord）。
    "blues_6_8": [
        (0.0, "bass"),    # 第一大拍下：强烈低音
        (2.0, "chord"),   # 第一大拍末：带有轻微切分的和弦
        (3.0, "bass"),    # 第二大拍下：强烈低音
        (5.0, "chord"),   # 第二大拍末：切分和弦，为下一小节做衔接
    ],
    
    # 2) 凯尔特音乐 (Celtic)
    # 凯尔特风格（如爱尔兰jig）常具有明快而富有装饰性的节奏，
    # 可采用较多细节事件：在每大拍内除下拍（bass）外，加上轻扫和装饰性的拨弦（pluck）。
    "celtic_6_8": [
        (0.0, "bass"),       # 第一大拍下：稳固低音
        (1.0, "chord"),      # 第一大拍内：简单和弦扫弦
        (2.0, "pluck"),      # 第一大拍末：轻柔拨弦，增添装饰
        (3.0, "bass"),       # 第二大拍下：稳固低音
        (4.0, "chord"),      # 第二大拍内：和弦扫弦
        (5.0, "pluck"),      # 第二大拍末：拨弦装饰，形成连贯律动
    ],
    
    # 3) 慢摇滚 (Slow Rock)
    # 慢摇滚6/8拍伴奏通常给人一种沉稳、稍带摇摆的感觉，
    # 强调每大拍的下拍（0与3），并在后拍加入较为饱满的和弦衬托。
    "slow_rock_6_8": [
        (0.0, "bass"),    # 第一大拍下：强调低音
        (2.0, "chord"),   # 第一大拍末：填充和弦
        (3.0, "bass"),    # 第二大拍下：低音重复
        (5.0, "chord"),   # 第二大拍末：和弦衬托
    ],
    
    # 4) 古典 (Classical) – 某些古典音乐中6/8伴奏
    # 这里常见的处理是将6/8看作两拍，每拍为一个点（dotted quarter），
    # 并采用流畅的琶音式伴奏。为了更细腻地表达“连绵”的感觉，
    # 可在每大拍内用一个延音型的琶音替代单一扫弦。
    # 此处采用分组的方式：在第一大拍，用低音在0拍，再在1.5拍处补充琶音；
    # 第二大拍同理，在3.0拍和4.5拍处进行。
    "classical_6_8": [
        (0.0, "bass"),         # 第一大拍下：根音
        (1.5, "arpeggio"),     # 第一大拍中：琶音展开
        (3.0, "bass"),         # 第二大拍下：根音
        (4.5, "arpeggio"),     # 第二大拍中：琶音补充
    ],
}


ACCOMPANIMENT_PATTERNS_5_4 = {
    # 1) 前卫摇滚 (Progressive Rock)
    # 采用5/4拍时，前卫摇滚常利用不对称的重音分布和一定的切分，
    # 例如：第一拍用强低音定下基调，接着在第二、第三拍间穿插和弦扫弦，
    # 第四拍再用一次较强的打击，最后一拍做缓冲或转接。
    "prog_5_4": [
        (0.0, "bass"),    # 第一拍：强烈低音，确定基调
        (1.0, "chord"),   # 第二拍：扫出和弦，营造律动
        (2.0, "drum"),    # 第三拍：打击乐切分，增加不对称感
        (3.0, "chord"),   # 第四拍：再次扫弦，带出旋律线
        (4.0, "accent"),  # 第五拍：强调式打击，为下小节做衔接
    ],
    
    # 2) 实验音乐 (Experimental)
    # 实验音乐往往追求意想不到的律动和音色组合，
    # 在5/4拍中可以采用部分非整数拍点来制造偶发的悬疑感，
    # 如在第2拍后加入一个 2.5 拍处的抽象音效，形成独特的节奏张力。
    "experimental_5_4": [
        (0.0, "drone"),      # 第一拍：持续的背景音（drone），建立音场
        (1.0, "percussion"), # 第二拍：不规则打击音
        (2.5, "glitch"),     # 同一小节中插入一个切分音（非整数拍），制造突兀感
        (3.0, "ambient"),    # 第四拍：柔和的环境音
        (4.0, "percussion"), # 第五拍：再次打击，为循环铺垫
    ],
    
    # 3) 电影配乐 (Film Score) —— 以《碟中谍》主题曲为例
    # 该主题曲的5/4拍伴奏充满紧张感与推进力，
    # 可采用重低音与精确的打击乐配合，突出节奏的悬疑与力度，
    # 例如：第一拍以低音定基调，第三拍作为过渡中突然的强调，第五拍再次引入和弦铺垫下小节。
    "film_5_4": [
        (0.0, "bass"),       # 第一拍：强烈低音，开篇定调
        (1.0, "chord"),      # 第二拍：简洁和弦，平滑过渡
        (2.0, "chord"),      # 第三拍：略带悬疑的和弦延续
        (3.0, "accent"),     # 第四拍：突出打击乐的重音，制造紧张感
        (4.0, "chord"),      # 第五拍：缓冲并引出下小节
    ],
}

ACCOMPANIMENT_PATTERNS_7_8 = {
    # 1) 前卫摇滚 (Progressive Rock)
    # 采用 3+2+2 分组：
    # – 第1组（第1～3拍）：第一拍用强低音定基调，接着用和弦填充；
    # – 第2组（第4～5拍）：在第4拍用打击乐或重音标记，再用和弦过渡；
    # – 第3组（第6～7拍）：再次用低音衔接，并用和弦结束，为下小节做铺垫。
    "prog_7_8": [
        (0.0, "bass"),    # 第1拍：强烈低音，确定开头 
        (1.0, "chord"),   # 第2拍：中等力度扫弦
        (2.0, "chord"),   # 第3拍：延续和弦填充
        (3.0, "accent"),  # 第4拍：重音打击，开启第二组 
        (4.0, "drum"),    # 第5拍：打击乐或轻扫，衔接第二组
        (5.0, "bass"),    # 第6拍：第三组起始，再次用低音带动
        (6.0, "chord"),   # 第7拍：用和弦收尾，为下小节过渡
    ],
    
    # 2) 巴尔干音乐 (Balkan)
    # 常见分组为 2+2+3：
    # – 第1组（第1～2拍）：以重低音开头，紧接轻扫和弦；
    # – 第2组（第3～4拍）：重复低音与和弦的对位；
    # – 第3组（第5～7拍）：在第三组中突出一个明显的重音（第5拍），后续用和弦及填充结束。
    "balkan_7_8": [
        (0.0, "bass"),    # 第1拍：强烈低音  [oai_citation_attribution:0‡blog.sina.com.cn](https://blog.sina.com.cn/s/blog_17c8fdd330102z9zg.html)
        (1.0, "chord"),   # 第2拍：简单和弦衔接
        (2.0, "bass"),    # 第3拍：第二组起始，再次强调低音
        (3.0, "chord"),   # 第4拍：继续扫弦
        (4.0, "accent"),  # 第5拍：第三组开始，用明显重音标记  [oai_citation_attribution:1‡blog.sina.com.cn](https://blog.sina.com.cn/s/blog_17c8fdd330102z9zg.html)
        (5.0, "chord"),   # 第6拍：和弦扫弦
        (6.0, "fill"),    # 第7拍：短暂填充，为下小节做过渡
    ],
    
    # 3) 实验性爵士 (Experimental Jazz)
    # 采用较自由的处理方式，在 7/8 拍内设置不规则音型：
    # – 第一拍用爵士式“comping”建立和声背景，
    # – 第二拍加入柔和的环境音色，
    # – 第三拍用打击乐突出节奏，
    # – 第四拍留白（space）制造悬念，
    # – 第五拍用短促的 staccato 和弦，
    # – 第六拍加入额外打击（percussion），
    # – 第七拍以即兴音效收尾，营造实验氛围  [oai_citation_attribution:2‡blog.sina.com.cn](https://blog.sina.com.cn/s/blog_56b5a3f50100oefv.html)
    "expjazz_7_8": [
        (0.0, "comping"),    # 第1拍：爵士式和弦伴奏
        (1.0, "ambient"),    # 第2拍：柔和环境音色
        (2.0, "drum"),       # 第3拍：突出打击乐效果
        (3.0, "space"),      # 第4拍：留白，制造悬疑感  [oai_citation_attribution:3‡blog.sina.com.cn](https://blog.sina.com.cn/s/blog_56b5a3f50100oefv.html)
        (4.0, "staccato"),   # 第5拍：短促扫弦，突显节奏
        (5.0, "percussion"), # 第6拍：加入额外打击音，增加变化
        (6.0, "improv"),     # 第7拍：即兴音效收尾，营造实验性氛围  [oai_citation_attribution:4‡blog.sina.com.cn](https://blog.sina.com.cn/s/blog_56b5a3f50100oefv.html)
    ],
}



ACCOMPANIMENT_PATTERNS_9_8 = {
    # 传统舞曲
    # (1) 爱尔兰（如 Slip Jig） – 3+3+3 分组：
    "irish_9_8": [
        (0.0, "bass"),    # 拍1：强烈低音，建立律动  [oai_citation_attribution:0‡tup.tsinghua.edu.cn](https://www.tup.tsinghua.edu.cn/upload/books/yz/078683-01.pdf)
        (1.0, "chord"),   # 拍2：轻扫和弦
        (2.0, "chord"),   # 拍3：延续和弦
        (3.0, "bass"),    # 拍4：新一组开头，再次强调低音
        (4.0, "chord"),   # 拍5：和弦填充
        (5.0, "chord"),   # 拍6：和弦延续
        (6.0, "bass"),    # 拍7：第三组起始，低音突显
        (7.0, "chord"),   # 拍8：和弦
        (8.0, "chord"),   # 拍9：和弦收尾
    ],

    # (2) 希腊传统舞曲 – 假设采用 2+2+2+3 分组：
    "greek_9_8": [
        (0.0, "bass"),    # 拍1：重音低音
        (1.0, "chord"),   # 拍2：简洁和弦
        (2.0, "bass"),    # 拍3：新组开头，重音
        (3.0, "chord"),   # 拍4：和弦填充
        (4.0, "bass"),    # 拍5：第三组起始（2拍组）
        (5.0, "chord"),   # 拍6：和弦衔接
        (6.0, "bass"),    # 拍7：最后一组开头，突出重音  [oai_citation_attribution:1‡blog.sina.com.cn](https://blog.sina.com.cn/s/blog_17c8fdd330102z9zg.html)
        (7.0, "chord"),   # 拍8：和弦填充
        (8.0, "chord"),   # 拍9：和弦或轻微装饰，为下小节过渡
    ],

    # (3) 土耳其传统舞曲 – 同样常用 2+2+2+3 分组，但在最后一组加入特色装饰：
    "turkish_9_8": [
        (0.0, "bass"),     # 拍1：重音低音
        (1.0, "chord"),    # 拍2：和弦
        (2.0, "bass"),     # 拍3：重音
        (3.0, "chord"),    # 拍4：和弦
        (4.0, "bass"),     # 拍5：重音
        (5.0, "chord"),    # 拍6：和弦衔接
        (6.0, "accent"),   # 拍7：最后一组开头，用更强烈的重音强调  [oai_citation_attribution:2‡blog.sina.com.cn](https://blog.sina.com.cn/s/blog_17c8fdd330102z9zg.html)
        (7.0, "chord"),    # 拍8：和弦填充
        (8.0, "fill"),     # 拍9：加入装饰性填充，典型的土耳其“花样”表现
    ],

    # 巴赫的某些作品 – 采用传统 9/8 复拍子处理，常看作 3 组 3 拍（3+3+3）：
    "bach_9_8": [
        (0.0, "bass"),         # 组1第1拍：强音低音，奠定和声基础  [oai_citation_attribution:3‡tup.tsinghua.edu.cn](https://www.tup.tsinghua.edu.cn/upload/books/yz/078683-01.pdf)
        (1.0, "arpeggio"),     # 组1第2拍：琶音展开
        (2.0, "arpeggio"),     # 组1第3拍：延续琶音
        (3.0, "bass"),         # 组2第1拍：重音低音
        (4.0, "arpeggio"),     # 组2第2拍：琶音和弦
        (5.0, "arpeggio"),     # 组2第3拍：继续琶音
        (6.0, "bass"),         # 组3第1拍：低音强化
        (7.0, "arpeggio"),     # 组3第2拍：琶音补充
        (8.0, "arpeggio"),     # 组3第3拍：琶音收尾，衔接下句
    ],
}


ACCOMPANIMENT_PATTERNS_12_8 = {
    # 1) 节奏布鲁斯 (Rhythm & Blues)
    # 常以温暖而略带摇摆的“boom-chick”感觉呈现。
    # 这里我们在第一拍（0拍）用强低音定基调，在第2拍（3拍处）用和弦扫弦，
    # 在第3拍（6拍处）再以低音补充，并在最后一拍（9拍处）用轻刷或小击突出进入下小节。
    "rnb_12_8": [
        (0.0, "bass"),    # 拍1：强烈低音（点1）
        (3.0, "chord"),   # 拍2：和弦扫弦（点2）
        (6.0, "bass"),    # 拍3：低音补充（点3）
        (9.0, "brush"),   # 拍4：轻刷或小击，过渡至下小节
    ],

    # 2) 慢摇滚 (Slow Rock)
    # 慢摇滚中的12/8拍伴奏通常注重拉长的、饱满的和弦铺垫，
    # 常将每个主要拍（0, 3, 6, 9拍）进行分层处理，偶尔在中间加些同步化填充。
    "slow_rock_12_8": [
        (0.0, "bass"),      # 拍1：低音稳固，定下节奏
        (2.0, "chord"),     # 拍1后半：加入略带装饰的和弦扫弦
        (3.0, "chord"),     # 拍2：中间强拍，用和弦加厚
        (6.0, "bass"),      # 拍3：再次以低音强调
        (7.0, "chord"),     # 拍3后半：轻扫和弦铺垫
        (9.0, "snare"),     # 拍4：适当的打击乐（或鼓点）为下小节做衔接
    ],

    # 3) 蓝调 (Blues)
    # 蓝调的12/8拍伴奏通常采用“摇摆”感十足的分组，
    # 常用类似“bass-chord-snare” 的模式，在每个小节中以低音强调下拍，
    # 并在第2和第4个主要拍之间加入略带“摆动”的和弦填充。
    "blues_12_8": [
        (0.0, "bass"),     # 拍1：强低音，定下摇摆基调
        (3.0, "chord"),    # 拍2：和弦扫弦，带出蓝调的律动
        (5.0, "snare"),    # 拍2后侧：适当的击打，增添摇摆感
        (6.0, "bass"),     # 拍3：再次用低音强调
        (9.0, "chord"),    # 拍4：和弦铺垫
        (11.0, "snare"),   # 拍4后侧：轻击，为下小节做引导
    ],

    # 4) 某些爵士风格 (Certain Jazz Styles)
    # 采用12/8拍时，爵士常以“摇摆comping”方式呈现，
    # 主要在每个点（0, 3, 6, 9拍）留出主要和弦伴奏，并在中间用轻刷或装饰性打击丰富律动，
    # 营造出既稳重又富有即兴感的伴奏效果。
    "jazz_12_8": [
        (0.0, "bass"),        # 拍1：稳固低音
        (1.5, "comping"),     # 第一拍内部：轻刷或分解和弦作为装饰
        (3.0, "comping"),     # 拍2：主要和弦伴奏
        (4.5, "brush"),       # 第二拍内部：轻刷增加连贯性
        (6.0, "bass"),        # 拍3：重复低音铺垫
        (7.5, "comping"),     # 第三拍内部：即兴装饰和弦
        (9.0, "comping"),     # 拍4：主要和弦继续
        (10.5, "brush"),      # 第四拍内部：轻刷作为过渡
    ],
}


ACCOMPANIMENT_PATTERNS_OLD = {
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


    # 4) Simple arpeggio up (4/4), 在每拍演奏 chord 的 不同音
    "arp_up_4_4": [
        (0.0, "lowest"),
        (1.0, "next"),
        (2.0, "next"),
        (3.0, "next"),
    ],

    # 5) Ostinato 16 (4/4), 假设每小节 8次16分(只是示例)
    #   例如 pattern写 [ (0.0, "lowest"), (0.5,"next"), (1.0,"next")... ]
    "ostinato_16": [
        (0.0, "lowest"),
        (0.5, "next"),
        (1.0, "middle"),
        (1.5, "next"),
        (2.0, "lowest"),
        (2.5, "next"),
        (3.0, "middle"),
        (3.5, "next"),
    ]
}


def pick_note_from_chord(chord_pcs_sorted, 
                         root_note, 
                         scale_mode="major",
                         note_position="lowest", 
                         lowestBass=36, 
                         highestBass=60, 
                         velocity=80, 
                         prev_note=None, 
                         max_jump=7, 
                         cyc_idx_dict=None):
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

    # **1️⃣ 获取当前调性内的合法音符**
    if scale_mode == "major":
        valid_scale_notes = set(sum(ALL_MAJOR_KEYS_CHORDS.get(root_note, {}).values(), []))
    else:
        valid_scale_notes = set(sum(ALL_MINOR_KEYS_CHORDS.get(root_note, {}).values(), []))

    # **确保和弦音符属于该调性**
    chord_pcs_sorted = [pc for pc in chord_pcs_sorted if pc in valid_scale_notes]
    if not chord_pcs_sorted:
        root_pc = NOTE_BASE_MAP.get(root_note, 0)  # 获取 root note 的音高类
        chord_pcs_sorted = [root_pc]  # 默认返回主音

    # 固定使用的八度：例如设为 3，即构造音符时用 12*(3+1)=48 作为基础，这样生成的音在 [48, 59] 内
    fixed_octv = 3

    def build_candidate(pc):
        """ 生成 MIDI 音高，并确保音符属于当前调性。"""
        if pc not in valid_scale_notes:
            print(f"Warning: {pc} 不属于 {root_note} {scale_mode} 调，将替换为主音！")
            pc = NOTE_BASE_MAP[root_note]  # 强制转换为主音
        
        octave = (prev_note // 12) if prev_note else fixed_octv + 1
        return (octave * 12) + (pc % 12)

    def get_base_pitch():
        nonlocal cyc_idx_dict
        # 按照 note_position 选择一个和弦内的音（仅返回音名 0~11）
        if not chord_pcs_sorted:
            return 0
        if note_position in ("lowest", "bass", "drum", "kick", "drone", "percussion"):
            return chord_pcs_sorted[0]
        elif note_position in ("highest", "snare", "snare/hihat", "accent", "accented_chord", "pluck", "light_strum"):
            return chord_pcs_sorted[-1]
        elif note_position in ("middle", "brush", "comping", "ambient", "staccato"):
            mid = len(chord_pcs_sorted) // 2
            return chord_pcs_sorted[mid]
        elif note_position in ("muted_chord", "fill"):
            if len(chord_pcs_sorted) >= 3:
                return chord_pcs_sorted[1]
            else:
                mid = len(chord_pcs_sorted) // 2
                return chord_pcs_sorted[mid]
        elif note_position in ("random", "improv", "glitch"):
            return random.choice(chord_pcs_sorted)
        elif note_position in ("next", "arpeggio"):
            if cyc_idx_dict is None:
                cyc_idx_dict = {"arp_index": 0}
            if "arp_index" not in cyc_idx_dict:
                cyc_idx_dict["arp_index"] = 0
            idx = cyc_idx_dict["arp_index"]
            pitch = chord_pcs_sorted[idx % len(chord_pcs_sorted)]
            cyc_idx_dict["arp_index"] = idx + 1
            return pitch
        elif note_position == "sustain":
            if prev_note is not None:
                return prev_note % 12  # 返回前一个音的音名
            else:
                mid = len(chord_pcs_sorted) // 2
                return chord_pcs_sorted[mid]
        elif note_position == "space":
            # 对于 "space"，这里返回中间音
            mid = len(chord_pcs_sorted) // 2
            return chord_pcs_sorted[mid]
        elif note_position == "chord":
            # "chord" 模式返回整个和弦，这里由于函数返回单音，取中间音
            mid = len(chord_pcs_sorted) // 2
            return chord_pcs_sorted[mid]
        else:
            # 默认返回最低音
            return chord_pcs_sorted[0]
    
    # **4️⃣ 生成候选音符**
    candidate = None
    trials = 0
    while trials < 30:
        base_pitch = get_base_pitch()
        cand = build_candidate(base_pitch)
        # 检查是否在允许范围内（虽然固定八度一般都在范围内）
        if cand < lowestBass or cand > highestBass:
            trials += 1
            continue

        # **检查跳跃幅度**
        if prev_note is not None:
            if abs(cand - prev_note) > max_jump:
                trials += 1
                continue

        candidate = cand
        break

    # **5️⃣ 回退策略**
    if candidate is None:
        possible = [build_candidate(pc) for pc in chord_pcs_sorted if lowestBass <= build_candidate(pc) <= highestBass]
        if prev_note is not None and possible:
            candidate = min(possible, key=lambda x: abs(x - prev_note))
        elif possible:
            candidate = possible[0]
        else:
            candidate = build_candidate(chord_pcs_sorted[0])
    
    return ([candidate], [velocity])

def generate_accompaniment(
    chord_pcs_sorted, 
    root_note,
    scale_mode,
    pattern_name,
    start_time=0.0,
    time_signature=(4,4),
    # beats_per_bar=4,
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
      time_signature: 一个元组 (numerator, denominator)，例如 (4,4)、(3,4)、(2,4)、(6,8) 等。
      // beats_per_bar: 每小节拍数
      tempo: BPM
      lowestBass, highestBass: 音域限制
      velocityBase: 基础力度
      cyc_idx_dict: 用于 "next" 模式的循环索引（可选）
    返回:
      note_events: 列表，每个元素为 (on_time, off_time, pitch_list, velocity_list)
    """
    # 根据 time_signature 构造对应的伴奏模式字典名，例如 "ACCOMPANIMENT_PATTERNS_3_4"
    num, den = time_signature
    if den == 8 and num in (6, 9, 12):
        # beats_per_measure = num // 3
        beat_unit_duration = 60.0 / tempo * (3.0 / den)  # 每“拍”取3个八分音符的时长
    else:
        # beats_per_measure = num
        beat_unit_duration = 60.0 / tempo * (4.0 / den)
    
    # bar_duration = beats_per_measure * beat_unit_duration
    
    global_pattern_dict_name = f"ACCOMPANIMENT_PATTERNS_{num}_{den}"
    pattern_dict = globals().get(global_pattern_dict_name)
    if pattern_dict is None:
        print(f"Warning: pattern dictionary {global_pattern_dict_name} not found, fallback to ACCOMPANIMENT_PATTERNS_4_4")
        pattern_dict = globals().get("ACCOMPANIMENT_PATTERNS_4_4")
    if pattern_name not in pattern_dict:
        print(f"Warning: pattern {pattern_name} not found in {global_pattern_dict_name}, fallback to 'pop_4_4'")
        pattern_name = "pop_4_4"
    pattern = pattern_dict[pattern_name]

    note_events = []

    for entry in pattern:
        if len(entry) == 2:
            beat_offset, note_position = entry
            duration_factor = 0.5
        else:
            beat_offset, note_position, duration_factor = entry

        event_on = start_time + beat_offset * beat_unit_duration
        event_off = event_on + duration_factor * beat_unit_duration

        pitches, velocities = pick_note_from_chord(
            chord_pcs_sorted,
            root_note,
            scale_mode,
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

def merge_measure(measure, time_signature=(4,4)):
    """
    合并连续相同音符，并确保小节时长符合 time_signature。
    """
    num, den = time_signature
    expected_duration = num / den  # 计算小节应有的总时长，例如 9/8 = 1.125

    if not measure:
        return []

    # **🚀 先转换 `measure` 为 (音符, 持续时间 count) 形式**
    converted_measure = []
    current = measure[0]
    count = 1

    for n in measure[1:]:
        if n == current:
            count += 1
        else:
            converted_measure.append((current, count))  # **✅ 记录当前音符**
            current = n
            count = 1

    converted_measure.append((current, count))  # **✅ 处理最后一个音符**

    # **✅ 确定基本单位**
    if den == 8 and num in (6, 9, 12):  
        base_unit = 1/8  # 复合拍 (6/8, 9/8, 12/8) 使用 1/8
    else:
        base_unit = 1/4  # 其他情况 (4/4, 3/4, 5/4) 使用 1/4

    # **✅ 计算小节总时长**
    total_duration = sum(count * base_unit for _, count in converted_measure)

    # **✅ 修正小节填充**
    while total_duration < expected_duration:
        converted_measure.append((converted_measure[-1][0], 1))
        total_duration += base_unit  # **填充音符，确保总时长符合小节**

    return converted_measure


def merge_measures(measures, time_signature=(4,4)):
    """
    处理所有小节的合并，并确保每小节符合 time_signature 的节拍要求。
    """
    return [merge_measure(m, time_signature) for m in measures]

def get_lilypond_duration(duration_factor):
    """
    将 duration_factor (如 1.5x, 1.0x, 0.75x) 转换为 LilyPond 的音符时值 (如 4, 8, 16)
    """
    if duration_factor >= 1.4:
        return "2"  # 二分音符
    elif duration_factor >= 1.0:
        return "4"  # 四分音符
    elif duration_factor >= 0.75:
        return "8"  # 八分音符
    else:
        return "16"  # 十六分音符
       


def duration_token(count, note_index, duration_curve=None, time_signature=(4,4)):
    """
    根据音符重复次数 count 生成 LilyPond 时值，适配复合拍 (6/8, 9/8, 12/8) 和 单拍 (4/4, 3/4, 2/4, 5/4)。
    """
    if duration_curve is not None:
        duration_factor = duration_curve[note_index]
        lily_duration = get_lilypond_duration(duration_factor)
        return lily_duration
    else:
        num, den = time_signature

        # **计算基本单位时值**
        base_duration = 1 / den  # 每个基本单位的时长，例如 8 表示 1/8 拍

        # **计算当前音符的总时长**
        total_duration = count * base_duration  # 真实音符时长

        # **区分 复合拍 (8 分音符为单位) vs. 简单拍 (4 分音符为单位)**
        if den == 8:  # **复合拍**
            mapping = {
                1/8: "8",   # 八分音符
                2/8: "4",   # 四分音符
                3/8: "4.",  # 点四分音符
                4/8: "2",   # 二分音符
                6/8: "2.",  # 点二分音符
                9/8: "2. 4.", # 9/8 = 点二分音符 + 点四分音符
                12/8: "2. 2."    # 12/8 = 全音符
            }
        else:  # **简单拍**
            mapping = {
                1/4: "4",   # 四分音符
                2/4: "2",   # 二分音符
                3/4: "2.",  # 点二分音符
                4/4: "1",   # 全音符
                5/4: "1 4"  # 5/4 = 全音符 + 1/4
            }

        # **直接匹配**
        if total_duration in mapping:
            return mapping[total_duration]

        # **如果没有直接匹配，递归分解**
        durations = []
        remaining_duration = total_duration

        while remaining_duration > 0:
            # **复合拍**
            if den == 8:
                if remaining_duration >= 9/8:
                    durations.append("2. 4.")  # **9/8**
                    remaining_duration -= 9/8
                elif remaining_duration >= 6/8:
                    durations.append("2.")
                    remaining_duration -= 6/8
                elif remaining_duration >= 4/8:
                    durations.append("2")
                    remaining_duration -= 4/8
                elif remaining_duration >= 3/8:
                    durations.append("4.")
                    remaining_duration -= 3/8
                elif remaining_duration >= 2/8:
                    durations.append("4")
                    remaining_duration -= 2/8
                elif remaining_duration >= 1/8:
                    durations.append("8")
                    remaining_duration -= 1/8
                else:
                    break
            else:  # **简单拍**
                if remaining_duration >= 3/4:
                    durations.append("2.")
                    remaining_duration -= 3/4
                elif remaining_duration >= 2/4:
                    durations.append("2")
                    remaining_duration -= 2/4
                elif remaining_duration >= 1/4:
                    durations.append("4")
                    remaining_duration -= 1/4
                else:
                    break

        return " ".join(durations)
    

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

def measures_to_lily_merged(merged_measures, duration_curve=None, time_signature=(4,4)):
    lines = []
    note_index = 0
    num, den = time_signature
    print(f"DEBUG: time signature = {num, den}")
    expected_duration = num / den  # 计算小节总时长
    
    for measure in merged_measures:
        tokens = []
        total_duration = 0  # 计算该小节的总时值
        
        for (midiv, count) in measure:
            duration_str = duration_token(count, note_index, duration_curve=duration_curve, time_signature=time_signature)

            # 直接计算 LilyPond 对应的时值
            # duration = count * (num / den)  # **更精确的时长计算方式**
            # ✅ LilyPond 音符时值
            duration_parts = duration_str.split()
            for d in duration_parts:
                duration = {
                    "1": 1, 
                    "2.": 3/4, 
                    "2": 1/2, 
                    "4.": 3/8, 
                    "4": 1/4, 
                    "8": 1/8
                }.get(d, 0)
                
                total_duration += duration  # **累加总时长**
            
            note_str = midi_to_lily_pitch(midiv) + duration_str  # **拼接 LilyPond 记谱**
            tokens.append(note_str)

            note_index += count
        
        # **打印当前小节的时长检查**
        print(f"[CHECK] 小节总时长: {total_duration} (期望: {expected_duration}) -> {'✅ 正确' if abs(total_duration - expected_duration) < 0.01 else '❌ 错误!'}")

        lines.append(" ".join(tokens) + " |")
    
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
    if os.path.exists(wav_file):  # 确保文件存在
        os.remove(wav_file)
        print(f"{wav_file} 已被删除")
    else:
        print(f"{wav_file} 不存在")

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
            if "arp_index" not in cyc_idx_dict:  
                cyc_idx_dict["arp_index"] = 0
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
    time_signature=(4,4),
    # beats=4,
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
    # print(deep_vec)
    
    # 固定随机种子，确保相同图像生成相同随机序列
    set_deterministic_seed(deep_vec)
    
    # c) 根据深度特征确定调性与节奏参数
    deep_root, deep_scale, deep_tempo = decide_deep_params(deep_vec)
    
    # 结合色彩映射（注意 color_to_tonality_new 返回三元组，此处只取第一个元素）
    # color_root = color_to_tonality(h_mean, s_mean, v_mean)
    color_root2, _, _ = color_to_tonality_new(h_mean, s_mean, v_mean)
    # final_root = color_root2 if random.random() < 0.5 else color_root
    # if random.random() < 0.5:
    #     final_root = deep_root
    print(deep_root, color_root2)
    final_root = color_root2
    
    # d) 构建 Markov 状态与和弦序列
    states = build_markov_states(deep_scale)
    transition = build_markov_transition(states, s_mean, v_mean)
    chord_seq = generate_chord_sequence(states, transition, length=length)
    print(f"[INFO] chord_seq= {chord_seq}, root= {final_root}, scale= {deep_scale}, tempo= {deep_tempo}")
    
    if deep_scale == "major":
        valid_scale_notes = set(sum(ALL_MAJOR_KEYS_CHORDS.get(final_root, {}).values(), []))
    else:
        valid_scale_notes = set(sum(ALL_MINOR_KEYS_CHORDS.get(final_root, {}).values(), []))

    beats_per_measure = time_signature[0]
    beat_unit_duration = 60.0 / deep_tempo * (4 / time_signature[1])
    bar_duration = beats_per_measure * beat_unit_duration

    # e) 根据 method 选择生成方式
    right_all = []
    left_all  = []
    
    if method == "dual":
        last_interval = None
        hist_notes = (None, None)
        for chord_label in chord_seq:
            chord_pcs = chord_pcs_in_scale(chord_label, final_root, deep_scale, use_scriabin=False)
            r_ms, l_ms, new_int, new_hist = generate_dual_voice_measure(
                chord_pcs, chord_label, final_root, deep_scale, time_signature=time_signature,
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
        current_time = 0.0
        cyc_dict = {}
        for chord_label in chord_seq:
            chord_pcs = chord_pcs_in_scale(chord_label, final_root, deep_scale, use_scriabin=False)
            # 右声部对位生成
            r_ms, _, new_int_r, new_hist_r = generate_dual_voice_measure(
                chord_pcs, chord_label, final_root, deep_scale, time_signature=time_signature,
                last_interval=last_interval_r,
                hist_notes=hist_notes_r
            )
            right_all.append(r_ms)
            last_interval_r = new_int_r
            hist_notes_r = new_hist_r

            
            # 左声部使用伴奏模式生成
            events = generate_accompaniment(
                chord_pcs_sorted=chord_pcs,
                root_note=final_root,
                scale_mode=deep_scale,
                pattern_name=pattern_name,
                start_time=current_time,
                time_signature=time_signature,
                tempo=deep_tempo,
                lowestBass=36,
                highestBass=60,
                velocityBase=80,
                cyc_idx_dict=cyc_dict
            )
            print(f"DEBUG: Generated events = {events}")
            current_time += bar_duration
            root_midi = NOTE_BASE_MAP.get(final_root, 60) + 12 * 4  # 选择当前调性的主音，并设定在 4 组八度
            measure_pitch_list = [root_midi] * beats_per_measure  # 使用正确的主音填充
            for (on_t, off_t, pitches, velocities) in events:
                rel_time = on_t - (current_time - bar_duration)
                beat_idx = int(rel_time // beat_unit_duration)
                if beat_idx < 0:
                    beat_idx = 0
                if beat_idx >= beats_per_measure:
                    beat_idx = beats_per_measure - 1
                if pitches:
                    # **确保音符符合调性**
                    if pitches[0] % 12 not in valid_scale_notes:
                        print(f"Warning: {pitches[0]} 不在 {final_root} {deep_scale}，可能需要检查！")
                    measure_pitch_list[beat_idx] = pitches[0]
            print(f"DEBUG: measure_pitch_list = {measure_pitch_list}")
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
    
    MIN_VELOCITY = 40  # 最低音量（对应 `\p`）
    MAX_VELOCITY = 100  # 最高音量（对应 `\f`）

    MIN_DURATION_FACTOR = 1.5  # 开始和结束的 duration 倍数
    NORMAL_DURATION_FACTOR = 1.0  # 中间部分的 duration
    MAX_DURATION_FACTOR = 1.5  # 结尾部分的 duration 倍数

    print(f"right_all structure: {right_all[:5]}")
    print(f"left_all structure: {left_all[:5]}")

    # 计算总音符数
    total_notes = len(right_all) * beats_per_measure

    # 生成音量变化曲线 (渐强 + 维持 + 渐弱)
    velocity_curve = []
    duration_curve = []
    for i in range(total_notes):
        if i < total_notes * 0.25:  # 🎼 前 25% 渐强
            velocity = MIN_VELOCITY + (MAX_VELOCITY - MIN_VELOCITY) * (i / (total_notes * 0.25))
            duration_factor = MIN_DURATION_FACTOR - (MIN_DURATION_FACTOR - NORMAL_DURATION_FACTOR) * (i / (total_notes * 0.25))
        elif i < total_notes * 0.75:  # 🎼 中间 50% 维持
            velocity = MAX_VELOCITY
            duration_factor = NORMAL_DURATION_FACTOR
        else:  # 🎼 后 25% 渐弱
            velocity = MAX_VELOCITY - (MAX_VELOCITY - MIN_VELOCITY) * ((i - total_notes * 0.75) / (total_notes * 0.25))
            duration_factor = NORMAL_DURATION_FACTOR + (MAX_DURATION_FACTOR - NORMAL_DURATION_FACTOR) * ((i - total_notes * 0.75) / (total_notes * 0.25))
        
        velocity_curve.append(int(velocity))  # 取整
        duration_curve.append(duration_factor)

    sec_per_beat = 60.0 / deep_tempo
    current_time = 0.0
 

    # 🎵 逐个音符设置 `velocity`
    note_index = 0
    
    sec_per_beat = 60.0 / deep_tempo
    current_time = 0.0

    # 归一化 deep_vec，使其映射到 0.5x ~ 2x 之间（更合理的范围）
    normalized_deep_vec = (deep_vec - np.min(deep_vec)) / (np.max(deep_vec) - np.min(deep_vec))  # 归一化到 0~1
    tempo_modifiers = 0.5 + normalized_deep_vec * 1.5  # 变换范围 0.5x ~ 2x

    # 使用滑动窗口方式（rolling window）让 1280 维特征影响所有音符
    num_notes = len(right_all) * 4  # 计算总音符数
    rolling_window_size = max(1, len(tempo_modifiers) // num_notes)  # 计算每个音符对应的窗口大小

    # 计算节奏权重，使其平滑过渡
    tempo_modifiers_resampled = np.convolve(tempo_modifiers, np.ones(rolling_window_size) / rolling_window_size, mode='same')

    merged_right = merge_measures(right_all, time_signature=time_signature)
    print("[DEBUG] merged_right:")
    for i, measure in enumerate(merged_right):
        print(f"小节 {i+1}: {measure}")
    merged_left = merge_measures(left_all, time_signature=time_signature)

    print("[DEBUG] merged_left:")
    for i, measure in enumerate(merged_left):
        print(f"小节 {i+1}: {measure}")

    note_index = 0
    current_time = 0.0

    for measure_i in range(len(merged_right)):  # ✅ 使用 merged_right 代替 right_all
        for (r_midi, count) in merged_right[measure_i]:  # ✅ 获取合并后的音符
            start_t = current_time
            base_duration = sec_per_beat  # 基础时长

            # 🎼 取当前音符的动态音量和时长
            velocity = velocity_curve[note_index]
            duration_factor = duration_curve[note_index]
            note_index += count  # ✅ 按照合并后的音符数更新索引

            # 🎵 计算音符结束时间（合并音符时考虑 count）
            end_t = current_time + base_duration * duration_factor * count

            # 🎹 生成 MIDI 音符
            nr = pretty_midi.Note(velocity=velocity, pitch=r_midi, start=start_t, end=end_t)

            # ✅ 打印 MIDI 右手音符信息
            # print(f"[MIDI] Right Hand: Pitch={nr.pitch} ({pretty_midi.note_number_to_name(nr.pitch)}), "
            #     f"Velocity={nr.velocity}, Duration={nr.end - nr.start:.3f}")

            right_instr.notes.append(nr)

            current_time = end_t  # ✅ 更新时间

    # 🎼 处理左手声部
    current_time = 0.0
    note_index = 0

    for measure_i in range(len(merged_left)):  # ✅ 使用 merged_left 代替 left_all
        for (l_midi, count) in merged_left[measure_i]:  # ✅ 获取合并后的音符
            start_t = current_time
            base_duration = sec_per_beat  # 基础时长

            # 🎼 取当前音符的动态音量和时长
            velocity = velocity_curve[note_index]
            duration_factor = duration_curve[note_index]
            note_index += count  # ✅ 按照合并后的音符数更新索引

            # 🎵 计算音符结束时间
            end_t = current_time + base_duration * duration_factor * count

            # 🎹 生成 MIDI 音符
            nl = pretty_midi.Note(velocity=velocity, pitch=l_midi, start=start_t, end=end_t)

            # ✅ 打印 MIDI 左手音符信息
            # print(f"[MIDI] Left Hand: Pitch={nl.pitch} ({pretty_midi.note_number_to_name(nl.pitch)}), "
            #     f"Velocity={nl.velocity}, Duration={nl.end - nl.start:.3f}")

            left_instr.notes.append(nl)

            current_time = end_t  # ✅ 更新时间


    right_instr.notes = merge_consecutive_notes(right_instr.notes)
    left_instr.notes = merge_consecutive_notes(left_instr.notes)

    pm.instruments.append(right_instr)
    pm.instruments.append(left_instr)
    pm.write(out_midi)
    from_midi_to_mp3(out_midi)

    convert_to_type0(pm, out_midi, left_program_index, right_program_index)

    # g) 生成 LilyPond 文本并转换为 PDF
    right_lily = measures_to_lily_merged(merged_right, duration_curve=None, time_signature=time_signature)
    left_lily = measures_to_lily_merged(merged_left, duration_curve=None, time_signature=time_signature)
    

    def attach_absolute_dynamic(measure_list, dynamic):
        """
        确保 `\p`, `\mp`, `\!` 这样的位置标记紧跟音符，而不是单独存在
        """
        for i, measure in enumerate(measure_list):
            tokens = measure.split()
            for j, token in enumerate(tokens):
                if token[-1].isdigit():  # 找到音符（如 d''4）
                    tokens[j] = f"{token}{dynamic}"  # 让动态标记紧贴音符
                    measure_list[i] = " ".join(tokens)
                    return measure_list
        return measure_list  # 没有音符，不修改

    def add_dyn(lily):
        # 1️⃣ **按小节拆分**
        measures = [m.strip() for m in lily.split("|") if m.strip()]
        total_measures = len(measures)

        if total_measures < 4:
            return lily  # 小节太少，不做动态处理

        # 2️⃣ **计算分段索引**
        seg1_count = max(1, total_measures // 4)  # 前 25%
        seg3_count = max(1, total_measures // 4)  # 后 25%
        seg2_count = total_measures - (seg1_count + seg3_count)  # 中间 50%

        # 3️⃣ **划分小节**
        seg1_measures = measures[:seg1_count]      # 前 25%
        seg2_measures = measures[seg1_count: seg1_count + seg2_count]  # 中间 50%
        seg3_measures = measures[-seg3_count:]     # 后 25%


        # 4️⃣ **修正 `\p` `\mp` 绑定音符**
        if seg1_measures:
            seg1_measures = attach_absolute_dynamic(seg1_measures, "\\p")  # `\p` 绑定音符
            seg1_measures = attach_absolute_dynamic(seg1_measures, "\\<")  # `\<` 绑定音符

        if seg2_measures:
            seg2_measures = attach_absolute_dynamic(seg2_measures, "\\!")  # `\!` 绑定音符

        if seg3_measures:
            seg3_measures = attach_absolute_dynamic(seg3_measures, "\>")  # `\>` 绑定音符
            seg3_measures = attach_absolute_dynamic(seg3_measures, " \! \\mp")  # `\mp` 绑定音符

        # 5️⃣ **重新拼接小节**
        final_lily = " | ".join(seg1_measures + seg2_measures + seg3_measures) + " |"

        return final_lily
    
    final_right_lily = add_dyn(right_lily)
    final_left_lily = add_dyn(left_lily)

    print(len(final_right_lily), len(final_left_lily))
    print(final_right_lily)
    print(final_left_lily)



    lily_root = convert_key_for_lily(final_root)
    print("[DEBUG] lily_root: ", lily_root, "deep_scale: ", deep_scale)

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
                \\time {time_signature[0]}/{time_signature[1]}
                % \\tempo 4={deep_tempo}
                {final_right_lily}
                \\bar "|."
            }}
            \\new Staff = "left" {{
                \\clef bass
                \\key {lily_root} \\{deep_scale}
                \\time {time_signature[0]}/{time_signature[1]}
                {final_left_lily}
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
    def parse_time_signature(s):
        try:
            # Remove any surrounding parentheses and whitespace
            s = s.strip().strip("()")
            # Split by comma
            num, den = s.split(',')
            return (int(num.strip()), int(den.strip()))
        except Exception as e:
            raise argparse.ArgumentTypeError("Time signature must be in the form (numerator, denominator), e.g., (6,8)") from e

    parser = argparse.ArgumentParser(
        description="Generate music from image using deep learning-based algorithm."
    )
    parser.add_argument("--img_path", type=str, required=True, help="Path to the input image.")
    parser.add_argument("--time_signature", type=parse_time_signature, default=(4,4), help="(numerator, denominator).")
    parser.add_argument("--length", type=int, default=24, help="Number of measures (chord sequence length).")
    parser.add_argument("--method", type=str, choices=["dual", "pattern"], default="dual",
                        help="Generation method: 'dual' for dual-voice counterpoint, 'pattern' for accompaniment pattern mode.")
    parser.add_argument("--pattern_name", type=str, default="pop_4_4",
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
        time_signature=args.time_signature,
        # beats=args.beats,
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