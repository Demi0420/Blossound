# music_gen.py

import pretty_midi
import random
import math
import numpy as np
import cv2
from sklearn.cluster import KMeans
from helpers import average_color, compute_block_complexity, find_closest_color, select_notes_from_block_with_kmeans, signed_normalize, convert_notes_tempo_velocity, extract_color_features, find_closest_tone_key, convert_note, velocity_to_dynamic, find_closest_tone_key_in_COLOR_MAP
from collections import Counter
import itertools


def note_to_midi(note, octave=4):
    """
    将音符（如 "C", "D", "E"）转换为 MIDI 音高。
    支持升号（#）和降号（b）。
    """
    note_map = {
        # 自然音
        "C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11,
        # 升音（#）
        "C#": 1, "D#": 3, "E#": 5, "F#": 6, "G#": 8, "A#": 10, "B#": 0,
        # 降音（b 或 - 号）
        "Cb": 11, "Db": 1, "D-": 1, "Eb": 3, "E-": 3, "Fb": 4, "F-": 4,
        "Gb": 6, "G-": 6, "Ab": 8, "A-": 8, "Bb": 10, "B-": 10
    }
    
    # 如果 note 中含有其它后缀（例如 "-b"），这里简单取其第一个部分
    note_clean = note.split('-')[0]  # 支持处理类似 "B-" 或 "A#-minor"
    if note_clean in note_map:
        midi_number = 12 * (octave + 1) + note_map.get(note_clean, 0)
        return midi_number
    else:
        raise ValueError(f"Invalid note: {note}")

def process_basic_unit(basic_unit):
    """
    对单个基本单元进行处理：
      1. 将该单元内 4 个区域（小节）内的所有音符（共 16 个）合并为一个列表，
      2. 随机打乱顺序，
      3. 扫描打乱后的序列，将相邻重复的音符合并为一个长音（时值累加）。
         每个音符原始时值为 1 拍，合并后的时值为连续相同音符的总拍数。
    返回最终处理后的音符列表，形式为 [(note, duration), ...]，总拍数应为 16。
    """
    # 合并所有区域的音符（每个区域 4 个音符，共 16 个）
    all_notes = []
    for measure in basic_unit["measures"]:
        all_notes.extend(measure)
    # 随机打乱顺序
    random.shuffle(all_notes)
    
    # 合并相邻重复音符
    final_notes = []
    if not all_notes:
        return final_notes
    current_note = all_notes[0]
    duration = 1  # 每个音符初始时值为 1 拍
    for note in all_notes[1:]:
        if note == current_note:
            duration += 1
        else:
            final_notes.append((current_note, duration))
            current_note = note
            duration = 1
    final_notes.append((current_note, duration))
    
    # 检查总拍数，理论上应为 16
    total_duration = sum(d for n, d in final_notes)
    if total_duration != 16:
        # 如果有偏差，可做比例调整（此处简单输出警告）
        print(f"警告：基本单元总拍数为 {total_duration}，而非预期 16。")
    return final_notes


def generate_midi_from_basic_units(basic_units, output_file, base_reference_tempo=120, tem_vel=True):
    """
    遍历所有基本单元，针对每个单元：
      - 调用 process_basic_unit 得到最终的音符序列（[(note, duration), ...]）
      - 将这些音符打上该单元的 tempo、velocity (即 (note, duration, tempo, velocity))
      - 输出该单元的最终音符（便于观察）
      - 将这些音符依次加入 MIDI 文件（按顺序衔接），velocity用于MIDI力度
      - 根据 (tempo / base_reference_tempo) 比例来缩放播放时长，以模拟快慢
    返回：final_notes_list
      - 这是一个二维列表，每个单元对应一个音符子列表：
        [[(note, duration, tempo, velocity), ...],  # 第1单元
         [(note, duration, tempo, velocity), ...],  # 第2单元
         ...
        ]
    """
    midi_object = pretty_midi.PrettyMIDI()
    current_time = 0.0
    final_notes_list = []

    for unit in basic_units:
        # 先得到当前单元的原始音符 (不含 tempo/velocity)
        final_notes = process_basic_unit(unit)  # => [(note_name, duration), ...]

        # 把 tempo 和 velocity 加到每条音符上
        # 例如 -> [(note_name, duration, tempo, velocity), ...]
        extended_notes = []
        if tem_vel:
            tempo = unit["tempo"]     
            velocity = unit["velocity"]
            
            for (note_name, raw_duration) in final_notes:
                extended_notes.append((note_name, raw_duration, tempo, velocity))

            # 将带tempo/velocity的音符列表保存到 final_notes_list
            final_notes_list.append(extended_notes)

        # MIDI生成过程
        instrument_name = unit["instrument"]
        try:
            instrument_program = pretty_midi.instrument_name_to_program(instrument_name)
        except ValueError:
            print(f"警告: {instrument_name} 不是有效的GM乐器名称，使用默认 Acoustic Grand Piano")
            instrument_program = pretty_midi.instrument_name_to_program("Acoustic Grand Piano")
        instrument = pretty_midi.Instrument(program=instrument_program, name=instrument_name)

        start_time = current_time
        for (note_name, raw_duration, adjusted_tempo, note_velocity) in extended_notes:
            # tempo_ratio 用于模拟当前单元的快慢
            tempo_ratio = adjusted_tempo / base_reference_tempo
            # print(tempo_ratio)
            # 根据 raw_duration 再 /4 (如果你原本让 duration=4 表示1拍之类)，
            # 然后再除以 tempo_ratio
            scaled_duration = (raw_duration / 4.0) / tempo_ratio
            # print(scaled_duration)

            pitch = note_to_midi(note_name)
            note = pretty_midi.Note(
                velocity=note_velocity,
                pitch=pitch,
                start=start_time,
                end=start_time + scaled_duration
            )
            instrument.notes.append(note)
            start_time += scaled_duration

        midi_object.instruments.append(instrument)
        current_time = start_time

    midi_object.write(output_file)
    print(f"MIDI 文件已生成: {output_file}")
    return final_notes_list



def generate_lilypond_single_staff(measures_4tuple, octave, output_filename="score.ly"):
    """
    将 measures_4tuple (二维列表:[[(note, duration, tempo, velocity), ...], ...])
    输出到 LilyPond 文件中，仅用单个 Staff(高音谱表\clef treble)。
    
    - 不分左右手
    - 可在每小节开头插入 tempo 变换
    - 可在每小节的第一个音符后插入力度标记(\mf等)
    - 小节末尾插 "|"
    - 末尾插 \\bar "|."
    """

    lily_str_parts = []
    last_tempo = None

    for measure_idx, notes_in_measure in enumerate(measures_4tuple):
        if not notes_in_measure:
            continue
        # 取第一个音符的 tempo / velocity
        first_tempo = notes_in_measure[0][2]
        first_velocity = notes_in_measure[0][3]

        # 如果 tempo 变了，就插入 \tempo 4=xx
        if last_tempo is None or not math.isclose(first_tempo, last_tempo, rel_tol=1e-7):
            tempo_int = int(round(first_tempo))
            lily_str_parts.append(f"\\tempo 4={tempo_int}")
            last_tempo = first_tempo

        # 逐个音符转 LilyPond
        for i, (note, dur, tempo, velocity) in enumerate(notes_in_measure):
            note_str = convert_note(note, dur, octave)
            if i == 0:
                # 第一个音符后面加力度标记
                dyn_mark = velocity_to_dynamic(velocity)
                note_str += dyn_mark
            lily_str_parts.append(note_str)

        # 小节结束加个 "|"
        lily_str_parts.append("|")

    # 最后加粗线
    lily_str_parts.append("\\bar \"|.\"")

    # 整合成字符串
    staff_notes_str = " ".join(lily_str_parts)

    # 拼 LilyPond 文件内容
    lilypond_content = f"""
        \\version "2.24.1"
        \\header {{
        title = "Single Staff Score"
        composer = "Test"
        }}

        \\score {{
        \\new Staff {{
            \\clef treble
            {staff_notes_str}
        }}
        \\layout {{}}
        \\midi {{}}
        }}
        """
    # 写到文件
    import os
    os.makedirs(os.path.dirname(output_filename), exist_ok=True)
    with open(output_filename, "w", encoding="utf-8") as f:
        f.write(lilypond_content)
    print(f"LilyPond 文件已生成: {output_filename}")

def process_basic_units(image, color_to_tone_map, color_tone_dict):
    """
    将图像按 25×25 的网格划分，
    每个 4×4 格子作为一个基本单元，在该单元内：
      - 取中心2×2区域计算平均颜色，进而确定调性（tone_mapping）
      - 对四个边缘区域（各 2×2 格子）调用 select_notes_from_block_with_kmeans 得到音符（每个区域4个音符）
      - [修改] 先收集每块的 complexity_score (可能为负数), 存入 basic_unit
      - [修改] 二次遍历: 根据所有 block 的 min/max score, 将其归一化到 [-1,1], 并计算 tempo/velocity
    返回包含 tempo/velocity 的 basic_units。
    """
    grid_size = 25
    height, width, _ = image.shape
    rows = height // grid_size
    cols = width // grid_size

    # 先分割图像到 25×25 的 cells
    grids = []
    for r in range(rows):
        row_grids = []
        for c in range(cols):
            cell = image[r*grid_size:(r+1)*grid_size, c*grid_size:(c+1)*grid_size, :]
            row_grids.append(cell)
        grids.append(row_grids)

    # =====================
    # 1) 第一阶段: 收集全部 basic_unit, 包含 complexity_score，但暂不做 tempo/velocity
    # =====================
    # 先计算整个图像本身的 complexity_score (供后面做差?), 
    # 原代码: complexity_score = compute_block_complexity(full_block_img) - total_complexity_score
    # 这里先预计算 total_complexity_score:
    total_complexity_score = compute_block_complexity(image)
    
    basic_units = []
    all_scores = []  # 用于存放所有 4×4 块的分数

    for r in range(0, rows - 3, 4):
        for c in range(0, cols - 3, 4):
            # 取 4×4 块
            block = [[grids[r+i][c+j] for j in range(4)] for i in range(4)]
            
            def combine_cells(cells):
                return np.vstack([np.hstack(row) for row in cells])
            
            # =============== 获取中心颜色 -> tone_key ===============
            center_cells = [[block[1][1], block[1][2]],
                            [block[2][1], block[2][2]]]
            center_img = combine_cells(center_cells)
            center_avg = average_color(center_img)
            center_rgb = center_avg[::-1]
            center_hex = '#{:02X}{:02X}{:02X}'.format(
                int(center_rgb[0]), int(center_rgb[1]), int(center_rgb[2])
            )

            tone_key = find_closest_color(center_hex, color_tone_dict)
            tone_mapping = color_to_tone_map.get(tone_key, color_to_tone_map["C-b"])
            scale = tone_mapping["scale"]
            base_tempo = tone_mapping["tempo"]
            key_val = tone_mapping["key"]
            instrument_name = tone_mapping["base_instrument"]
            
            # =============== 四个边缘区域 -> 音符 ===============
            # 左上
            tl_cells = [[block[0][0], block[0][1]],
                        [block[1][0], block[1][1]]]
            tl_img = combine_cells(tl_cells)
            tl_melody = select_notes_from_block_with_kmeans(tl_img, scale)
            
            # 右上
            tr_cells = [[block[0][2], block[0][3]],
                        [block[1][2], block[1][3]]]
            tr_img = combine_cells(tr_cells)
            tr_melody = select_notes_from_block_with_kmeans(tr_img, scale)
            
            # 左下
            bl_cells = [[block[2][0], block[2][1]],
                        [block[3][0], block[3][1]]]
            bl_img = combine_cells(bl_cells)
            bl_melody = select_notes_from_block_with_kmeans(bl_img, scale)
            
            # 右下
            br_cells = [[block[2][2], block[2][3]],
                        [block[3][2], block[3][3]]]
            br_img = combine_cells(br_cells)
            br_melody = select_notes_from_block_with_kmeans(br_img, scale)
            
            unit_measures = [tl_melody, tr_melody, bl_melody, br_melody]
            
            # =============== 计算 4×4 块自身的 complexity_score ===============
            full_block_img = combine_cells(block)
            c_score = compute_block_complexity(full_block_img) - total_complexity_score
            # print("complexity_score:", c_score)

            # 不立刻计算 tempo/velocity, 先存 complexity_score
            basic_unit = {
                "position": (r*grid_size, c*grid_size, 4*grid_size, 4*grid_size),
                "tone_mapping": tone_mapping,
                "instrument": instrument_name,
                "key": key_val,
                "scale": scale,
                "base_tempo": base_tempo,
                "complexity_score": c_score,    # 暂存
                "measures": unit_measures
            }
            basic_units.append(basic_unit)
            all_scores.append(c_score)

    # =====================
    # 2) 计算全局最小值/最大值
    # =====================
    score_min = min(all_scores) if all_scores else 0.0
    score_max = max(all_scores) if all_scores else 0.0
    
    # print("Global complexity_score range:", score_min, "to", score_max)

    # =====================
    # 3) 第二阶段: 根据 [-1,1] 映射，计算 tempo & velocity
    # =====================
    for unit in basic_units:
        c = unit["complexity_score"]
        # 归一化到 [-1,1] => norm_val
        norm_val = signed_normalize(c, score_min, score_max)
        
        # 例如 tempo 在 base_tempo 基础上做一定上下浮动
        # 这里给个示例：当 norm_val=1 => tempo = base_tempo+20; norm_val=-1 => tempo= base_tempo-20
        # norm_val=0 => base_tempo
        base_tempo = unit["base_tempo"]
        # adjusted_tempo = base_tempo + 20 * norm_val
        if norm_val > 0:
            max_increase = 90 - base_tempo  # 最大可增加量
            adjusted_tempo = base_tempo + max_increase * norm_val
        else:
            max_decrease = base_tempo - 40  # 最大可减少量
            adjusted_tempo = base_tempo + max_decrease * norm_val

        # 确保最终值在 [40, 90] 之间
        adjusted_tempo = max(40, min(90, adjusted_tempo))
        
        # 力度 [80, 120] 之间上下浮动 20
        # norm_val=1 => velocity=120; norm_val=-1 => velocity=80; norm_val=0 => 100
        # 下面做一个线性映射:  velocity = 100 + 20 * norm_val
        # 这样 norm_val=±1 => velocity= [80,120],  norm_val=0 => 100
        velocity_raw = 100 + 10 * norm_val
        # MIDI 力度须是 int
        adjusted_velocity = int(round(velocity_raw))

        # 存回 unit
        unit["tempo"] = adjusted_tempo
        unit["velocity"] = adjusted_velocity

    return basic_units

def process_basic_units_new(image, TONE_TO_SCALE, COLOR_MAP, grid_size=25):
    """
    将图像按 25×25 像素划分为小格，每 4×4 个小格构成一个基本单元。
    对于每个基本单元：
      1. 取中心 2×2 区域计算平均颜色，利用 find_closest_tone_key 在 COLOR_MAP 中确定 tone_key。
      2. 根据 tone_key 从 TONE_TO_SCALE 中获取对应的 scale（以及 tempo、instrument 等）。
      3. 对基本单元内四个边缘 2×2 区域调用 select_notes_from_block_with_kmeans，
         并传入该 scale 生成音符（每个区域返回 4 个音符）。
      4. [可选] 计算当前 4×4 图像块的复杂度（边缘+纹理），动态调整 tempo 与 velocity。
      5. 拼装成含 "position"、"tempo"、"velocity"、"instrument"、"key"、"measures" 等字段的 basic_unit。
    返回所有单元组成的列表 basic_units。
    """
    height, width, _ = image.shape
    rows = height // grid_size
    cols = width // grid_size

    # 先将图像切分为 grid_size×grid_size 的小格
    grids = []
    for r in range(rows):
        row_cells = []
        for c in range(cols):
            cell = image[r*grid_size:(r+1)*grid_size, c*grid_size:(c+1)*grid_size, :]
            row_cells.append(cell)
        grids.append(row_cells)

    # 辅助函数：将指定区域的小格合并为一个图像块
    def combine_cells(r0, c0, nr, nc):
        rows_list = []
        for i in range(nr):
            row_cells = [grids[r0 + i][c0 + j] for j in range(nc)]
            rows_list.append(np.hstack(row_cells))
        return np.vstack(rows_list)

    basic_units = []

    for r in range(0, rows - 3, 4):
        for c in range(0, cols - 3, 4):
            # 1) 整个 4×4 基本单元（100×100 像素）
            unit_block = combine_cells(r, c, 4, 4)

            # 2) 中心 2×2 区域（位置：(r+1, c+1) 至 (r+2, c+2)），提取颜色 -> tone_key
            center_block = combine_cells(r + 1, c + 1, 2, 2)
            center_features = extract_color_features(center_block)
            # find_closest_tone_key 返回 (tone_key, diff, matched_hex, matched_ratio) 等
            tone_key, _, _, _ = find_closest_tone_key(center_features, COLOR_MAP)

            # 3) 根据 tone_key 从 TONE_TO_SCALE 获取音阶信息
            if TONE_TO_SCALE and tone_key in TONE_TO_SCALE:
                scale_info = TONE_TO_SCALE[tone_key]
                base_scale = scale_info["scale"]
                base_tempo = scale_info["tempo"]
                instrument_name = scale_info["instrument"]
            else:
                # 如果找不到对应音阶，就给一个默认值
                base_scale = ["C", "D-", "E-", "F", "G-", "A-", "B-", "C"] 
                base_tempo = 75
                instrument_name = "Acoustic Grand Piano"

            # 4) 四个边缘 2×2 区域：分别提取音符
            tl_block = combine_cells(r, c, 2, 2)     # 左上
            tr_block = combine_cells(r, c + 2, 2, 2) # 右上
            bl_block = combine_cells(r + 2, c, 2, 2) # 左下
            br_block = combine_cells(r + 2, c + 2, 2, 2) # 右下

            left_tl = select_notes_from_block_with_kmeans(tl_block, base_scale)
            left_tr = select_notes_from_block_with_kmeans(tr_block, base_scale)
            left_bl = select_notes_from_block_with_kmeans(bl_block, base_scale)
            left_br = select_notes_from_block_with_kmeans(br_block, base_scale)

            # 每个单元 4 个区域，每个区域 4 个音符 => 共 16 个音符
            unit_measures = [left_tl, left_tr, left_bl, left_br]

            # 5) [可选] 计算图像复杂度，用于动态调整 tempo & velocity
            # complexity_score = compute_block_complexity(unit_block)
            # 简单截断到 [0, 10]
            # complexity_clamped = min(max(complexity_score, 0.0), 10.0)
            # normalized = complexity_clamped / 10.0

            # normalized = scale_complexity_log(complexity_score, max_score=2000)

            # 假设当 normalized=1 时，比原 tempo +10，力度从80~120浮动
            # adjusted_tempo = base_tempo + 20 * normalized
            # adjusted_velocity = int(80 + (120 - 80) * normalized)

            # 6) 拼装出与原版类似的 basic_unit
            # 这里 key 用 tone_key，tempo/velocity 则是微调后的值
            basic_unit = {
                "position": (r * grid_size, c * grid_size, 4 * grid_size, 4 * grid_size),
                "tone_mapping": {
                    "scale": base_scale,
                    # "tempo": adjusted_tempo,
                    "base_instrument": instrument_name
                },
                "instrument": instrument_name,
                "tempo": base_tempo,
                "velocity": 50,
                "key": tone_key,
                "measures": unit_measures
            }
            basic_units.append(basic_unit)
    
    return basic_units

    

# 3. 将一个音符列表转换为 LilyPond 字符串
def convert_notes(notes):
    """
    将一个 (音符, 时值) 列表转换为 LilyPond 格式字符串，各个音符之间以空格分隔。
    """
    return " ".join([convert_note(n, d) for n, d in notes])



def convert_measures_with_hairpin(measures, fade_in=False, fade_out=False, octave=4):
    """
    将 measures (二维列表) 转换为 LilyPond 字符串，每小节末尾加 "|",
    并在音符后面插入渐强/渐弱标记，以避免“Unattached”警告。
    
    参数:
      - measures: 形如 [ [(note, dur, tempo, vel), ...],  [(note, dur, tempo, vel), ...] ]
      - fade_in: 若 True，在 第1小节第1音符 后追加 "\ppp\<"，在 最后小节最后音符 后追加 "\!\mf"
      - fade_out: 若 True，在 第1小节第1音符 后追加 "\f\>"，在 最后小节最后音符 后追加 "\!\p"
      - octave: 默认 4 表示 c' = C4
    """
    parts = []
    measure_count = len(measures)
    
    from music_gen import convert_note  # 或你自己已有的函数

    for m_idx, measure in enumerate(measures):
        measure_str_list = []
        for n_idx, (note, dur, tempo, vel) in enumerate(measure):
            note_str = convert_note(note, dur, octave)

            # 1) 如果是 fade_in, 在"首个音符"后面插 "\ppp\<"
            if fade_in and (m_idx == 0 and n_idx == 0):
                note_str += "\\ppp\\<"
            
            # 2) 如果是 fade_out, 在"首个音符"后面插 "\f\>"
            if fade_out and (m_idx == 0 and n_idx == 0):
                note_str += "\\mf\\>"

            # 3) 如果是 fade_in, 在"最后小节最后音符"后面插 "\!\mf"
            if fade_in and (m_idx == measure_count - 1 and n_idx == len(measure) - 1):
                note_str += "\\!\\mf"

            # 4) 如果是 fade_out, 在"最后小节最后音符"后面插 "\!\p"
            if fade_out and (m_idx == measure_count - 1 and n_idx == len(measure) - 1):
                note_str += "\\!\\ppp"

            measure_str_list.append(note_str)

        # 一小节结束，加 "|"
        joined_measure = " ".join(measure_str_list)
        parts.append(joined_measure + " |")

    # 将多个小节拼接成一段 LilyPond 代码
    return " ".join(parts)


def generate_lilypond_score_with_brace(intro_notes_4tuple, right_notes_4tuple, coda_notes_4tuple, right_octave,
                                       left_notes_4tuple, left_octave,
                                       filename):
    """
    根据:
      - right_notes_4tuple: 右手 [(note, duration, tempo, velocity), ...]
      - left_notes_4tuple:  左手 [(note, duration, tempo, velocity), ...]
    生成带钢琴括号的 LilyPond 乐谱文件 (treble+bass) 并在遇到新 tempo 时插入 \\tempo 标记。
    
    文件末尾加 \\bar "|." 用粗线结束符。
    """
    # 1) 转换右手
    # print(right_notes_4tuple)
    right_lily_str = convert_notes_tempo_velocity(right_notes_4tuple, octave=right_octave, insert_tempo=False, insert_velco=False)
    # right_lily_str += " \\bar \"|.\""  # 末尾加粗线
    intro_lily_str = convert_measures_with_hairpin(intro_notes_4tuple, fade_in=True)
    coda_lily_str  = convert_measures_with_hairpin(coda_notes_4tuple, fade_out=True)

    
    # 2) 转换左手
    left_lily_str = convert_notes_tempo_velocity(left_notes_4tuple, octave=left_octave, insert_tempo=False, insert_velco=False)
    # left_lily_str += " \\bar \"|.\""
    
    # 3) 拼装 LilyPond 文件内容
    lilypond_content = f"""\\version "2.24.1"
        \\header {{
        title = "Sheet Music"
        % composer = "Yao."
        }}

        \\score {{
        % 使用钢琴连谱号 (PianoStaff)
        \\new PianoStaff <<
            \\new Staff = "right" {{
            \\clef treble
            {intro_lily_str}

            {right_lily_str}

            {coda_lily_str}

            \\bar "|."
            }}
            \\new Staff = "left" {{
            \\clef bass
            % 让左手整体音量更低
            \\set Staff.midiMinimumVolume = #0.2
            \\set Staff.midiMaximumVolume = #0.5
            {left_lily_str}
            \\bar "|."
            }}
        >>
        \\layout {{}}
        \\midi {{}}
        }}
        """
    # 4) 写入文件
    with open(filename, "w", encoding="utf-8") as f:
        f.write(lilypond_content)
    print(f"LilyPond 文件已生成: {filename}")


def group_notes_into_measures(flat_notes, beats_per_measure=4.0):
    """
    将一维的音符列表 (note, duration, tempo, velocity) 按小节拆分，要求每小节时值总和为 beats_per_measure (默认4拍)。
    如果遇到时值之和恰好凑满4拍，则开始下一小节。
    
    flat_notes: 形如 [
        ('D', 0.75, 100, 100),
        ('G', 0.25, 100, 100),
        ...
    ]
    返回:
    [
      [ (note, duration, tempo, velocity), ... ],  # measure1 (时值之和=4.0)
      [ (note, duration, tempo, velocity), ... ],  # measure2
      ...
    ]
    如果某段时值总和超出或无法整除4.0，则视情况处理(可扩展).
    """
    
    measures = []
    current_measure = []
    current_sum = 0.0
    
    for item in flat_notes:
        note, duration, tempo, velocity = item
        
        current_measure.append(item)
        current_sum += duration
        
        # 用一个小阈值来判断是否等于4拍(如|current_sum-4|<1e-9)
        if abs(current_sum - beats_per_measure) < 1e-9:
            # 当前小节够4拍，保存到 measures
            measures.append(current_measure)
            # 准备下一小节
            current_measure = []
            current_sum = 0.0
        elif current_sum > beats_per_measure + 1e-9:
            # 如果意外超过4拍，可以决定抛异常或进行额外处理
            raise ValueError(f"小节时值超过 {beats_per_measure} 拍: {current_sum}")
    
    # 如果遍历结束后还剩下零头(没有凑够4拍)，可视需求决定如何处理
    if current_measure:
        # 如果你确定每段一定要凑满4拍，则这里也可能 raise Error
        # 也可以把残余的音符视作最后一小节(不足4拍)
        measures.append(current_measure)
    
    return measures



def process_basic_units_from_top_colors(image, COLOR_MAP, TONE_TO_SCALE, 
                                        num_colors=30, 
                                        notes_per_color=16):
    """
    根据整张图片提取前 num_colors 种主色，然后为每种颜色找到对应的 tone_label，
    从 TONE_TO_SCALE 中获取 scale, tempo, instrument 等信息，并生成一个 basic_unit 结构。
    
    参数:
      - image: 整张图片 (BGR, numpy 数组)
      - COLOR_MAP: { tone_label: [hex1, hex2, ...], ... }
      - TONE_TO_SCALE: { tone_label: { "scale": [...], "tempo":..., "instrument":...}, ... }
      - num_colors: 需要提取的颜色数量 (默认为 30)
      - notes_per_color: 为每个颜色生成多少个音符 (示例 8)
    
    返回:
      - basic_units_new: 类似 [
          {
            "tone_label": ...,
            "tempo": ...,
            "instrument": ...,
            "measures": [[note1, note2, ...], ...],  # 你也可以把 8 个音符做成 2 小节 × 4 音符
            ...
          },
          ...
        ]
    """
    # 1) 提取前 num_colors 种主色
    #    可以简单地用 KMeans(n_clusters=num_colors) 或统计像素频次
    #    这里示例: KMeans
    pixels = image.reshape(-1, 3)
    kmeans = KMeans(n_clusters=num_colors, n_init="auto", random_state=42).fit(pixels)
    centers = kmeans.cluster_centers_  # shape: (num_colors, 3)
    labels, counts = np.unique(kmeans.labels_, return_counts=True)
    
    # 对聚类中心按像素计数排序
    sorted_idx = np.argsort(-counts)
    top_centers = centers[sorted_idx]  # 取前 num_colors
    
    # 2) 将 BGR -> RGB -> HEX
    def bgr_to_hex(bgr_arr):
        bgr_int = [int(x) for x in bgr_arr]
        rgb = bgr_int[::-1]
        return '#{:02X}{:02X}{:02X}'.format(*rgb)
    
    top_hex_colors = [ bgr_to_hex(top_centers[i]) for i in range(num_colors) ]
    
    # 3) 为每个 hex_color 找到最相近的 tone_label
    #    先把 hex_color -> lab，再比对 COLOR_MAP 里的 tone_label => hex_list
    basic_units_new = []
    
    for i, hex_color in enumerate(top_hex_colors):
        tone_label = find_closest_tone_key_in_COLOR_MAP(hex_color, COLOR_MAP)
        # tone_label = e.g. "C-b", "G-", etc.
       
        if tone_label in TONE_TO_SCALE:
            scale_info = TONE_TO_SCALE[tone_label]
            scale = scale_info["scale"]
            tempo = scale_info["tempo"]
            instrument = scale_info["instrument"]
        else:
            scale = ["C", "D", "E", "F", "G", "A", "B"]
            tempo = 80
            instrument = "Acoustic Grand Piano"
        
        # 4) 生成音符(内在标准)
        #    示例: 随机从 scale 抽 notes_per_color 个音符
        #    每个音符默认 "1 拍" (raw_duration=1)
        #    你可改成 4 音符 × 4 measures, etc.
        chosen_notes = []
        for _ in range(notes_per_color):
            note_name = random.choice(scale)
            chosen_notes.append(note_name)
        
        # 把这些音符打包成 4 小节(每小节 4 个音符) => measures
        measures = []
        measure_size = 4
        for idx in range(0, notes_per_color, measure_size):
            measure_notes = chosen_notes[idx: idx+measure_size]
            if measure_notes:
                measures.append(measure_notes)
        
        new_unit = {
            "tone_label": tone_label,
            "tempo": tempo,
            "velocity": 50,
            "instrument": instrument,
            "hex_color": hex_color, 
            "measures": measures 
        }
        basic_units_new.append(new_unit)
    
    return basic_units_new


from collections import Counter
import itertools

def extract_main_melody_by_repetition(grouped_measures, top_n=16):
    """
    对 grouped_measures (二维列表) 所有音符做频率统计，
    选出出现次数最多的 top_n 个音符，依次排列成若干小节(4个音符/小节)。
    
    返回类似:
      [
        [(note, dur, tempo, vel), ...], # measure1
        [ ... ],                       # measure2
        ...
      ]
    
    这里演示：把 duration/tempo/velocity 都简单地用同一个值(1拍, tempo=80, velocity=70),
    或可改用其他策略(比如随机从原音符中继承).
    """
    # Flatten
    all_notes = list(itertools.chain(*grouped_measures))  # => [(note, dur, tempo, vel), ...]
    
    # 1) 统计 note 出现次数
    note_counts = Counter(note for (note, _, _, _) in all_notes)
    
    # 2) 取出现最多的 top_n 音符
    most_common_notes = note_counts.most_common(top_n)  # => [(note, freq), (note2, freq2), ...]
    # 只要 note，忽略 freq
    repeated_notes = [item[0] for item in most_common_notes]
    
    # 3) 组装成小节(每小节4个音符 => 4拍)
    #   这里简化：都设 duration=1.0, tempo=80, velocity=70, 仅演示
    #   你也可以在统计时把 tempo/vel 做平均
    measures = []
    measure_size = 4
    tempo = 80
    velocity = 70
    for i in range(0, len(repeated_notes), measure_size):
        measure_slice = repeated_notes[i:i+measure_size]
        measure_list = []
        for n in measure_slice:
            measure_list.append( (n, 1.0, tempo, velocity) )
        measures.append(measure_list)
    
    return measures


def create_intro_measures(scale_notes, 
                          num_measures=4,
                          total_notes=16,
                          start_velocity=40,
                          end_velocity=80,
                          ascending=True,
                          tempo=60):
    """
    生成intro小节:
      - num_measures 个小节, 共 total_notes 个音符
      - 力度从 start_velocity -> end_velocity 线性增(渐强)
      - 若 ascending=True, 则音高从 scale_notes[0]->[-1] (线性插值)
        else 反向
    """
    measures = []
    scale_len = len(scale_notes)
    if scale_len<2:
        # 以防万一
        scale_notes = ["C", "D", "E", "F", "G", "A", "B", "C"]
        scale_len = len(scale_notes)

    # 线性分配 note indices
    def lerp(a, b, t):
        return a + (b - a)*t

    note_indices = []
    for i in range(total_notes):
        t = i/(total_notes-1) if total_notes>1 else 0
        if ascending:
            idx_f = lerp(0, scale_len-1, t)
        else:
            idx_f = lerp(scale_len-1, 0, t)
        idx_i = int(round(idx_f))
        if idx_i<0: idx_i=0
        if idx_i>=scale_len: idx_i=scale_len-1
        note_indices.append(idx_i)
    
    velocity_step = (end_velocity - start_velocity)/float(total_notes-1) if total_notes>1 else 0
    
    current_idx = 0
    notes_per_measure = total_notes // num_measures  # 例如8音符/2小节 =>4音符每小节
    remainder = total_notes % num_measures
    # (如果有余数,可分配到最后一小节,仅演示)

    for m in range(num_measures):
        measure_size = notes_per_measure + (1 if m<(remainder) else 0)
        measure_list = []
        for i in range(measure_size):
            note_i = note_indices[current_idx]
            v = int(round(start_velocity + velocity_step*current_idx))
            measure_list.append( ( scale_notes[note_i], 1.0, tempo, v ) )
            current_idx+=1
        measures.append(measure_list)
    return measures


def create_intro_basic_measures(
    scale_notes,  # 一组音名，如 ["C","D","E","F","G","A","B"]
    num_measures=4,
    total_notes=16,
    start_velocity=40,
    end_velocity=80,
    start_tempo=60,
    ascending=True
):
    """
    生成 intro 段，使音量(velocity)从 start_velocity 到 end_velocity 线性增长。
    如果 ascending=True，则音高从 scale_notes[0]->scale_notes[-1]。
    每音符(时值=1拍) 为简化示例。
    
    返回: 
      [
        [(note, dur=1.0, tempo, velocity), ...],  # measure1
        [(...), ...],                             # measure2
      ]
    """
    measures = []
    scale_len = len(scale_notes)
    if scale_len < 2:
        scale_notes = ["C","D","E","F","G","A","B"]
        scale_len = len(scale_notes)
    
    # 线性插值函数
    def lerp(a, b, t):
        return a + (b - a)*t
    
    # 分配 "note indices"
    note_indices = []
    for i in range(total_notes):
        t = i/(total_notes-1) if total_notes>1 else 0
        if ascending:
            idx_f = lerp(0, scale_len-1, t)
        else:
            idx_f = lerp(scale_len-1, 0, t)
        idx_i = int(round(idx_f))
        if idx_i < 0: idx_i=0
        if idx_i >= scale_len: idx_i=scale_len-1
        note_indices.append(idx_i)

    # 分配 "velocity" 线性
    velocity_values = []
    for i in range(total_notes):
        t = i/(total_notes-1) if total_notes>1 else 0
        v = lerp(start_velocity, end_velocity, t)
        velocity_values.append(int(round(v)))
    
    # 组装小节
    notes_per_measure = total_notes // num_measures  # 每小节音符数
    remainder = total_notes % num_measures
    current_idx = 0
    for m in range(num_measures):
        measure_size = notes_per_measure + (1 if m<remainder else 0)
        measure_list = []
        for i in range(measure_size):
            note_i = note_indices[current_idx]
            vel = velocity_values[current_idx]
            # tempo 固定 start_tempo(示例); 你也可加 tempo 线性变化
            measure_list.append( (scale_notes[note_i], 1.0, start_tempo, vel) )
            current_idx += 1
        measures.append(measure_list)
    return measures



def create_coda_basic_measures(
    scale_notes,
    num_measures=4,
    total_notes=16,
    start_velocity=80,
    end_velocity=40,
    start_tempo=60,
    descending=True
):
    """
    类似create_intro_basic_measures，但力度从start_velocity -> end_velocity(渐弱),
    音名从scale_notes[-1]->scale_notes[0]若descending=True.
    """
    # 直接调用前面函数, 只是不再 ascending & 力度反向
    return create_intro_basic_measures(
        scale_notes=scale_notes,
        num_measures=num_measures,
        total_notes=total_notes,
        start_velocity=start_velocity,
        end_velocity=end_velocity,
        start_tempo=start_tempo,
        ascending=not descending  # coda一般下行
    )

def get_top_notes(grouped_measures, top_count=10):
    """
    返回出现次数最多的 top_count 个音名(不含dur/tempo/vel).
    """
    all_notes = list(itertools.chain(*grouped_measures))  # flatten
    note_counter = Counter(n for (n, _, _, _) in all_notes)
    most_common = note_counter.most_common(top_count)
    top_notes = [item[0] for item in most_common]
    return top_notes


def create_empty_measures(num_measures, tempo=80, velocity=0):
    """
    生成 num_measures 个空小节, 
    每小节4拍(4个音符), 用 'R' 或 'rest' 之类, 并 velocity=0
    """
    empties = []
    for _ in range(num_measures):
        measure = []
        for i in range(4):
            measure.append(("R", 1.0, tempo, velocity))
        empties.append(measure)
    return empties



def extract_top_notes_and_durations(grouped_measures, note_count=10, dur_count=5):
    """
    从 grouped_measures (二维list) 中提取最常见的 note (音名) 和最常见的 dur (时值),
    分别返回 notes_list, durations_list.
    """
    # Flatten
    all_notes = list(itertools.chain(*grouped_measures))  # => [(note, dur, tempo, vel), (note, dur, ...), ...]

    # 1) 统计音名出现次数
    note_counter = Counter(n for (n,d,t,v) in all_notes)
    top_notes_data = note_counter.most_common(note_count)  # [(note, freq), ...]
    top_notes = [item[0] for item in top_notes_data]

    # 2) 统计时值出现次数
    dur_counter = Counter(d for (n,d,t,v) in all_notes)
    top_durs_data = dur_counter.most_common(dur_count)    # [(dur, freq), ...]
    top_durs = [item[0] for item in top_durs_data]

    return top_notes, top_durs



def create_4x4_measures_from_top(note_list, dur_list, seed=42, 
                                 total_measures=4, 
                                 tempo=80, 
                                 velocity=70):
    """
    随机组合音名+时值, 使每小节合计4拍, 生成 total_measures 个小节。
    - 使用固定随机种子保证复现。
    - 如果挑到的dur会超出4拍,尝试其他dur或截断。
    
    返回形如:
      [
        [(note, dur, tempo, velocity), (note, dur, tempo, velocity),...],  # measure1 sum=4
        [ ... ],  # measure2 sum=4
        ...
      ]
    """
    random.seed(seed)
    measures = []

    for m in range(total_measures):
        measure_notes = []
        sum_beats = 0.0
        while abs(sum_beats - 4.0) > 1e-9:  # 还没凑到4拍
            # 如果已接近4拍(例如 3.5) 并且 dur_list里大多数dur>0.5 可能超拍, 
            # 就多尝试几次
            picked_note = random.choice(note_list)
            picked_dur  = random.choice(dur_list)

            # 如果 sum_beats + picked_dur <=4.0 => 直接用
            if sum_beats + picked_dur <= 4.0 + 1e-9:
                measure_notes.append( (picked_note, picked_dur, tempo, velocity) )
                sum_beats += picked_dur
            else:
                # 超拍 -> 尝试换个更小dur, 如果仍不行, 截断
                smaller_found = False
                for _ in range(5):  # 最多尝试5次
                    alt_dur = random.choice(dur_list)
                    if sum_beats + alt_dur <=4.0 + 1e-9:
                        measure_notes.append( (picked_note, alt_dur, tempo, velocity) )
                        sum_beats += alt_dur
                        smaller_found = True
                        break
                if not smaller_found:
                    # 截断: 让最后一音符时值= (4 - sum_beats)
                    remain = 4.0 - sum_beats
                    if remain<1e-9:
                        # 表示已经凑满(浮点毛刺?), break
                        break
                    measure_notes.append( (picked_note, remain, tempo, velocity) )
                    sum_beats+= remain

        measures.append(measure_notes)
    return measures


def create_intro_basic_from_grouped(grouped_basic_notes):
    # 1) 提取最常见 10 个音名, 5 个时值
    top_notes, top_durs = extract_top_notes_and_durations(grouped_basic_notes, note_count=10, dur_count=5)
    # 2) 随机组合 => 4 小节, each 4拍
    intro_measures = create_4x4_measures_from_top(
        note_list=top_notes,
        dur_list=top_durs,
        seed=42,
        total_measures=4,
        tempo=80,
        velocity=60  # 例如稍弱
    )
    return intro_measures


def adjust_intro_for_seamless_connection(intro_measures, main_right):
    """
    将 intro_measures 的最后一个音符
    替换成 main_right 第一个音符的 (note, tempo, velocity) 以保证音高/力度/tempo 衔接。
    duration 可以保留 intro 原先的 or 也可改用 main 的。
    """
    if not intro_measures or not main_right:
        return intro_measures  # 空就不处理
    
    # 1) 找到 main_right 第一个音符
    first_note_of_main = main_right[0][0]  # => (note, dur, tempo, vel)
    (main_note, main_dur, main_tempo, main_vel) = first_note_of_main
    
    # 2) 找到 intro_measures 最后一个音符
    last_measure = intro_measures[-1]
    last_note_idx = len(last_measure) - 1
    (intro_note, intro_dur, intro_tempo, intro_vel) = last_measure[last_note_idx]
    
    # 3) 替换: pitch, tempo, velocity => 与 main 的第一音符相同
    #   但保留 original  duration?
    new_duration = intro_dur  # 也可以改成 main_dur or (intro_dur+main_dur)/2
    last_measure[last_note_idx] = (main_note, new_duration, main_tempo, main_vel)
    return intro_measures


def adjust_coda_for_seamless_connection(coda_measures, main_right):
    """
    将 coda_measures 的第一个音符替换成 main_right 最后一个音符的 (note, tempo, velocity)。
    然后让 coda 后续音符逐渐延长时值 => create a '拉长'效果
    """
    if not coda_measures or not main_right:
        return coda_measures
    
    # 1) 找到 main_right 最后一个音符
    last_note_of_main = main_right[-1][-1]
    (main_note, main_dur, main_tempo, main_vel) = last_note_of_main
    
    # 2) 替换 coda_measures 第一个音符
    first_measure = coda_measures[0]
    if first_measure:
        # 先替换 pitch, tempo, velocity
        (coda_note, coda_dur, coda_tempo, coda_vel) = first_measure[0]
        # 保留 original coda_dur? or take main_dur?
        new_duration = coda_dur
        first_measure[0] = (main_note, new_duration, main_tempo, main_vel)
    
    # 3) 让 coda 里后面音符逐步延长 => e.g. linearly map durations
    #    具体设计: first note保留, 后面音符从 scale=1.0 ~ 2.0...
    #    这里只是演示
    coda_durations = []
    max_factor = 3.0  # 终点时值变3倍?
    
    # flatten
    all_coda_notes = []
    for meas in coda_measures:
        all_coda_notes.extend(meas)
    total_coda_count = len(all_coda_notes)
    if total_coda_count > 1:
        max_factor = 3.0
        for i, (nt, dr, tm, vl) in enumerate(all_coda_notes):
            t = i/(total_coda_count-1)
            factor = 1.0 + (max_factor - 1.0)*t
            new_dr = dr*factor
            # ---- 关键：round掉，以免出现 1.3690… ----
            new_dr = round(new_dr, 2)
            all_coda_notes[i] = (nt, new_dr, tm, vl)
    
    
    # 再按原结构写回 coda_measures
    idx=0
    for meas_idx, meas in enumerate(coda_measures):
        for note_idx in range(len(meas)):
            coda_measures[meas_idx][note_idx] = all_coda_notes[idx]
            idx+=1

    return coda_measures


def make_coda_44_from_raw(coda_measures, main_right):
    """
    1) flatten coda_measures -> all_coda_notes
    2) 可做拉长 / velocity渐弱 / or 改写第一个音符
    3) 用 create_measures_4_4(...) 重新组装成 4/4 measure
    4) adjust_coda_for_seamless_connection => 让 coda第1音符=main末
    5) 返回 coda_measures_44
    """
    # 1) flatten
    all_coda_notes = []
    for meas in coda_measures:
        all_coda_notes.extend(meas)

    # 2) 可做时值拉长(示例: factor 1~2)
    total_count = len(all_coda_notes)
    if total_count>1:
        for i, (nt, dr, tm, vl) in enumerate(all_coda_notes):
            t = i/(total_count-1)
            factor = 1.0 + (2.0-1.0)*t  # 1→2
            new_dr = dr*factor
            # 也可 round(new_dr,2)
            new_dr = round(new_dr,2)
            all_coda_notes[i] = (nt, new_dr, tm, vl)

    # 3) 也可先把第一个音符 => main_right[-1][-1] for seamless
    #    (实际上 adjust_coda_for_seamless_connection 也可以做)
    last_note_main = main_right[-1][-1]
    # (main_note, main_dur, main_tm, main_vel) = last_note_main
    # all_coda_notes[0] = (main_note, all_coda_notes[0][1], main_tm, main_vel)

    # 4) "create_measures_4_4" 需要“可能notes & durs” OR
    #    直接写一个 "group_into_44" 函数把all_coda_notes按4拍分组 => coda_measures_44
    coda_measures_44 = group_notes_into_44(all_coda_notes, epsilon=1e-3)

    # 5) 让coda第一个音符= main末 => adjust_coda_for_seamless_connection
    coda_measures_44 = adjust_coda_for_seamless_connection(coda_measures_44, main_right)

    return coda_measures_44

def group_notes_into_44(note_list, epsilon=1e-4):
    """
    将一维音符列表(可能时值不合4拍)按4拍分割 => measure2D
    原理: 不断累加dur, 超过4 => 截断 => 下一小节
    """
    measures = []
    current_measure = []
    sum_beats = 0.0

    for (nt,dr,tm,vl) in note_list:
        remainder = 4.0 - sum_beats
        if dr < remainder + epsilon:
            # 不超拍
            current_measure.append((nt,dr,tm,vl))
            sum_beats+= dr
        else:
            # 超拍 => 截断 or leftover
            cut = remainder
            if cut>0:
                current_measure.append((nt, cut, tm, vl))
                sum_beats += cut
            measures.append(current_measure)
            # start new measure
            current_measure = []
            sum_beats = 0.0

            # 剩余 = dr - cut
            leftover = dr - cut
            if leftover>epsilon:
                # 递归 or 直接再处理 leftover?
                # simplest => create new measure if leftover>4 => multi-split
                # 这里简单处理
                # note with leftover => check if leftover>4 => etc.
                if leftover < 4.0 + epsilon:
                    # this leftover note fill next measure
                    new_measure = []
                    new_measure.append((nt,leftover,tm,vl))
                    sum_beats = leftover
                    current_measure = new_measure
                else:
                    # leftover>4 => do multiple
                    # omitted for brevity
                    pass

    # 把最后 current_measure
    if current_measure:
        # 若sum_beats<4 => check if < epsilon => merge leftover
        diff = 4.0 - sum_beats
        if diff>epsilon and current_measure:
            # 并给最后音符
            last_note, last_dr, last_tm, last_vl = current_measure[-1]
            current_measure[-1] = (last_note, last_dr+diff, last_tm, last_vl)
            sum_beats+= diff
        measures.append(current_measure)

    return measures

def flatten_coda_measures(coda_measures):
    all_notes = []
    for meas in coda_measures:
        all_notes.extend(meas)
    return all_notes

def generate_lilypond_with_intro_coda(intro_measures, main_right, coda_measures, right_octave, left_str, left_octave, filename):
    # 1) 衔接intro => main
    intro_measures = adjust_intro_for_seamless_connection(intro_measures, main_right)
    # 2) 衔接main => coda
    coda_measures  = adjust_coda_for_seamless_connection(coda_measures, main_right)
    
    # 1) 先 flatten + do whatever expansions
    all_coda_notes = flatten_coda_measures(coda_measures)

    # 2) group into 4/4 measures
    coda_measures_44 = group_notes_into_44(all_coda_notes, epsilon=1e-3)

    # 3) adjust coda => let first note= main_right last
    coda_measures_44 = adjust_coda_for_seamless_connection(coda_measures_44, main_right)

    # 4) hairpin fade_out
    coda_str = convert_measures_with_hairpin(coda_measures_44, fade_out=True, octave=right_octave)

    # 3) convert => lily string
    intro_str = convert_measures_with_hairpin(intro_measures, fade_in=True, octave=right_octave)
    # coda_str  = convert_measures_with_hairpin(coda_measures, fade_out=True, octave=right_octave)
    # main melody => 你已有: convert_measures_with_hairpin(main_right, fade_in=False, fade_out=False)
    # 这里假设 main_right -> main_str (already have)
    main_str = convert_notes_tempo_velocity(main_right, octave=right_octave, insert_tempo=False, insert_velco=False)

    # 4) 组装 right staff
    right_lily = f"{intro_str} {main_str} {coda_str} \\bar \"|.\""
    left_lily = convert_notes_tempo_velocity(left_str, octave=left_octave, insert_tempo=False, insert_velco=False)


    # left_str 保持不变 or 也加 coda?
    lilypond_content = f"""
        \\version "2.22.1"
        \\header {{
        title = "Sheet Music"
        composer = "Your Name"
        }}
        \\score {{
        \\new PianoStaff <<
            \\new Staff = "right" {{
            \\clef treble
            {right_lily}
            }}
            \\new Staff = "left" {{
            \\clef bass
            {left_lily}
            \\bar "|."
            }}
        >>
        \\layout {{}}
        \\midi {{}}
        }}
        """
    with open(filename,"w",encoding="utf-8") as f:
        f.write(lilypond_content)
    print(f"LilyPond 文件已生成: {filename}")


def create_coda_measures(
    note_list,             # top_notes
    dur_list,              # top_durs
    seed=999,
    total_measures=4,      # coda小节数
    base_tempo=80,
    base_velocity=80,
    main_last_note=None,   # (note, dur, tempo, velocity) 供衔接用
    epsilon=1e-4
):
    """
    生成一个 Coda 段(4/4 * total_measures 小节)，
    1) 第一个小节第一个音符 => 衔接 main_last_note(音高/tempo/velocity相同)
    2) 在后续音符中让时值逐渐变长(如从1拍逐渐增大到2拍或4拍)
    3) 每小节总拍数约等于4.0(±epsilon)，避免 LilyPond "barcheck failed"
    4) 如果遇到时值截断/拼合，自动处理，避免 ValueError

    返回 measures => [ [ (note,dur,tempo,vel), ... ], ... ] (4个小节)
    """

    random.seed(seed)
    
    measures = []
    # 若你想做一个"渐长"的最大时值序列, 
    # 比如 measure i => max_factor = 1*(2^i) => measure0=1, measure1=2, measure2=4, measure3=8
    # 这里先举例 measure i => factor=2^i (0-based)

    for m_idx in range(total_measures):
        measure_notes = []
        sum_beats = 0.0
        
        # 计算当前小节 "max_dur_factor"
        # 例如 measure0 => factor=1, measure1 =>2, measure2 =>4, measure3 =>8
        max_dur_factor = 2 ** m_idx

        # “渐长”并不一定是马上从 0.5变到4.0, 
        # 这里设置: "有效dur" = random_dur * max_dur_factor(但不超过4?).
        
        while True:
            remainder = 4.0 - sum_beats
            if remainder <= epsilon:
                # 已接近4拍 => break
                break
            
            # 若是第1小节 & 第1音符 => 用 main_last_note
            if (m_idx==0) and (abs(sum_beats)<1e-9) and (main_last_note is not None):
                # 衔接
                (mn_note, mn_dur, mn_tempo, mn_vel) = main_last_note
                # 但dur不一定要mn_dur, 这里可保留, 也可自定义, 先保留:
                used_dur = min(remainder, mn_dur)
                # tempo,velocity 保持一致
                measure_notes.append((mn_note, used_dur, mn_tempo, mn_vel))
                sum_beats += used_dur
            else:
                # 随机 pick
                picked_note = random.choice(note_list)
                base_d = random.choice(dur_list)
                
                # 让时值做 "max_dur_factor"倍, 并 round
                scaled_d = base_d * max_dur_factor
                scaled_d = round(scaled_d, 2)  # 避免浮点难拆
                
                if scaled_d < epsilon:
                    scaled_d = epsilon
                
                if sum_beats + scaled_d < 4.0 - epsilon:
                    # 放进去
                    measure_notes.append((picked_note, scaled_d, base_tempo, base_velocity))
                    sum_beats += scaled_d
                else:
                    # 超拍 => 截断
                    leftover = remainder
                    if leftover>epsilon:
                        # note with leftover
                        measure_notes.append((picked_note, leftover, base_tempo, base_velocity))
                        sum_beats += leftover
                    # 结束该小节
                    break
        
        # 小节结束,若 sum_beats<4 => 并到最后
        diff = 4.0 - sum_beats
        if diff>epsilon and measure_notes:
            last_note, last_dur, last_tm, last_vl = measure_notes[-1]
            measure_notes[-1] = (last_note, last_dur+diff, last_tm, last_vl)
            sum_beats += diff
        
        measures.append(measure_notes)
    
    return measures