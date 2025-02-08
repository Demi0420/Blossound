# main.py
import os
import sys
import cv2 # type: ignore
import subprocess
import itertools
from config import COLOR_TONE_DICT, COLOR_TO_TONE_MAP, TONE_TO_SCALE, COLOR_MAP, BASIC_OCTAVE, LEFT_OCTAVE
from music_gen import generate_midi_from_basic_units, generate_lilypond_score_with_brace, process_basic_units, process_basic_units_new, group_notes_into_measures, generate_lilypond_single_staff, process_basic_units_from_top_colors, extract_main_melody_by_repetition, get_top_notes, create_intro_measures, create_empty_measures, create_intro_basic_measures, create_coda_basic_measures, extract_top_notes_and_durations, create_4x4_measures_from_top, create_coda_measures

def from_midi_to_mp3(midi_file):
    """
    将 MIDI 文件转换为 WAV/MP3 的逻辑示例
    假设你通过 fluidsynth + ffmpeg/ffmpeg 来实现
    """
    wav_file = midi_file.replace(".midi", ".wav")
    mp3_file = midi_file.replace(".midi", ".mp3")

    # 1) 用 fluidsynth 生成 WAV
    subprocess.run([
        "fluidsynth",
        "-ni",
        "/usr/share/sounds/sf2/FluidR3_GM.sf2",  # 确保这里是正确的 SoundFont 路径
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



def main():
    if len(sys.argv) < 2:
        print("用法: python3 main.py <image_path>")
        sys.exit(1)
    
    image_path = sys.argv[1] # e.g. figures/<unique_id>/<filename>
    if not os.path.isfile(image_path):
        print(f"图像文件不存在: {image_path}")
        sys.exit(1)
    
    # 读取图像
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("无法读取图像，请检查路径。")
    
    # 获取 <unique_id> 与 <filename>
    # 假设 image_path 形如 "figures/73bd00a7-3042-4bb3-b2e2-dc8124184b61/IMG_8148.jpg"
    parts = image_path.split("/")
    if len(parts) >= 3:
        # parts[0] = "figures", parts[1] = <unique_id>, parts[2] = <filename>
        unique_id = parts[1]
    else:
        unique_id = "default"

    # 使用 os.path.basename 获取文件名，如 "IMG_8148.jpg"
    filename = os.path.basename(image_path)
    # 使用 os.path.splitext 分离扩展名，得到 "IMG_8148"
    image_name, _ = os.path.splitext(filename)

    # 构造输出目录: outputs/<unique_id>/<image_name>/
    out_dir = os.path.join("outputs", unique_id, image_name)
    os.makedirs(out_dir, exist_ok=True)
    abs_out_dir = os.path.abspath(out_dir)

    target_height = 500
    ratio = target_height / image.shape[0]
    new_width = int(image.shape[1]*ratio)
    image = cv2.resize(image, (new_width,target_height))

    # 1) 处理图像 -> basic_units
    basic_units = process_basic_units(image, COLOR_TO_TONE_MAP, COLOR_TONE_DICT) 
    # 2) 生成 MIDI
    basic_midi_output = os.path.join(out_dir, "basic.mid")
    basic_notes_list = generate_midi_from_basic_units(basic_units, basic_midi_output)

    flattened_basic_notes = list(itertools.chain(*basic_notes_list))
    basic_notes = [(note, duration / 4.0, t, v) for (note, duration, t, v) in flattened_basic_notes]
    grouped_basic_notes = group_notes_into_measures(basic_notes, beats_per_measure=4.0)


    # 3) 生成左手音符数
    left_units = process_basic_units_new(image, TONE_TO_SCALE, COLOR_MAP, grid_size=25)
    # 根据处理后的基本单元生成 MIDI 文件，并输出最终音符序列
    left_midi_output = os.path.join(out_dir, "left.mid")
    left_notes_list = generate_midi_from_basic_units(left_units, left_midi_output)

    flattened_left_notes = list(itertools.chain(*left_notes_list))
    left_notes = [(note, duration / 4.0, t, v) for (note, duration, t, v) in flattened_left_notes]
    grouped_left_notes = group_notes_into_measures(left_notes, beats_per_measure=4.0)


    left_units_new = process_basic_units_from_top_colors(image, COLOR_MAP, TONE_TO_SCALE, num_colors=30, notes_per_color=16)
    left_midi_output_new = os.path.join(out_dir, "left_new.mid")
    left_notes_new_list = generate_midi_from_basic_units(left_units_new, left_midi_output_new)

    flattened_left_notes_new = list(itertools.chain(*left_notes_new_list))
    left_notes_new = [(note, duration / 4.0, t, v) for (note, duration, t, v) in flattened_left_notes_new]
    grouped_left_notes_new = group_notes_into_measures(left_notes_new, beats_per_measure=4.0)
    # 如果 left 长于 right => 截断
    if len(grouped_left_notes_new) > len(grouped_basic_notes):
        grouped_left_notes_new = grouped_left_notes_new[:len(grouped_basic_notes)]
    # 如果 left 小于 right => 加空小节
    elif len(grouped_left_notes_new) < len(grouped_basic_notes):
        diff = len(grouped_basic_notes) - len(grouped_left_notes_new)
        extra_empty = create_empty_measures(diff)
        grouped_left_notes_new += extra_empty



    # 1) 抽取主体
    # 2) scale_for_intro : 10最重要音符
    # scale_for_intro = get_top_notes(grouped_basic_notes, top_count=10)
    intro_end_vel = grouped_basic_notes[0][0][3]
    intro_tempo   = grouped_basic_notes[0][0][2]

    top_notes, top_durs = extract_top_notes_and_durations(grouped_basic_notes, note_count=10, dur_count=5)
    intro_measures = create_4x4_measures_from_top(
        note_list=top_notes,
        dur_list=top_durs,
        seed=42,
        total_measures=4,
        tempo=intro_tempo,
        velocity=intro_end_vel  
    )

    # 4) coda_right
    coda_start_vel = grouped_basic_notes[-1][-1][3]
    coda_tempo     = grouped_basic_notes[-1][-1][2]


    top_notes, top_durs = extract_top_notes_and_durations(grouped_basic_notes, note_count=10, dur_count=5)
    main_last_note = grouped_basic_notes[-1][-1]
    coda_measures = create_coda_measures(
        note_list=top_notes,
        dur_list=top_durs,
        seed=999,
        total_measures=4,
        base_tempo=coda_tempo,
        base_velocity=coda_start_vel,
        main_last_note=main_last_note
    )

    # 4) 合并

    # full_right = intro_measures + grouped_basic_notes + coda_measures

    num_intro_measures = len(intro_measures)
    num_coda_measures  = len(coda_measures)

    intro_left  = create_empty_measures(num_intro_measures, tempo=80, velocity=0)
    coda_left   = create_empty_measures(num_coda_measures,  tempo=80, velocity=0)

    full_left = intro_left + grouped_left_notes + coda_left
    full_left_new = intro_left + grouped_left_notes_new + coda_left
    
    print(len(grouped_basic_notes), len(grouped_left_notes), len(grouped_left_notes_new))
    

    # 4) 生成 LilyPond
    # ly_filename = os.path.join("outputs", image_name, os.path.splitext(os.path.basename(image_path))[0] + "-score.ly")
    # LilyPond 文件 => outputs/<unique_id>/<image_name>-score.ly
    ly_filename = os.path.join(abs_out_dir, f"{image_name}-score.ly")
    # ly_filename = f"{image_name}-score.ly"
    generate_lilypond_score_with_brace(intro_measures, 
                                       grouped_basic_notes, 
                                       coda_measures, 
                                       BASIC_OCTAVE, 
                                       full_left, 
                                       LEFT_OCTAVE, 
                                       filename=ly_filename)
                

    # 5) 调用 LilyPond 和 ImageMagick
    output_base = os.path.join(abs_out_dir, f"{image_name}-score")  # e.g. outputs/<unique_id>/IMG_8148-score
    # output_base = f"{image_name}-score"
    subprocess.run([
        "lilypond",
        "-o", output_base,     # 输出基名
        ly_filename            # 输入的 .ly 文件
    ], check=True)


    # LilyPond 会生成 PDF/MIDI，文件名分别为 output_base+".pdf" / output_base+".midi"
    midi_file = output_base + ".midi"
    from_midi_to_mp3(midi_file)

    pdf_file = output_base + ".pdf"
    png_file = output_base + ".png"
    subprocess.run(
        ["convert", 
        "-density", "300", 
        pdf_file, 
        png_file
    ],check=True)


    ly_filename_new = os.path.join(abs_out_dir, f"{image_name}-score-new.ly")
    # ly_filename_new = f"{image_name}-score-new.ly"
    generate_lilypond_score_with_brace(intro_measures, 
                                       grouped_basic_notes, 
                                       coda_measures, 
                                       BASIC_OCTAVE, 
                                       full_left_new, 
                                       LEFT_OCTAVE, 
                                       filename=ly_filename_new)
                

    output_base_new = os.path.join(abs_out_dir, f"{image_name}-score-new") 
    # output_base_new = f"{image_name}-score-new"
    subprocess.run([
        "lilypond",
        "-o", output_base_new,     
        ly_filename_new         
    ], check=True)


    midi_file_new = output_base_new + ".midi"
    from_midi_to_mp3(midi_file_new)

    pdf_file_new = output_base_new + ".pdf"
    png_file_new = output_base_new + ".png"
    subprocess.run(
        ["convert", 
        "-density", "300", 
        pdf_file_new, 
        png_file_new
    ],check=True)



if __name__=="__main__":
    main()